from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from transformers.activations import ACT2FN

try:
    # transformers v5: dispatch decorator that picks between
    #   grouped_mm (torch._grouped_mm, Hopper+)
    #   batched_mm (bmm-based, any GPU)
    #   eager      (the class's own forward — our fallback, see below)
    # and allows custom backends (e.g. NPU fused MoE kernel) via
    # ALL_EXPERTS_FUNCTIONS.register("<name>", fn) + setting
    # config._experts_implementation = "<name>".
    from transformers.integrations.moe import use_experts_implementation
except ImportError:  # transformers v4 — no dispatch system, we just run our forward
    use_experts_implementation = None

from ..registry import register_ffn


class _SharedMLP(nn.Module):
    """Shared expert inside the MoE block — SwiGLU with its own intermediate_size.

    Param names match Qwen3_5MoeMLP(config, intermediate_size=shared_expert_intermediate_size).
    """

    def __init__(self, config) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.shared_expert_intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class _Experts(nn.Module):
    """Grouped experts as 3D weight tensors — byte-identical to HF Qwen3_5MoeExperts.

    Weights:
      gate_up_proj : [num_experts, 2*intermediate, hidden]
      down_proj    : [num_experts, hidden, intermediate]

    Forward dispatch:
      On transformers v5, this class is wrapped by
      ``@use_experts_implementation`` (see end of file), which routes ``forward``
      through ``ALL_EXPERTS_FUNCTIONS`` based on ``config._experts_implementation``:

        * ``grouped_mm`` (default on modern GPU): torch._grouped_mm fused kernel
        * ``batched_mm``: bmm-based batched path, available on any GPU
        * ``eager`` / unavailable backend: falls through to the forward below.

      The forward below is a byte-identical port of HF's ``Qwen3_5MoeExperts.forward``
      (per-expert F.linear loop with ``index_add_``). On NPU — where grouped_mm
      cannot dispatch and HF auto-falls-back to "eager" — both the HF reference and
      alloy run this same per-expert loop, so fp32 is bit-exact.

      Custom backends (including a future NPU fused MoE) can be plugged in without
      modifying alloy:

          from transformers.integrations.moe import ALL_EXPERTS_FUNCTIONS
          ALL_EXPERTS_FUNCTIONS.register("npu_fused_moe", my_fn)
          config._experts_implementation = "npu_fused_moe"
    """

    def __init__(self, config) -> None:
        super().__init__()
        self.num_experts = config.num_experts
        self.hidden_dim = config.hidden_size
        self.intermediate_dim = config.moe_intermediate_size
        self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts, 2 * self.intermediate_dim, self.hidden_dim))
        self.down_proj = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim, self.intermediate_dim))
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(
        self,
        hidden_states: torch.Tensor,  # [N, H]
        top_k_index: torch.Tensor,    # [N, K]
        top_k_weights: torch.Tensor,  # [N, K]
    ) -> torch.Tensor:
        """Eager-path: byte-identical port of HF Qwen3_5MoeExperts.forward.

        Iterates hit experts, applies each expert's F.linear to its routed tokens,
        accumulates with index_add_. Op-for-op the same as HF's eager forward, so
        fp32 outputs match HF bit-exactly when both fall back to "eager".
        """
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = F.one_hot(top_k_index, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            if expert_idx == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            gate, up = F.linear(current_state, self.gate_up_proj[expert_idx]).chunk(2, dim=-1)
            current_hidden_states = self.act_fn(gate) * up
            current_hidden_states = F.linear(current_hidden_states, self.down_proj[expert_idx])
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

        return final_hidden_states

    def _alloy_init_weights(self, init_std: float) -> None:
        # gate_up_proj and down_proj are bare 3D Parameters (not nn.Linear);
        # they aren't covered by the parent traversal's stdlib branches.
        nn.init.normal_(self.gate_up_proj, mean=0.0, std=init_std)
        nn.init.normal_(self.down_proj, mean=0.0, std=init_std)


if use_experts_implementation is not None:
    # Attach HF's dispatch layer. Adds self.has_gate / has_bias / is_transposed
    # (all matching our layout), self.config, and a forward shim that routes
    # through ALL_EXPERTS_FUNCTIONS. The class object is the same afterwards.
    _Experts = use_experts_implementation(_Experts)


class _TopKRouter(nn.Module):
    """Top-K softmax router matching Qwen3_5MoeTopKRouter."""

    def __init__(self, config) -> None:
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_experts
        self.hidden_dim = config.hidden_size
        self.weight = nn.Parameter(torch.zeros(self.num_experts, self.hidden_dim))

    def forward(self, hidden_states: torch.Tensor):
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        router_logits = F.linear(hidden_states, self.weight)
        router_probs = F.softmax(router_logits, dtype=torch.float, dim=-1)
        router_top_value, router_indices = torch.topk(router_probs, self.top_k, dim=-1)
        router_top_value = router_top_value / router_top_value.sum(dim=-1, keepdim=True)
        router_top_value = router_top_value.to(router_logits.dtype)
        return router_logits, router_top_value, router_indices

    def _alloy_init_weights(self, init_std: float) -> None:
        # weight is a bare 2D Parameter, not nn.Linear's — own init.
        nn.init.normal_(self.weight, mean=0.0, std=init_std)


@register_ffn("qwen3_5_moe")
class Qwen35SparseMoE(nn.Module):
    """Sparse MoE with top-K routing, grouped experts, and a gated shared expert.

    Port of Qwen3_5MoeSparseMoeBlock. Submodule names (``gate``, ``experts``,
    ``shared_expert``, ``shared_expert_gate``) match the HF state_dict.
    """

    def __init__(self, config, layer_idx: int | None = None) -> None:
        super().__init__()
        del layer_idx
        self.gate = _TopKRouter(config)
        self.experts = _Experts(config)
        self.shared_expert = _SharedMLP(config)
        self.shared_expert_gate = nn.Linear(config.hidden_size, 1, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states_reshaped = hidden_states.view(-1, hidden_dim)

        shared_expert_output = self.shared_expert(hidden_states_reshaped)
        _, routing_weights, selected_experts = self.gate(hidden_states_reshaped)
        expert_output = self.experts(hidden_states_reshaped, selected_experts, routing_weights)

        shared_expert_output = torch.sigmoid(self.shared_expert_gate(hidden_states_reshaped)) * shared_expert_output
        expert_output = expert_output + shared_expert_output
        return expert_output.reshape(batch_size, sequence_length, hidden_dim)
