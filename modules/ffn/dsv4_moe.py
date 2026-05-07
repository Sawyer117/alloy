"""DeepSeek-V4 sparse MoE block — TopK and Hash routing variants.

DSV4 ships two MoE flavors selected per-layer by the ``mlp_layer_types``
config in HF; in alloy this maps to ``ffn_types[i]`` directly:

  * ``dsv4_moe``      — TopK-router MoE (learned argmax)
  * ``dsv4_hash_moe`` — Hash-router MoE (frozen ``tid2eid[input_ids]``
                        lookup decides which experts; learned gate scores
                        still weight their contributions)

Both variants share one :class:`DeepseekV4SparseMoeBlock` class —
``__init__`` branches on ``config.ffn_types[layer_idx]`` to construct
either :class:`DeepseekV4TopKRouter` or :class:`DeepseekV4HashRouter` as
``self.gate``. The forward dispatches accordingly; hash variants need
``input_ids`` threaded in by the decoder layer (phase 1E wires this).

Source provenance: ported from
``references/dsv4/modeling_deepseek_v4.py`` lines 909-1035. Class
structure preserved including config field names: routers and experts
read ``config.n_routed_experts`` (DSV4's source-coupled name; HF
attribute_map'd to ``num_local_experts``), NOT alloy's qwen3.5-derived
``num_experts``. Each port stays faithful to its source's config
interface — alloy doesn't pretend two families' "expert count"
fields are the same field.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from transformers.activations import ACT2FN

# HF transformers v5+ ships an experts dispatch system that swaps in
# fused grouped_mm / batched_mm / NPU backends in place of the eager
# forward defined below. alloy's qwen3_5_moe applies the same pattern;
# match it here so DSV4 experts also get the fast path when available.
try:
    from transformers.integrations.moe import use_experts_implementation
except ImportError:
    use_experts_implementation = None

from ..registry import register_ffn


# =========================================================================== #
# Shared MLP (used by SparseMoeBlock as the always-on shared expert)
# =========================================================================== #


class DeepseekV4MLP(nn.Module):
    """Standard SwiGLU MLP used as DSV4's shared expert.

    Not registered as an alloy FFN type — it's an internal building block
    of :class:`DeepseekV4SparseMoeBlock`. Distinct from
    :class:`Qwen3MLP` only by the ``mlp_bias`` config field; matched
    here for source provenance even though math is the same when bias
    is off.
    """

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        bias = bool(getattr(config, "mlp_bias", False))
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=bias)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


# =========================================================================== #
# Packed experts (3D weights) with eager fallback forward
# =========================================================================== #


class _Experts(nn.Module):
    """Collection of expert weights stored as 3D tensors.

    Wrapped by :func:`use_experts_implementation` at the bottom of this
    module so the forward routes through ``ALL_EXPERTS_FUNCTIONS`` based
    on ``config._experts_implementation`` (``grouped_mm`` / ``batched_mm``
    / ``flash`` from binder / ``eager`` fallback to the body below).

    Eager forward applies an extra clamp to gate/up before SwiGLU — DSV4
    uses ``swiglu_limit`` to prevent overflow with the wider intermediate
    dim. Matches HF's ``_apply_gate`` convention so grouped_mm / batched_mm
    backends call back into the same gate function on top of their packed
    output.
    """

    def __init__(self, config) -> None:
        super().__init__()
        self.num_experts = config.n_routed_experts
        self.hidden_dim = config.hidden_size
        self.intermediate_dim = config.intermediate_size
        self.gate_up_proj = nn.Parameter(
            torch.empty(self.num_experts, 2 * self.intermediate_dim, self.hidden_dim)
        )
        self.down_proj = nn.Parameter(
            torch.empty(self.num_experts, self.hidden_dim, self.intermediate_dim)
        )
        self.act_fn = ACT2FN[config.hidden_act]
        self.limit = float(getattr(config, "swiglu_limit", 10.0))

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Per-expert eager loop. Byte-identical to HF reference when both
        sides take this fallback. Backends registered in
        ``ALL_EXPERTS_FUNCTIONS`` route through different paths.
        """
        final = torch.zeros_like(hidden_states)
        with torch.no_grad():
            mask = F.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)
            hit = torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx in hit:
            expert_idx = expert_idx[0]
            if expert_idx == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(mask[expert_idx])
            current = self._apply_gate(F.linear(hidden_states[token_idx], self.gate_up_proj[expert_idx]))
            current = F.linear(current, self.down_proj[expert_idx]) * top_k_weights[token_idx, top_k_pos, None]
            final.index_add_(0, token_idx, current.to(final.dtype))
        return final

    def _apply_gate(self, gate_up: torch.Tensor) -> torch.Tensor:
        """Clamped SwiGLU. Defined as a method so HF's grouped_mm / batched_mm
        backends apply the same clamp + activation on top of their packed
        gate_up output instead of bypassing it.
        """
        gate, up = gate_up.chunk(2, dim=-1)
        gate = gate.clamp(max=self.limit)
        up = up.clamp(min=-self.limit, max=self.limit)
        return self.act_fn(gate) * up

    def _alloy_init_weights(self, init_std: float) -> None:
        """gate_up_proj / down_proj are bare 3D Parameters — parent init
        traversal doesn't cover them, so we handle them explicitly here.
        """
        nn.init.normal_(self.gate_up_proj, mean=0.0, std=init_std)
        nn.init.normal_(self.down_proj, mean=0.0, std=init_std)


if use_experts_implementation is not None:
    DeepseekV4Experts = use_experts_implementation(_Experts)
else:
    DeepseekV4Experts = _Experts


# =========================================================================== #
# Routers (TopK + Hash)
# =========================================================================== #


class DeepseekV4TopKRouter(nn.Module):
    """Standard learned-argmax router with DSV4's score-correction bias.

    The ``e_score_correction_bias`` buffer is added to scores during the
    top-k selection (the *which-experts* decision) but NOT to the weights
    that go to experts (kept directly from the bias-free score). This is
    the DSV3-style trick to bias selection without contaminating the
    weighting signal.
    """

    def __init__(self, config) -> None:
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.n_routed_experts
        self.hidden_dim = config.hidden_size
        self.weight = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim))
        score_fn_name = getattr(config, "scoring_func", "sqrtsoftplus")
        self.score_fn = ACT2FN[score_fn_name]
        self.routed_scaling_factor = float(getattr(config, "routed_scaling_factor", 1.5))
        self.register_buffer(
            "e_score_correction_bias",
            torch.zeros(self.num_experts),
            persistent=True,
        )

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        flat = hidden_states.reshape(-1, self.hidden_dim)
        logits = F.linear(flat.float(), self.weight.float())
        scores = self.score_fn(logits)
        indices = torch.topk(
            scores + self.e_score_correction_bias, self.top_k, dim=-1, sorted=False
        ).indices
        weights = scores.gather(1, indices)
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
        return logits, weights * self.routed_scaling_factor, indices

    def _alloy_init_weights(self, init_std: float) -> None:
        nn.init.normal_(self.weight, mean=0.0, std=init_std)


class DeepseekV4HashRouter(nn.Module):
    r"""Hash routing for the first ``hash_moe`` layers (paper §2.1).

    Expert selection is a fixed ``tid2eid[input_ids]`` lookup — a frozen
    token-id → expert-id table — instead of a learned argmax. The learned
    gate ``weight`` still produces the per-expert scores that weight the
    selected experts' activations; only the *which-experts* decision is
    static per token id.

    Requires ``input_ids`` argument to forward — alloy's
    :class:`AlloyDecoderLayer` threads it through (phase 1E adds the
    plumbing) when the FFN type is ``dsv4_hash_moe``.
    """

    def __init__(self, config) -> None:
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.n_routed_experts
        self.hidden_dim = config.hidden_size
        self.weight = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim))
        score_fn_name = getattr(config, "scoring_func", "sqrtsoftplus")
        self.score_fn = ACT2FN[score_fn_name]
        self.routed_scaling_factor = float(getattr(config, "routed_scaling_factor", 1.5))
        self.register_buffer(
            "tid2eid",
            torch.zeros(config.vocab_size, self.top_k, dtype=torch.long),
            persistent=True,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        flat = hidden_states.reshape(-1, self.hidden_dim)
        logits = F.linear(flat.float(), self.weight.float())
        scores = self.score_fn(logits)
        indices = self.tid2eid[input_ids.reshape(-1)].long()
        weights = scores.gather(1, indices)
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
        return logits, weights * self.routed_scaling_factor, indices

    def _alloy_init_weights(self, init_std: float) -> None:
        # Don't touch tid2eid — it's a buffer, not a parameter, and is
        # populated externally (loaded from checkpoint).
        nn.init.normal_(self.weight, mean=0.0, std=init_std)


# =========================================================================== #
# SparseMoeBlock — registered under both alloy FFN names
# =========================================================================== #


@register_ffn("dsv4_moe")
@register_ffn("dsv4_hash_moe")
class DeepseekV4SparseMoeBlock(nn.Module):
    """DSV4 sparse MoE block: router + experts + always-on shared expert.

    Two FFN names registered → one class. ``__init__`` reads
    ``config.ffn_types[layer_idx]`` (alloy convention) to decide whether
    ``self.gate`` is :class:`DeepseekV4TopKRouter` (``"dsv4_moe"``) or
    :class:`DeepseekV4HashRouter` (``"dsv4_hash_moe"``). HashRouter
    requires ``input_ids``; that threading happens in alloy's decoder
    layer (phase 1E).

    The shared expert (a regular :class:`DeepseekV4MLP`) runs on every
    token regardless of routing — its output is added to the routed
    expert output. Matches DSV3-style "1 shared + K routed" pattern.

    Forward signature: ``forward(hidden_states, input_ids=None)``.
    ``input_ids`` is optional for TopK variant; required (raises if
    None) for Hash variant.
    """

    def __init__(self, config, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        # alloy ffn name; e.g. "dsv4_moe" or "dsv4_hash_moe"
        self.ffn_type = config.ffn_types[layer_idx]
        self.is_hash = self.ffn_type == "dsv4_hash_moe"
        self.gate = DeepseekV4HashRouter(config) if self.is_hash else DeepseekV4TopKRouter(config)
        self.experts = DeepseekV4Experts(config)
        # DSV4's shared expert reuses the standard MLP at the same
        # intermediate_size as the routed experts. Some DSV4 configs have
        # a separate shared_expert_intermediate_size knob; the reference
        # binds shared_experts to the same MLP class without distinction,
        # so do the same here.
        self.shared_experts = DeepseekV4MLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        batch, seq_len, hidden_dim = hidden_states.shape
        residual = hidden_states
        flat = hidden_states.view(-1, hidden_dim)
        if self.is_hash:
            if input_ids is None:
                raise ValueError(
                    "DeepseekV4SparseMoeBlock with hash routing requires input_ids; "
                    "ensure AlloyDecoderLayer threads it through (phase 1E)."
                )
            _, weights, indices = self.gate(hidden_states, input_ids)
        else:
            _, weights, indices = self.gate(hidden_states)
        routed = self.experts(flat, indices, weights).view(batch, seq_len, hidden_dim)
        return routed + self.shared_experts(residual)


__all__ = [
    "DeepseekV4MLP",
    "DeepseekV4Experts",
    "DeepseekV4TopKRouter",
    "DeepseekV4HashRouter",
    "DeepseekV4SparseMoeBlock",
]
