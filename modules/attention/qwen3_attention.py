from __future__ import annotations

from typing import Callable

import torch
from torch import nn

from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from ..registry import get_mixer, register_mixer
from ..shared.attention_kernels import eager_attention_forward
from ..shared.norm import RMSNorm
from ..shared.rotary import apply_rotary_pos_emb


class Qwen3Attention(nn.Module):
    """Grouped-Query Attention ported from qwen3 / qwen3.5.

    - ``config.attn_output_gate=False``  => qwen3 style (no output gate, q_proj out = H*D)
    - ``config.attn_output_gate=True``   => qwen3.5 style (output gate, q_proj out = H*D*2)

    Parameter names (q_proj/k_proj/v_proj/o_proj/q_norm/k_norm) and their shapes
    match the corresponding HF checkpoints exactly, so HF state_dict loads cleanly.
    """

    def __init__(self, config, layer_idx: int) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        # Look up our own registry entry to find out whether this layer is a
        # sliding-window variant — without comparing string literals.
        layer_name = config.layer_types[layer_idx] if config.layer_types else None
        self.mask_kind = get_mixer(layer_name).mask_kind if layer_name else "causal"

        self.head_dim = getattr(config, "head_dim", None) or (config.hidden_size // config.num_attention_heads)
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_kv_heads
        self.scaling = self.head_dim ** -0.5
        self.attention_dropout = getattr(config, "attention_dropout", 0.0)
        self.is_causal = True
        self.attn_output_gate = bool(getattr(config, "attn_output_gate", False))

        q_out_mul = 2 if self.attn_output_gate else 1
        bias = bool(getattr(config, "attention_bias", False))

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim * q_out_mul, bias=bias)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=bias)

        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps, unit_offset=config.rms_norm_unit_offset)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps, unit_offset=config.rms_norm_unit_offset)

        self.sliding_window = config.sliding_window if self.mask_kind == "sliding" else None

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        past_key_values=None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        if self.attn_output_gate:
            # q_proj -> [*, H*D*2] -> chunk into (q, gate) each [*, H*D]
            q_and_gate = self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2)
            query_states, gate = torch.chunk(q_and_gate, 2, dim=-1)
            gate = gate.reshape(*input_shape, -1)
            query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
        else:
            gate = None
            query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)

        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            getattr(self.config, "_attn_implementation", "eager"), eager_attention_forward
        )
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        if gate is not None:
            attn_output = attn_output * torch.sigmoid(gate)
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


# Register the same class under two alloy keys with different mask_kinds.
# Sliding-window semantics are encoded by the mask the model precompute hands in
# plus the `sliding_window` attribute consumed by attention kernels.
register_mixer("qwen3_attention", attr_name="self_attn", mask_kind="causal")(Qwen3Attention)
register_mixer("qwen3_attention_sliding", attr_name="self_attn", mask_kind="sliding")(Qwen3Attention)
