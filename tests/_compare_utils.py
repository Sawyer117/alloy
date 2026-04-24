"""Shared helpers for large-checkpoint equivalence scripts.

The generic building blocks (``build_skeleton``, ``build_on_device``,
``load_state_dict_from_disk``, ``empty_cache``) now live in
:mod:`alloy.loading` as public API. This module re-exports them for the
comparison scripts under ``alloy.tests.gpu`` and adds test-specific
helpers (``pick_device``, config translators, diff reporters).
"""
from __future__ import annotations

import torch

from ..configuration_alloy import AlloyConfig
from ..loading import (  # noqa: F401  re-exported for back-compat
    build_on_device,
    build_skeleton,
    empty_cache,
    load_state_dict_from_disk,
)


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------


def pick_device(prefer: str | None = None) -> torch.device:
    """Return the best available accelerator (cuda, npu, cpu) or the requested one."""
    if prefer is not None:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    try:
        import torch_npu  # noqa: F401

        if torch.npu.is_available():
            return torch.device("npu")
    except (ImportError, AttributeError):
        pass
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Config conversion (HF qwen3 / qwen3.5 -> AlloyConfig)
# ---------------------------------------------------------------------------


def alloy_config_from_qwen3(qwen3_cfg) -> AlloyConfig:
    """qwen3-style checkpoint -> AlloyConfig (no output gate, ones-init RMSNorm, mlp FFN)."""
    num_layers = qwen3_cfg.num_hidden_layers
    layer_types = list(getattr(qwen3_cfg, "layer_types", None) or ["full_attention"] * num_layers)
    return AlloyConfig(
        vocab_size=qwen3_cfg.vocab_size,
        hidden_size=qwen3_cfg.hidden_size,
        num_hidden_layers=num_layers,
        num_attention_heads=qwen3_cfg.num_attention_heads,
        num_key_value_heads=qwen3_cfg.num_key_value_heads,
        head_dim=getattr(qwen3_cfg, "head_dim", None)
        or qwen3_cfg.hidden_size // qwen3_cfg.num_attention_heads,
        intermediate_size=qwen3_cfg.intermediate_size,
        max_position_embeddings=qwen3_cfg.max_position_embeddings,
        hidden_act=qwen3_cfg.hidden_act,
        rms_norm_eps=qwen3_cfg.rms_norm_eps,
        rms_norm_unit_offset=False,
        attention_bias=qwen3_cfg.attention_bias,
        attention_dropout=qwen3_cfg.attention_dropout,
        attn_output_gate=False,
        sliding_window=getattr(qwen3_cfg, "sliding_window", None),
        layer_types=layer_types,
        ffn_types=["mlp"] * num_layers,
        rope_parameters=dict(qwen3_cfg.rope_parameters),
        tie_word_embeddings=qwen3_cfg.tie_word_embeddings,
    )


def alloy_config_from_qwen3_5_text(q35_text_cfg) -> AlloyConfig:
    """qwen3.5-style text config -> AlloyConfig (gated attn, (1+w)-init RMSNorm, moe FFN)."""
    num_layers = q35_text_cfg.num_hidden_layers
    layer_types = list(q35_text_cfg.layer_types)

    rope = dict(q35_text_cfg.rope_parameters)
    rope.setdefault("partial_rotary_factor", 0.25)
    rope.setdefault("mrope_interleaved", True)
    rope.setdefault("mrope_section", getattr(q35_text_cfg, "mrope_section", [4, 4, 4]))

    return AlloyConfig(
        vocab_size=q35_text_cfg.vocab_size,
        hidden_size=q35_text_cfg.hidden_size,
        num_hidden_layers=num_layers,
        num_attention_heads=q35_text_cfg.num_attention_heads,
        num_key_value_heads=q35_text_cfg.num_key_value_heads,
        head_dim=getattr(q35_text_cfg, "head_dim", None)
        or q35_text_cfg.hidden_size // q35_text_cfg.num_attention_heads,
        intermediate_size=getattr(
            q35_text_cfg, "intermediate_size", q35_text_cfg.shared_expert_intermediate_size
        ),
        max_position_embeddings=q35_text_cfg.max_position_embeddings,
        hidden_act=q35_text_cfg.hidden_act,
        rms_norm_eps=q35_text_cfg.rms_norm_eps,
        rms_norm_unit_offset=True,
        attention_bias=q35_text_cfg.attention_bias,
        attention_dropout=q35_text_cfg.attention_dropout,
        attn_output_gate=True,
        sliding_window=None,
        layer_types=layer_types,
        ffn_types=["moe"] * num_layers,
        linear_num_key_heads=q35_text_cfg.linear_num_key_heads,
        linear_num_value_heads=q35_text_cfg.linear_num_value_heads,
        linear_key_head_dim=q35_text_cfg.linear_key_head_dim,
        linear_value_head_dim=q35_text_cfg.linear_value_head_dim,
        linear_conv_kernel_dim=q35_text_cfg.linear_conv_kernel_dim,
        num_experts=q35_text_cfg.num_experts,
        num_experts_per_tok=q35_text_cfg.num_experts_per_tok,
        moe_intermediate_size=q35_text_cfg.moe_intermediate_size,
        shared_expert_intermediate_size=q35_text_cfg.shared_expert_intermediate_size,
        rope_parameters=rope,
        tie_word_embeddings=q35_text_cfg.tie_word_embeddings,
    )


# ---------------------------------------------------------------------------
# Output comparison
# ---------------------------------------------------------------------------


def diff_logits(ref: torch.Tensor, ours: torch.Tensor) -> dict[str, float]:
    """Pointwise absolute / relative diff stats between two logits tensors."""
    ref = ref.to(torch.float32)
    ours = ours.to(torch.float32)
    diff = (ref - ours).abs()
    ref_abs_max = ref.abs().max().clamp_min(1e-12)
    return {
        "max_abs": diff.max().item(),
        "mean_abs": diff.mean().item(),
        "max_ref_abs": ref.abs().max().item(),
        "relative_max": (diff.max() / ref_abs_max).item(),
    }


def compare_tokens(ref_tokens: torch.Tensor, our_tokens: torch.Tensor) -> dict[str, object]:
    """Check greedy-generated token IDs are identical (the strongest behavioral check)."""
    match = torch.equal(ref_tokens, our_tokens)
    first_divergence = None
    if not match:
        eq = (ref_tokens == our_tokens)
        for i in range(eq.shape[0]):
            row = eq[i]
            if not row.all():
                first_divergence = (i, int((~row).nonzero()[0].item()))
                break
    return {
        "match": match,
        "ref_tokens": ref_tokens.tolist(),
        "our_tokens": our_tokens.tolist(),
        "first_divergence_row_col": first_divergence,
    }
