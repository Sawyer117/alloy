"""Smoke-test that AlloyForCausalLM constructs and forwards for both
qwen3-style and qwen3.5-MoE-style configurations without crashing.

This is NOT a numerical equivalence test — it only checks:
  - the model builds with the given layer_types / ffn_types
  - forward produces logits of the expected shape
  - no registry keys are missing

The numerical equivalence vs upstream HF qwen3 / qwen3.5 is meant to be
verified separately by loading checkpoint weights into both models and
diffing their outputs.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

# Make `alloy` importable when run as a script from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from alloy import AlloyConfig, AlloyForCausalLM


def _qwen3_like_config() -> AlloyConfig:
    """All layers = full_attention + mlp, no output gate, ones-init RMSNorm."""
    num_layers = 2
    return AlloyConfig(
        vocab_size=1024,
        hidden_size=128,
        num_hidden_layers=num_layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        intermediate_size=256,
        max_position_embeddings=256,
        hidden_act="silu",
        rms_norm_eps=1e-6,
        rms_norm_unit_offset=False,
        attn_output_gate=False,
        layer_types=["full_attention"] * num_layers,
        ffn_types=["mlp"] * num_layers,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
    )


def _qwen3_5_moe_like_config() -> AlloyConfig:
    """4 layers: [linear, linear, linear, full] × moe FFN, gated attn, (1+w)-init RMSNorm."""
    num_layers = 4
    return AlloyConfig(
        vocab_size=1024,
        hidden_size=128,
        num_hidden_layers=num_layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        intermediate_size=256,
        max_position_embeddings=256,
        hidden_act="silu",
        rms_norm_eps=1e-6,
        rms_norm_unit_offset=True,
        attn_output_gate=True,
        layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        ffn_types=["moe"] * num_layers,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        linear_conv_kernel_dim=4,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=128,
        shared_expert_intermediate_size=128,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10000.0,
            "partial_rotary_factor": 0.25,
        },
    )


def _run_forward(config: AlloyConfig, tag: str) -> None:
    torch.manual_seed(0)
    model = AlloyForCausalLM(config).eval()
    model.to(dtype=torch.float32)

    batch, seq = 2, 16
    input_ids = torch.randint(0, config.vocab_size, (batch, seq))
    with torch.no_grad():
        out = model(input_ids=input_ids, use_cache=False)

    expected = (batch, seq, config.vocab_size)
    assert out.logits.shape == expected, f"[{tag}] logits shape {tuple(out.logits.shape)} != {expected}"
    assert torch.isfinite(out.logits).all(), f"[{tag}] logits contain non-finite values"
    num_params = sum(p.numel() for p in model.parameters())
    print(f"[{tag}] OK — logits {tuple(out.logits.shape)}, {num_params / 1e6:.2f}M params")


def _run_json_roundtrip(config: AlloyConfig, tag: str) -> None:
    """Save -> parse -> reconstruct; assert round-trip fidelity and verify
    that the serialized JSON actually contains section markers.
    """
    import json as _json

    json_str = config.to_json_string()
    raw = _json.loads(json_str)

    markers = [k for k in raw if k.startswith("_section_")]
    assert markers, f"[{tag}] expected _section_* marker keys in JSON, got none"

    # Every key should now fit one of: known config field, section marker, or
    # inherited PretrainedConfig field. Reconstruct and verify core fields.
    rebuilt = AlloyConfig(**raw)
    for field in ("vocab_size", "hidden_size", "num_hidden_layers",
                  "layer_types", "ffn_types", "rms_norm_unit_offset",
                  "attn_output_gate", "num_experts"):
        assert getattr(rebuilt, field) == getattr(config, field), (
            f"[{tag}] roundtrip mismatch on {field}: "
            f"{getattr(rebuilt, field)!r} != {getattr(config, field)!r}"
        )
    # Re-emit and confirm no section markers leaked onto the instance
    for attr in vars(rebuilt):
        assert not attr.startswith("_section_"), (
            f"[{tag}] section marker {attr!r} leaked onto config instance"
        )
    print(f"[{tag}] JSON roundtrip OK ({len(markers)} section markers)")


def main() -> int:
    print("Constructing qwen3-like alloy model...")
    q3_cfg = _qwen3_like_config()
    _run_forward(q3_cfg, "qwen3-like")
    _run_json_roundtrip(q3_cfg, "qwen3-like")

    print("Constructing qwen3.5-MoE-like alloy model...")
    q35_cfg = _qwen3_5_moe_like_config()
    _run_forward(q35_cfg, "qwen3.5-MoE-like")
    _run_json_roundtrip(q35_cfg, "qwen3.5-MoE-like")

    print("\nAll smoke tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
