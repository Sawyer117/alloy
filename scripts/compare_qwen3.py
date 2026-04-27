"""Numerical equivalence test vs HF Qwen3ForCausalLM.

Strategy: build a small Qwen3 reference with random init, construct an equivalent
AlloyForCausalLM from the same config, copy state_dict (key names match 1:1),
run both on the same input, diff.

If this passes with tight tolerances in fp32, the math is equivalent and loading
real Qwen3-4B weights will behave identically (only shape/param count changes).

Usage:
    python -m alloy.scripts.compare_qwen3
    # or with real checkpoint:
    python -m alloy.scripts.compare_qwen3 --pretrained Qwen/Qwen3-4B
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from alloy import AlloyConfig, AlloyForCausalLM
from alloy.configuration_alloy import hf_layer_types_to_alloy


def _tiny_qwen3_config():
    from transformers.models.qwen3 import Qwen3Config

    return Qwen3Config(
        vocab_size=1024,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        max_position_embeddings=256,
        hidden_act="silu",
        rms_norm_eps=1e-6,
        tie_word_embeddings=False,
        attention_bias=False,
        attention_dropout=0.0,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
        use_sliding_window=False,
    )


def _alloy_from_qwen3_config(qwen3_cfg) -> AlloyConfig:
    num_layers = qwen3_cfg.num_hidden_layers
    return AlloyConfig(
        vocab_size=qwen3_cfg.vocab_size,
        hidden_size=qwen3_cfg.hidden_size,
        num_hidden_layers=num_layers,
        num_attention_heads=qwen3_cfg.num_attention_heads,
        num_key_value_heads=qwen3_cfg.num_key_value_heads,
        head_dim=qwen3_cfg.head_dim,
        intermediate_size=qwen3_cfg.intermediate_size,
        max_position_embeddings=qwen3_cfg.max_position_embeddings,
        hidden_act=qwen3_cfg.hidden_act,
        rms_norm_eps=qwen3_cfg.rms_norm_eps,
        rms_norm_unit_offset=False,
        attention_bias=qwen3_cfg.attention_bias,
        attention_dropout=qwen3_cfg.attention_dropout,
        attn_output_gate=False,
        sliding_window=getattr(qwen3_cfg, "sliding_window", None),
        layer_types=hf_layer_types_to_alloy(qwen3_cfg.layer_types),
        ffn_types=["qwen3_mlp"] * num_layers,
        rope_parameters=dict(qwen3_cfg.rope_parameters),
        tie_word_embeddings=qwen3_cfg.tie_word_embeddings,
    )


def _diff_forward(ref: torch.nn.Module, ours: torch.nn.Module, vocab_size: int) -> dict[str, float]:
    torch.manual_seed(42)
    batch, seq = 2, 32
    input_ids = torch.randint(0, vocab_size, (batch, seq))

    ref.eval()
    ours.eval()
    with torch.no_grad():
        ref_out = ref(input_ids=input_ids, use_cache=False).logits
        our_out = ours(input_ids=input_ids, use_cache=False).logits

    diff = (ref_out - our_out).abs()
    return {
        "shape_ref": tuple(ref_out.shape),
        "shape_ours": tuple(our_out.shape),
        "max_abs": diff.max().item(),
        "mean_abs": diff.mean().item(),
        "max_ref": ref_out.abs().max().item(),
        "relative_max": (diff.max() / (ref_out.abs().max() + 1e-12)).item(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained",
        default=None,
        help="Optional HF hub id or local path to load instead of random init.",
    )
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--dtype", choices=["fp32", "bf16", "fp16"], default="fp32")
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM

    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype]

    if args.pretrained:
        print(f"Loading pretrained reference: {args.pretrained}")
        ref = AutoModelForCausalLM.from_pretrained(
            args.pretrained, torch_dtype=dtype, attn_implementation="eager"
        )
        qwen3_cfg = ref.config
    else:
        print("Building tiny random-init Qwen3 reference")
        qwen3_cfg = _tiny_qwen3_config()
        qwen3_cfg._attn_implementation = "eager"
        from transformers.models.qwen3 import Qwen3ForCausalLM

        torch.manual_seed(0)
        ref = Qwen3ForCausalLM(qwen3_cfg).to(dtype)

    print(f"Ref layer_types: {qwen3_cfg.layer_types}")
    print(f"Ref dtype: {dtype}")

    alloy_cfg = _alloy_from_qwen3_config(qwen3_cfg)
    alloy_cfg._attn_implementation = "eager"
    ours = AlloyForCausalLM(alloy_cfg).to(dtype)

    # Copy state_dict. Keys must match 1:1.
    ref_state = ref.state_dict()
    our_state = ours.state_dict()
    missing_in_ours = sorted(set(ref_state) - set(our_state))
    unexpected_in_ours = sorted(set(our_state) - set(ref_state))
    if missing_in_ours:
        print(f"Ref keys NOT in ours (first 10): {missing_in_ours[:10]}")
    if unexpected_in_ours:
        print(f"Our keys NOT in ref  (first 10): {unexpected_in_ours[:10]}")

    load_res = ours.load_state_dict(ref_state, strict=False)
    print(f"load_state_dict: missing={len(load_res.missing_keys)} unexpected={len(load_res.unexpected_keys)}")

    metrics = _diff_forward(ref, ours, qwen3_cfg.vocab_size)
    print("\nForward diff:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    ok = metrics["max_abs"] <= args.atol + args.rtol * metrics["max_ref"]
    print(f"\n{'PASS' if ok else 'FAIL'} (tol: atol={args.atol}, rtol={args.rtol})")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
