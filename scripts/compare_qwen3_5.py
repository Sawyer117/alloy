"""Numerical equivalence test vs HF Qwen3_5MoeForCausalLM (text-only).

Builds a small random-init Qwen3_5Moe reference with a [linear_attention × 3,
full_attention] layer pattern and MoE FFN, mirrors the config into AlloyConfig
(rms_norm_unit_offset=True, attn_output_gate=True, partial_rotary_factor=0.25,
mrope_interleaved=True), copies state_dict, and diffs the forward outputs.

Random init + same seeds produces identical weights; state_dict copy across
matching key names makes our model byte-equivalent to the HF reference if the
module math is right.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from alloy import AlloyConfig, AlloyForCausalLM


def _tiny_qwen3_5_moe_text_config():
    from transformers.models.qwen3_5_moe import Qwen3_5MoeTextConfig

    return Qwen3_5MoeTextConfig(
        vocab_size=512,
        hidden_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        hidden_act="silu",
        max_position_embeddings=256,
        rms_norm_eps=1e-6,
        tie_word_embeddings=False,
        attention_bias=False,
        attention_dropout=0.0,
        # GDN
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        linear_conv_kernel_dim=4,
        # MoE
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=128,
        shared_expert_intermediate_size=128,
        output_router_logits=False,
        layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10000.0,
            "partial_rotary_factor": 0.25,
            "mrope_interleaved": True,
            "mrope_section": [4, 4, 4],
        },
    )


def _alloy_from_qwen3_5_config(q35_cfg) -> AlloyConfig:
    num_layers = q35_cfg.num_hidden_layers
    return AlloyConfig(
        vocab_size=q35_cfg.vocab_size,
        hidden_size=q35_cfg.hidden_size,
        num_hidden_layers=num_layers,
        num_attention_heads=q35_cfg.num_attention_heads,
        num_key_value_heads=q35_cfg.num_key_value_heads,
        head_dim=q35_cfg.head_dim,
        intermediate_size=getattr(q35_cfg, "intermediate_size", q35_cfg.shared_expert_intermediate_size),
        max_position_embeddings=q35_cfg.max_position_embeddings,
        hidden_act=q35_cfg.hidden_act,
        rms_norm_eps=q35_cfg.rms_norm_eps,
        rms_norm_unit_offset=True,           # (1 + w) * x
        attention_bias=q35_cfg.attention_bias,
        attention_dropout=q35_cfg.attention_dropout,
        attn_output_gate=True,               # q_proj = H*D*2 with sigmoid gate
        sliding_window=None,
        layer_types=list(q35_cfg.layer_types),
        ffn_types=["moe"] * num_layers,
        # GDN
        linear_num_key_heads=q35_cfg.linear_num_key_heads,
        linear_num_value_heads=q35_cfg.linear_num_value_heads,
        linear_key_head_dim=q35_cfg.linear_key_head_dim,
        linear_value_head_dim=q35_cfg.linear_value_head_dim,
        linear_conv_kernel_dim=q35_cfg.linear_conv_kernel_dim,
        # MoE
        num_experts=q35_cfg.num_experts,
        num_experts_per_tok=q35_cfg.num_experts_per_tok,
        moe_intermediate_size=q35_cfg.moe_intermediate_size,
        shared_expert_intermediate_size=q35_cfg.shared_expert_intermediate_size,
        rope_parameters=dict(q35_cfg.rope_parameters),
        tie_word_embeddings=q35_cfg.tie_word_embeddings,
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
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--dtype", choices=["fp32", "bf16", "fp16"], default="fp32")
    args = parser.parse_args()

    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype]

    print("Building tiny random-init Qwen3.5-MoE text reference")
    q35_cfg = _tiny_qwen3_5_moe_text_config()
    q35_cfg._attn_implementation = "eager"

    from transformers.models.qwen3_5_moe import Qwen3_5MoeForCausalLM

    torch.manual_seed(0)
    ref = Qwen3_5MoeForCausalLM(q35_cfg).to(dtype)

    print(f"Ref layer_types: {q35_cfg.layer_types}")
    print(f"Ref dtype: {dtype}")

    alloy_cfg = _alloy_from_qwen3_5_config(q35_cfg)
    alloy_cfg._attn_implementation = "eager"
    ours = AlloyForCausalLM(alloy_cfg).to(dtype)

    ref_state = ref.state_dict()
    our_state = ours.state_dict()
    missing_in_ours = sorted(set(ref_state) - set(our_state))
    unexpected_in_ours = sorted(set(our_state) - set(ref_state))
    if missing_in_ours:
        print(f"\nRef keys NOT in ours ({len(missing_in_ours)}):")
        for k in missing_in_ours[:20]:
            print(f"  + {k}  shape={tuple(ref_state[k].shape)}")
    if unexpected_in_ours:
        print(f"\nOur keys NOT in ref ({len(unexpected_in_ours)}):")
        for k in unexpected_in_ours[:20]:
            print(f"  - {k}  shape={tuple(our_state[k].shape)}")

    load_res = ours.load_state_dict(ref_state, strict=False)
    print(f"\nload_state_dict: missing={len(load_res.missing_keys)} unexpected={len(load_res.unexpected_keys)}")

    metrics = _diff_forward(ref, ours, q35_cfg.vocab_size)
    print("\nForward diff:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    ok = metrics["max_abs"] <= args.atol + args.rtol * metrics["max_ref"]
    print(f"\n{'PASS' if ok else 'FAIL'} (tol: atol={args.atol}, rtol={args.rtol})")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
