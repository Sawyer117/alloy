"""Qwen3.5-MoE alloy port vs HF reference — equivalence test on random weights.

Same protocol as the DSV4 / qwen3 contract-1 tests: build a small qwen3.5
architecture (gated delta net + MoE FFN + gated attn) both as alloy
``AlloyForCausalLM`` and HF ``Qwen3_5MoeForCausalLM``, copy state_dict,
forward identical input, compare logits.

PASS criterion in fp32: ``max_abs_diff <= atol`` (default ``atol=0`` byte-exact).
Note: stacked GatedDeltaNet + MoE with random weights is a stricter equivalence
target than dense qwen3 — any off-by-one in the gated update or routing math
shows up here as a non-zero fp32 diff.

Usage::

    pytest alloy/tests/models/qwen3_5_moe/test_alloy_eq_hf.py     # CI: fp32 byte-exact
    python -m alloy.tests.models.qwen3_5_moe.test_alloy_eq_hf     # standalone CLI
"""
from __future__ import annotations

import argparse

import torch


def _build_configs(attn_implementation: str = "eager"):
    """Return (hf_text_config, alloy_config) for a small qwen3.5 architecture.

    Mix of ``linear_attention`` (gated delta net) layers + one
    ``full_attention`` so both code paths are exercised on every forward.
    """
    from transformers.models.qwen3_5_moe import Qwen3_5MoeTextConfig
    from alloy.tests._compare_utils import alloy_config_from_qwen3_5_text

    hf_cfg = Qwen3_5MoeTextConfig(
        vocab_size=1024,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        max_position_embeddings=256,
        layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        linear_conv_kernel_dim=4,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=128,
        shared_expert_intermediate_size=128,
        pad_token_id=0,
        attn_implementation=attn_implementation,
    )
    alloy_cfg = alloy_config_from_qwen3_5_text(hf_cfg)
    alloy_cfg._attn_implementation = attn_implementation
    # Pin alloy GDN to torch so the comparison is alloy_torch == HF_torch
    # (contract 1). Without this, a bridge import elsewhere in the session
    # flips DEFAULT_IMPL["qwen3_5_gdn"] to "triton" and the binder triton
    # kernel (which rejects fp32) hijacks the forward.
    #
    # Do NOT pin _experts_implementation here — HF and alloy both read the
    # same field from the same config dict at experts forward time, so
    # whatever HF's eager-or-default impl is, alloy picks the same.
    # Explicitly setting "eager" on alloy_cfg only would diverge if HF's
    # implicit default isn't named "eager".
    alloy_cfg._qwen3_5_gdn_implementation = "torch"
    return hf_cfg, alloy_cfg


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dtype", default="fp32", choices=["fp32", "bf16"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--atol", type=float, default=0.0)
    parser.add_argument("--attn-impl", default="eager", choices=["eager", "sdpa"])
    args = parser.parse_args(argv)

    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16}[args.dtype]
    device = torch.device(args.device)

    hf_cfg, alloy_cfg = _build_configs(attn_implementation=args.attn_impl)

    from transformers.models.qwen3_5_moe import Qwen3_5MoeForCausalLM
    from alloy import AlloyForCausalLM

    print("[1/3] Building HF Qwen3_5MoeForCausalLM ...", flush=True)
    torch.manual_seed(args.seed)
    hf_model = Qwen3_5MoeForCausalLM(hf_cfg).to(device=device, dtype=dtype).eval()

    print("[2/3] Building alloy AlloyForCausalLM ...", flush=True)
    alloy_model = AlloyForCausalLM(alloy_cfg).to(device=device, dtype=dtype).eval()

    sd = hf_model.state_dict()
    res = alloy_model.load_state_dict(sd, strict=False)
    print(f"      keys total      : {len(sd)}")
    print(f"      missing keys    : {len(res.missing_keys)}")
    print(f"      unexpected keys : {len(res.unexpected_keys)}")
    if res.missing_keys:
        print(f"      first 5 missing : {res.missing_keys[:5]}")
    if res.unexpected_keys:
        print(f"      first 5 unexpected: {res.unexpected_keys[:5]}")

    torch.manual_seed(args.seed)
    input_ids = torch.randint(
        0, hf_cfg.vocab_size, (args.batch_size, args.seq_len), device=device
    )

    print(f"[3/3] Forward {args.batch_size}x{args.seq_len}, dtype={args.dtype} ...")
    with torch.inference_mode():
        hf_out = hf_model(input_ids=input_ids).logits
        alloy_out = alloy_model(input_ids=input_ids).logits

    diff = (hf_out.float() - alloy_out.float()).abs()
    print(f"  max_abs_diff  : {diff.max().item():.4e}")
    print(f"  mean_abs_diff : {diff.mean().item():.4e}")

    if args.dtype == "fp32":
        ok = (diff.max().item() <= args.atol
              and len(res.missing_keys) == 0
              and len(res.unexpected_keys) == 0)
        print(f"{'PASS' if ok else 'FAIL'} — fp32 (max_abs={diff.max().item():.4e}, atol={args.atol})")
        return 0 if ok else 1
    else:
        print(f"INFO — bf16 drift: max_abs={diff.max().item():.4e}")
        return 0


def test_qwen3_5_moe_alloy_eq_hf_fp32_byte_exact() -> None:
    """Contract 1: alloy qwen3.5 port == HF reference, fp32 byte-exact on random weights."""
    assert main([]) == 0


if __name__ == "__main__":
    raise SystemExit(main())
