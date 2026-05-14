"""Qwen3 (dense) alloy port vs HF reference — equivalence test on random weights.

Mirror of the DSV4 contract-1 test: build the same small qwen3 architecture
both as alloy ``AlloyForCausalLM`` and HF ``Qwen3ForCausalLM``, copy HF's
``state_dict`` into alloy, forward identical input, compare logits.

PASS criterion in fp32: ``max_abs_diff <= atol`` (default ``atol=0`` byte-exact).
Random init + same seed + same config means math is the only thing that can
drift; non-zero in fp32 is a real port bug.

Usage::

    pytest alloy/tests/models/qwen3/test_alloy_eq_hf.py         # CI: fp32 byte-exact
    python -m alloy.tests.models.qwen3.test_alloy_eq_hf         # standalone CLI
    python -m alloy.tests.models.qwen3.test_alloy_eq_hf --dtype bf16
"""
from __future__ import annotations

import argparse

import torch


def _build_configs(attn_implementation: str = "eager"):
    """Return (hf_qwen3_config, alloy_config) for the same small qwen3 dense
    architecture. Small enough that a CPU forward runs in seconds."""
    from transformers.models.qwen3 import Qwen3Config
    from alloy.tests._compare_utils import alloy_config_from_qwen3

    hf_cfg = Qwen3Config(
        vocab_size=1024,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        max_position_embeddings=256,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        attn_implementation=attn_implementation,
    )
    alloy_cfg = alloy_config_from_qwen3(hf_cfg)
    alloy_cfg._attn_implementation = attn_implementation
    return hf_cfg, alloy_cfg


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dtype", default="fp32", choices=["fp32", "bf16"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--atol", type=float, default=0.0,
                        help="fp32 PASS tolerance. Default 0.0 = byte-exact required.")
    parser.add_argument("--attn-impl", default="eager", choices=["eager", "sdpa"])
    args = parser.parse_args(argv)

    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16}[args.dtype]
    device = torch.device(args.device)

    hf_cfg, alloy_cfg = _build_configs(attn_implementation=args.attn_impl)

    from transformers.models.qwen3 import Qwen3ForCausalLM
    from alloy import AlloyForCausalLM

    print("[1/3] Building HF Qwen3ForCausalLM ...", flush=True)
    torch.manual_seed(args.seed)
    hf_model = Qwen3ForCausalLM(hf_cfg).to(device=device, dtype=dtype).eval()

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


def test_qwen3_alloy_eq_hf_fp32_byte_exact() -> None:
    """Contract 1: alloy qwen3 port == HF reference, fp32 byte-exact on random weights."""
    assert main([]) == 0


if __name__ == "__main__":
    raise SystemExit(main())
