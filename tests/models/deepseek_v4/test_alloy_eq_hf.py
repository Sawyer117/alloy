"""DeepSeek-V4 alloy port vs HF reference — equivalence test on random weights.

Builds the same (small) DSV4 architecture both as alloy ``AlloyForCausalLM``
(``use_mhc=True``) and HF ``DeepseekV4ForCausalLM``, copies HF's
``state_dict()`` into alloy (verifies key-by-key match, no missing /
unexpected), forwards identical input through both, and compares logits.

PASS criterion in fp32: ``max_abs_diff == 0`` (byte-exact). The whole
point of using random weights with the same init seed is that the math
is the only thing that can drift — there's no quantization noise from
training, no tokenizer / vocab mismatch, no version skew. Anything other
than 0 in fp32 is a real port bug.

In bf16 the comparison degrades to "drift consistent with bf16 noise
floor", which the script reports without enforcing.

Requires transformers >= 5.7 (the version that ships DSV4 + the
``sqrtsoftplus`` activation + the ``LAYER_TYPE_CACHE_MAPPING``
auto-registration). On older transformers the script will fail at the
HF model import or at the activation lookup — that's the expected
gating behavior.

Usage::

    pytest alloy/tests/models/deepseek_v4/test_alloy_eq_hf.py    # CI: fp32 byte-exact
    python -m alloy.tests.models.deepseek_v4.test_alloy_eq_hf    # standalone CLI
    python -m alloy.tests.models.deepseek_v4.test_alloy_eq_hf --dtype bf16
"""
from __future__ import annotations

import argparse

import torch


# --------------------------------------------------------------------------- #
# Config builders — same architecture spec for both sides.
# --------------------------------------------------------------------------- #


def _build_configs(attn_implementation: str = "eager"):
    """Return (hf_config, alloy_config) for the same small DSV4 architecture.

    Small enough for fast random-weight tests (~few seconds for forward
    on CPU); covers all 3 mixer flavors + both MoE flavors + MHC.
    """
    from transformers.models.deepseek_v4 import DeepseekV4Config

    from alloy import AlloyConfig

    common = dict(
        vocab_size=256,
        hidden_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=1,  # DSV4 is single-KV-head MQA
        head_dim=32,
        intermediate_size=128,
        max_position_embeddings=512,
        sliding_window=64,
        rms_norm_eps=1e-6,
        attention_bias=False,
        attention_dropout=0.0,
        # MHC
        hc_mult=2,
        hc_sinkhorn_iters=2,
        hc_eps=1e-6,
        # MoE — use DSV4's source-coupled name (``n_routed_experts``).
        # AlloyConfig accepts it because dsv4_moe port reads it directly;
        # DeepseekV4Config reads it as the storage field for its
        # ``num_local_experts`` attribute_map alias.
        n_routed_experts=4,
        num_experts_per_tok=2,
        scoring_func="sqrtsoftplus",
        routed_scaling_factor=1.5,
        swiglu_limit=10.0,
        mlp_bias=False,
        # DSV4 attention
        q_lora_rank=64,
        o_groups=2,
        o_lora_rank=64,
        # Lightning Indexer (CSA only)
        index_n_heads=4,
        index_head_dim=32,
        index_topk=8,
        compress_rates={
            "compressed_sparse_attention": 4,
            "heavily_compressed_attention": 8,
        },
        rope_parameters={
            "main":     {"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.25},
            "compress": {"rope_type": "default", "rope_theta": 100000.0, "partial_rotary_factor": 0.25},
        },
    )

    # 4 layers covering all 3 attention flavors + both router flavors.
    layer_types_hf    = ["heavily_compressed_attention", "heavily_compressed_attention",
                         "compressed_sparse_attention", "sliding_attention"]
    layer_types_alloy = ["dsv4_hca_attention", "dsv4_hca_attention",
                         "dsv4_csa_attention", "dsv4_sliding_attention"]
    # First N layers use hash routing per DSV4 convention.
    mlp_types_hf      = ["hash_moe", "moe", "moe", "moe"]
    ffn_types_alloy   = ["dsv4_hash_moe", "dsv4_moe", "dsv4_moe", "dsv4_moe"]

    # Force eager attention on both sides so the byte-equivalence test
    # exercises the same kernel family (alloy's _eager_attention_with_sinks
    # vs HF's eager_attention_forward, which have identical bodies).
    # Without this, ``_attn_implementation`` defaults to whatever HF
    # detects per-side at construction time — if alloy lands on sdpa and
    # HF lands on eager (or vice versa), sinks handling diverges silently
    # and the equivalence test fails by 1e-1 even with identical weights.
    hf_cfg = DeepseekV4Config(
        layer_types=layer_types_hf,
        mlp_layer_types=mlp_types_hf,
        attn_implementation=attn_implementation,
        **common,
    )
    alloy_cfg = AlloyConfig(
        layer_types=layer_types_alloy,
        ffn_types=ffn_types_alloy,
        use_mhc=True,
        attn_implementation=attn_implementation,
        **common,
    )
    # Pin alloy DSV4 attention + MHC to torch so the comparison is
    # alloy_torch == HF_torch (contract 1). Bridge import elsewhere in
    # the session sets DEFAULT_IMPL for these surfaces to whatever binder
    # DEFAULTS["auto"] says; pinning here makes the test outcome
    # independent of bridge state.
    #
    # Don't pin _experts_implementation — alloy reads it from the same
    # config dict HF reads from, so whatever HF's implicit default is,
    # alloy picks the same. Explicitly setting "eager" diverges if HF
    # internally names its default something other than "eager".
    alloy_cfg._dsv4_csa_implementation = "torch"
    alloy_cfg._dsv4_hca_implementation = "torch"
    alloy_cfg._dsv4_sliding_implementation = "torch"
    alloy_cfg._dsv4_mhc_implementation = "torch"
    return hf_cfg, alloy_cfg


# --------------------------------------------------------------------------- #
# Comparison
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dtype", default="fp32", choices=["fp32", "bf16"],
                        help="fp32 = byte-exact equivalence test; bf16 = noise floor probe.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default="cpu",
                        help="cpu / cuda / npu — random-weight test is hardware-agnostic.")
    parser.add_argument("--atol", type=float, default=0.0,
                        help="fp32 PASS tolerance. Default 0.0 = byte-exact required.")
    parser.add_argument(
        "--attn-impl", default="eager", choices=["eager", "sdpa", "flash_attention_2"],
        help="Attention backend forced on both sides. Default 'eager' is the "
             "byte-exact reference path (alloy and HF have identical eager "
             "bodies that honour DSV4's per-head sinks). 'sdpa' is the "
             "production-default backend on most envs but currently does NOT "
             "consume the s_aux=sinks kwarg, so this comparison will *fail* "
             "with sdpa even when the alloy port is algorithmically correct — "
             "exactly the production caveat to be aware of.",
    )
    args = parser.parse_args(argv)

    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16}[args.dtype]
    device = torch.device(args.device)

    hf_cfg, alloy_cfg = _build_configs(attn_implementation=args.attn_impl)

    # ----- 1. Build HF reference -----
    from transformers.models.deepseek_v4 import DeepseekV4ForCausalLM

    print("[1/4] Building HF reference DeepseekV4ForCausalLM ...", flush=True)
    torch.manual_seed(args.seed)
    hf_model = DeepseekV4ForCausalLM(hf_cfg).to(device=device, dtype=dtype).eval()

    # ----- 2. Build alloy port + load HF state_dict -----
    from alloy import AlloyForCausalLM

    print("[2/4] Building alloy AlloyForCausalLM (use_mhc=True) ...", flush=True)
    alloy_model = AlloyForCausalLM(alloy_cfg).to(device=device, dtype=dtype).eval()

    # Diagnostic: confirm both sides resolved to the same attention backend.
    # If these differ, equivalence is impossible regardless of state_dict copy.
    print(f"      HF    _attn_implementation: {hf_model.config._attn_implementation!r}")
    print(f"      alloy _attn_implementation: {alloy_model.config._attn_implementation!r}")

    print("[3/4] Loading HF state_dict into alloy (strict=True) ...", flush=True)
    sd = hf_model.state_dict()
    res = alloy_model.load_state_dict(sd, strict=False)
    print(f"      keys total      : {len(sd)}")
    print(f"      missing keys    : {len(res.missing_keys)}")
    print(f"      unexpected keys : {len(res.unexpected_keys)}")
    if res.missing_keys:
        print(f"      first 5 missing : {res.missing_keys[:5]}")
    if res.unexpected_keys:
        print(f"      first 5 unexpected: {res.unexpected_keys[:5]}")

    # ----- 3. Forward identical input through both -----
    torch.manual_seed(args.seed)
    input_ids = torch.randint(
        0, hf_cfg.vocab_size, (args.batch_size, args.seq_len), device=device
    )

    print(f"[4/4] Forward {args.batch_size}x{args.seq_len} tokens, dtype={args.dtype} ...")
    with torch.inference_mode():
        hf_out = hf_model(input_ids=input_ids).logits
        alloy_out = alloy_model(input_ids=input_ids).logits

    diff = (hf_out.float() - alloy_out.float()).abs()
    print()
    print(f"  HF logits    shape : {tuple(hf_out.shape)}")
    print(f"  alloy logits shape : {tuple(alloy_out.shape)}")
    print(f"  max_abs_diff       : {diff.max().item():.4e}")
    print(f"  mean_abs_diff      : {diff.mean().item():.4e}")
    print(f"  max_ref_abs        : {hf_out.abs().max().item():.4e}")

    hf_top1 = hf_out.argmax(-1)
    alloy_top1 = alloy_out.argmax(-1)
    token_match = torch.equal(hf_top1, alloy_top1)
    print(f"  top1-token match   : {token_match}")
    print()

    # PASS criterion: fp32 should be byte-exact (within --atol).
    if args.dtype == "fp32":
        ok = diff.max().item() <= args.atol and len(res.missing_keys) == 0 and len(res.unexpected_keys) == 0
        print(f"{'PASS' if ok else 'FAIL'} — fp32 byte-equivalence "
              f"(max_abs={diff.max().item():.4e}, atol={args.atol})")
        return 0 if ok else 1
    else:
        # bf16 — informational only; report drift, expect non-zero noise.
        print("INFO — bf16 path; fp32 is the byte-exact reference. "
              f"Observed drift: max_abs={diff.max().item():.4e}, "
              f"mean_abs={diff.mean().item():.4e}.")
        return 0


def test_dsv4_alloy_eq_hf_fp32_byte_exact() -> None:
    """Contract 1: alloy DSV4 port == HF reference, fp32 byte-exact on random weights."""
    try:
        from transformers.models import deepseek_v4  # noqa: F401
    except ImportError:
        import pytest

        pytest.skip("HF transformers lacks deepseek_v4 (needs >=5.7)")
    assert main([]) == 0


if __name__ == "__main__":
    raise SystemExit(main())
