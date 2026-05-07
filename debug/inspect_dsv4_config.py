"""Diagnostic: inspect DSV4Config attribute_map + actual field where 'num_experts' lands.

Goal: confirm the size-mismatch root cause for compare_dsv4_random.py is
that DSV4Config and AlloyConfig use different field names for the same
quantity (alloy: ``num_experts``; DSV4: ``n_routed_experts`` / aliased
``num_local_experts``). After this prints, we'll know exactly which
fields need to be passed under which names in ``_build_configs()``.

Delete this file (and the ``debug/`` directory) once the test is fixed.
"""
from __future__ import annotations

import sys


def main() -> int:
    try:
        from transformers.models.deepseek_v4 import DeepseekV4Config
    except ImportError as e:
        print(f"FAIL — DSV4 not in installed transformers: {e}")
        print(
            "Install transformers main / 5.8.0.dev0:\n"
            "  pip install -e /path/to/transformers\n"
            "or:\n"
            "  pip install git+https://github.com/huggingface/transformers.git@main"
        )
        return 1

    print("=" * 60)
    print("DSV4Config defaults (no kwargs passed)")
    print("=" * 60)
    cfg = DeepseekV4Config()
    fields = [
        # Expert / MoE related — most likely culprits
        "num_experts",
        "num_local_experts",
        "n_routed_experts",
        "num_experts_per_tok",
        "n_shared_experts",
        # FFN size
        "intermediate_size",
        "moe_intermediate_size",
        # MHC
        "hc_mult",
        "hc_sinkhorn_iters",
        "hc_eps",
        # Attention
        "q_lora_rank",
        "o_groups",
        "o_lora_rank",
        # Indexer
        "index_n_heads",
        "index_head_dim",
        "index_topk",
        # Compress
        "compress_rates",
        # Hash routing
        "default_num_hash_layers",
        # Activation
        "scoring_func",
        "swiglu_limit",
        "routed_scaling_factor",
        "mlp_bias",
        # Misc
        "vocab_size",
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "sliding_window",
    ]
    for name in fields:
        val = getattr(cfg, name, "<MISSING>")
        # Truncate dict / list values for readability
        sval = str(val)
        if len(sval) > 80:
            sval = sval[:77] + "..."
        print(f"  {name:32s} = {sval}")

    print()
    print("=" * 60)
    print("attribute_map (HF aliases — left side reads right side)")
    print("=" * 60)
    if hasattr(cfg, "attribute_map") and cfg.attribute_map:
        for k, v in cfg.attribute_map.items():
            print(f"  {k:32s} -> {v}")
    else:
        print("  (no attribute_map / empty)")

    print()
    print("=" * 60)
    print("Probe: pass num_experts=4 and see where it actually lands")
    print("=" * 60)
    cfg2 = DeepseekV4Config(num_experts=4)
    probes = ["num_experts", "num_local_experts", "n_routed_experts"]
    for name in probes:
        val = getattr(cfg2, name, "<MISSING>")
        print(f"  cfg2.{name:24s} = {val}")
    # The real test: which field does the modeling read at expert
    # construction? Per references/dsv4/modeling_deepseek_v4.py:931
    # self.num_experts = config.num_local_experts.
    # If cfg2.num_local_experts != 4, num_experts kwarg was silently ignored.
    if getattr(cfg2, "num_local_experts", None) != 4:
        print(
            f"\n  Confirmed: passing num_experts=4 does NOT set num_local_experts "
            f"(it stays at default {getattr(cfg2, 'num_local_experts', '?')}). "
            f"DSV4Config's modeling reads num_local_experts. Fix: pass "
            f"n_routed_experts=4 (the actual storage field, attribute_map'd to "
            f"num_local_experts)."
        )
    else:
        print(
            f"\n  Surprise: cfg2.num_local_experts == 4. "
            f"Then root cause is elsewhere — keep digging."
        )

    print()
    print("=" * 60)
    print("Same probe for related fields (verify which kwarg names work)")
    print("=" * 60)
    test_kwargs = {
        "num_experts": 4,
        "num_local_experts": 4,
        "n_routed_experts": 4,
        "num_experts_per_tok": 2,
        "intermediate_size": 128,
        "moe_intermediate_size": 128,
    }
    for kw, val in test_kwargs.items():
        try:
            c = DeepseekV4Config(**{kw: val})
        except Exception as e:
            print(f"  pass {kw}={val}: ERROR {type(e).__name__}: {e}")
            continue
        # Check the modeling-readable canonical fields
        nle = getattr(c, "num_local_experts", "?")
        nrx = getattr(c, "n_routed_experts", "?")
        ne = getattr(c, "num_experts", "?")
        ist = getattr(c, "intermediate_size", "?")
        mist = getattr(c, "moe_intermediate_size", "?")
        print(
            f"  pass {kw:24s}={val:4d}  -> "
            f"num_local_experts={nle}  n_routed_experts={nrx}  num_experts={ne}  "
            f"intermediate_size={ist}  moe_intermediate_size={mist}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
