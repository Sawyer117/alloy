"""DFlash draft (DSV4-style hybrid) for DeepSeek-V4-Pro.

Deliberately violates z-lab's "draft = Qwen3 dense" convention: keeps
main's MLA attention machinery (q_lora_rank / o_lora_rank / o_groups +
head_dim=512 + per-head sinks via the attention spec) plus alternating
HCA/CSA sparse attention. Drops MoE — FFN is dense ``qwen3_mlp`` to
avoid the routing overhead that would dominate draft step latency.

Hypothesis being tested: at ultra-long context (>=128K) MLA's tiny KV
cache and CSA's sparse attention save enough bandwidth in the draft
to offset the implementation overhead. The Qwen3-dense canonical
sibling (``dsv4_pro_dflash.py``) is the comparison baseline.

Per-forward param count ~4.1B, ~0.6% of main's ~671B.

Run::

    python -m alloy.examples.roofline.dflash.dsv4_pro_dflash_hybrid              # --level 1
    python -m alloy.examples.roofline.dflash.dsv4_pro_dflash_hybrid --level 3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

import torch

from alloy.roofline import (
    CustomHardware,
    format_comparison,
    roofline_decode,
    roofline_mini_prefill,
    roofline_prefill,
)


def build_dsv4_pro_dflash_hybrid_config():
    num_hidden_layers = 8
    # Alternating HCA/CSA matches main's pattern but compressed to 8 layers.
    layer_types = ["dsv4_hca_attention", "dsv4_csa_attention"] * (num_hidden_layers // 2)
    return SimpleNamespace(
        # Top-level shape (same hidden + vocab as main)
        hidden_size=7168,
        vocab_size=129280,
        num_hidden_layers=num_hidden_layers,
        tie_word_embeddings=False,

        # Attention (DSV4 MLA, same per-head shapes as main)
        num_attention_heads=128,       # = main
        num_key_value_heads=1,         # MQA
        head_dim=512,                  # = main MLA head_dim
        q_lora_rank=1536,              # = main
        o_groups=16,                   # = main
        o_lora_rank=1024,              # = main
        sliding_window=128,
        # HCA at 128:1, CSA at 4:1 — same compress rates as main
        compress_rates={
            "heavily_compressed_attention": 128,
            "compressed_sparse_attention":   4,
        },
        # Lightning Indexer for CSA — half of main's index_topk so draft
        # indexer compute scales down. Hyperparameter to tune in practice.
        index_n_heads=64,
        index_head_dim=128,
        index_topk=256,

        # FFN (DENSE — no MoE). qwen3_mlp reads ``intermediate_size`` directly.
        intermediate_size=14336,       # 2x hidden (smaller than canonical to keep params modest)
        mlp_bias=False,

        # Layer composition
        layer_types=layer_types,
        ffn_types=["qwen3_mlp"] * num_hidden_layers,

        # DFlash-runtime metadata
        num_target_layers=61,
        block_size=16,
        dflash_config={
            "target_layer_ids": [1, 9, 17, 25, 33, 41, 49, 58],
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--level", type=int, choices=[1, 2, 3], default=1)
    args = parser.parse_args()

    config = build_dsv4_pro_dflash_hybrid_config()
    from collections import Counter
    print(f"DSV4-Pro-DFlash-hybrid (DSV4-style draft, {config.num_hidden_layers} layers, hidden={config.hidden_size})")
    print(f"  layer_types: {dict(Counter(config.layer_types))}")
    print(f"  ffn_types  : {dict(Counter(config.ffn_types))}")
    print()

    my_device = CustomHardware(
        name="my-device", hbm_bandwidth=8e12,
        fp16=1000e12, bf16=1000e12, fp8=2000e12,
        # fp4 ~ 4 PFLOPS placeholder (not stored).
    )

    cases = [
        ("prefill (seq=8192)",    lambda hw: roofline_prefill(config, batch=1, seq_len=8192,       hardware=hw)),
        ("decode (cache=8192)",   lambda hw: roofline_decode (config, batch=1, kv_cache_len=8192,  hardware=hw)),
        ("decode (cache=131072)", lambda hw: roofline_decode (config, batch=1, kv_cache_len=131072, hardware=hw)),
    ]
    hardware_options = ["H100", "Ascend950PR", my_device]

    if args.level == 1:
        for case_name, runner in cases:
            print(format_comparison([runner(hw) for hw in hardware_options], title=case_name))
            print()
    else:
        for case_name, runner in cases:
            for hw in hardware_options:
                print(runner(hw).format(level=args.level))
                print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
