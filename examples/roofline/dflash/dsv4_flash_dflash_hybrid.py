"""DFlash draft (DSV4-style hybrid) for DeepSeek-V4-Flash.

DSV4 architecture mini, parallel to ``dsv4_pro_dflash_hybrid.py`` but
sized to V4-Flash main (43 layers, hidden=4096):

  * MLA attention (head_dim=512, MQA, q_lora_rank=1024, o_groups=8) — same
    shapes as main
  * Alternating CSA / HCA layers (sparse + sliding) — 6 layers total
  * Dense ``qwen3_mlp`` FFN (no MoE)
  * Index_topk=128 (down from main's 512)

Per-forward param count ~1.5B, ~0.8-1.2% of main.

Run::

    python -m alloy.examples.roofline.dflash.dsv4_flash_dflash_hybrid             # --level 1
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


def build_dsv4_flash_dflash_hybrid_config():
    num_hidden_layers = 6
    layer_types = ["dsv4_csa_attention", "dsv4_hca_attention"] * (num_hidden_layers // 2)
    return SimpleNamespace(
        hidden_size=4096,
        vocab_size=129280,
        num_hidden_layers=num_hidden_layers,
        tie_word_embeddings=False,

        # DSV4 MLA shapes (= main)
        num_attention_heads=64,
        num_key_value_heads=1,
        head_dim=512,
        q_lora_rank=1024,
        o_groups=8,
        o_lora_rank=1024,
        sliding_window=128,
        compress_rates={
            "heavily_compressed_attention": 128,
            "compressed_sparse_attention":   4,
        },
        index_n_heads=64,
        index_head_dim=128,
        index_topk=128,

        # Dense FFN
        intermediate_size=8192,        # 2x hidden
        mlp_bias=False,

        layer_types=layer_types,
        ffn_types=["qwen3_mlp"] * num_hidden_layers,

        num_target_layers=43,
        block_size=16,
        dflash_config={
            "target_layer_ids": [1, 9, 17, 25, 33, 41],
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--level", type=int, choices=[1, 2, 3], default=1)
    args = parser.parse_args()

    config = build_dsv4_flash_dflash_hybrid_config()
    from collections import Counter
    print(f"DSV4-Flash-DFlash-hybrid (DSV4-style draft, {config.num_hidden_layers} layers, hidden={config.hidden_size})")
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
