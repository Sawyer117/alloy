"""DFlash draft (canonical Qwen3-dense form) for DeepSeek-V4-Flash.

Follows the same z-lab DFlash convention as ``dsv4_pro_dflash.py`` but
sized to the smaller V4-Flash main (43 layers, hidden=4096, ~120-180B
total):

  * model_type=qwen3 dense
  * hidden_size = 4096 (= main)
  * num_hidden_layers = 6 (~14% of main's 43)
  * intermediate_size = 12288 (3x hidden)
  * head_dim = 128, num_attention_heads = 32, GQA 1:8
  * vocab_size = 129280 (= main)

Per-forward param count ~1.8B, ~1.0-1.5% of main. Smaller absolute
size than the Pro draft but proportionally larger w.r.t. main — typical
of the trend "smaller main, relatively larger draft".

Run::

    python -m alloy.examples.roofline.dflash.dsv4_flash_dflash             # --level 1
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


def build_dsv4_flash_dflash_config():
    num_hidden_layers = 6
    return SimpleNamespace(
        hidden_size=4096,
        vocab_size=129280,
        num_hidden_layers=num_hidden_layers,
        tie_word_embeddings=False,

        num_attention_heads=32,        # = hidden / head_dim
        num_key_value_heads=4,         # GQA 1:8
        head_dim=128,
        sliding_window=None,

        intermediate_size=12288,       # 3x hidden
        mlp_bias=False,

        layer_types=["qwen3_attention"] * num_hidden_layers,
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

    config = build_dsv4_flash_dflash_config()
    from collections import Counter
    print(f"DSV4-Flash-DFlash (Qwen3-dense draft, {config.num_hidden_layers} layers, hidden={config.hidden_size})")
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
