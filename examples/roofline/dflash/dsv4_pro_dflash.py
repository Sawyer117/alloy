"""DFlash draft (canonical Qwen3-dense form) for DeepSeek-V4-Pro.

Modeled after z-lab's observed DFlash convention:
  * model_type=qwen3 dense (no MoE / MLA / CSA — main's complexity is
    deliberately NOT inherited)
  * hidden_size = main's hidden_size (7168)
  * num_hidden_layers = 8 (small, ~13% of main's 61)
  * intermediate_size = 3x hidden = 21504
  * head_dim = 128, num_attention_heads = 64 (hidden/128), GQA 1:8
  * vocab_size = main's vocab_size = 129280 (shared tokenizer)

Per-forward param count ~5.8 B, ~0.9% of main's ~671 B. Sized to keep
draft forward cheap relative to main verify so speculative decoding
arithmetic favours longer ``block_size``.

Run::

    python -m alloy.examples.roofline.dflash.dsv4_pro_dflash               # default --level 1
    python -m alloy.examples.roofline.dflash.dsv4_pro_dflash --level 2
    python -m alloy.examples.roofline.dflash.dsv4_pro_dflash --level 3
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


def build_dsv4_pro_dflash_config():
    num_hidden_layers = 8
    return SimpleNamespace(
        # Top-level shape
        hidden_size=7168,
        vocab_size=129280,
        num_hidden_layers=num_hidden_layers,
        tie_word_embeddings=False,

        # Attention (Qwen3 dense, plain SDPA-style)
        num_attention_heads=64,        # = hidden / head_dim
        num_key_value_heads=8,         # GQA 1:8
        head_dim=128,
        sliding_window=None,

        # FFN (dense SwiGLU)
        intermediate_size=21504,       # ~3x hidden
        mlp_bias=False,

        # Layer composition — all plain full-attention + qwen3_mlp
        layer_types=["qwen3_attention"] * num_hidden_layers,
        ffn_types=["qwen3_mlp"] * num_hidden_layers,

        # DFlash-runtime metadata (not consumed by roofline, kept here so the
        # config object mirrors what the eventual HF config.json would carry).
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

    config = build_dsv4_pro_dflash_config()
    from collections import Counter
    print(f"DSV4-Pro-DFlash (Qwen3-dense draft, {config.num_hidden_layers} layers, hidden={config.hidden_size})")
    print(f"  layer_types: {dict(Counter(config.layer_types))}")
    print(f"  ffn_types  : {dict(Counter(config.ffn_types))}")
    print()

    my_device = CustomHardware(
        name="my-device", hbm_bandwidth=8e12,
        fp16=1000e12, bf16=1000e12, fp8=2000e12,
        # fp4 ~ 4 PFLOPS placeholder (not stored).
    )

    # Decode-heavy workload: draft is invoked once per speculative token,
    # so per-step latency at the operative context length is the dominant
    # number. Two decode cases (short + long) plus a prefill for cold-start.
    cases = [
        ("prefill (seq=8192)",         lambda hw: roofline_prefill(config, batch=1, seq_len=8192,                hardware=hw)),
        ("decode (cache=8192)",        lambda hw: roofline_decode (config, batch=1, kv_cache_len=8192,           hardware=hw)),
        ("decode (cache=131072)",      lambda hw: roofline_decode (config, batch=1, kv_cache_len=131072,         hardware=hw)),
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
