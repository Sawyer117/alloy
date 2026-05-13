"""Roofline analysis for DeepSeek-V4-Flash on H100, Ascend910C, and a custom device.

Same reporting structure as ``dsv4_pro.py`` (level 1 cross-hardware compact,
level 2 per-module table, level 3 aggregated by ``(kind, name)``). The two
examples are deliberately parallel so users can A/B compare the two Flash
and Pro variants of DSV4 by running them back-to-back.

V4-Flash architecture (from
https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/main/config.json):

  43 hidden layers, hidden_size=4096, head_dim=512, MQA (num_kv_heads=1)
  q_lora_rank=1024, o_groups=8, o_lora_rank=1024
  Lightning Indexer: index_n_heads=64, index_head_dim=128, index_topk=512

  Per-layer attention flavor driven by ``compress_ratios``:
    ratio   0 -> dsv4_sliding_attention (sliding only, no long-range path)
    ratio 128 -> dsv4_hca_attention   (heavy compression, sliding + 1 entry / 128 tokens)
    ratio   4 -> dsv4_csa_attention   (CSA + Lightning Indexer top-512)

  FFN: first 3 layers use hash routing (``dsv4_hash_moe``), rest use topk
  (``dsv4_moe``); 256 routed experts + 1 always-on shared expert,
  num_experts_per_tok=6, moe_intermediate_size=2048.

V4-Pro vs V4-Flash composition difference:

  * V4-Pro main network is HCA / CSA alternating end-to-end, no
    sliding-only layers.
  * V4-Flash starts with **2 pure-sliding layers** (compress_ratio 0 at
    indices 0, 1) that act as local feature extractors before long-range
    pathways (HCA / CSA alternating) kick in at layer 2 onward.

Run::

    python -m alloy.examples.roofline.dsv4_flash               # default --level 1
    python -m alloy.examples.roofline.dsv4_flash --level 2
    python -m alloy.examples.roofline.dsv4_flash --level 3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch

from alloy.roofline import (
    CustomHardware,
    format_comparison,
    roofline_decode,
    roofline_mini_prefill,
    roofline_prefill,
)


# --------------------------------------------------------------------------- #
# V4-Flash config (verbatim from the HF repo) -> alloy roofline-compatible form
# --------------------------------------------------------------------------- #


# 44 entries; the last one (0) is for the next-token-prediction head, not a
# real hidden layer. Take the first 43 for the model proper. The first TWO
# entries are also 0 (pure-sliding layers at the network bottom) — those ARE
# main-network layers and stay in the slice.
COMPRESS_RATIOS = [
    0, 0, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128,
    4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128,
    4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128,
    4, 0,
]
NUM_HIDDEN_LAYERS = 43
NUM_HASH_LAYERS = 3

_RATIO_TO_LAYER_TYPE = {
    0:   "dsv4_sliding_attention",
    128: "dsv4_hca_attention",
    4:   "dsv4_csa_attention",
}


def build_v4_flash_config():
    layer_types = [_RATIO_TO_LAYER_TYPE[r] for r in COMPRESS_RATIOS[:NUM_HIDDEN_LAYERS]]
    ffn_types = (
        ["dsv4_hash_moe"] * NUM_HASH_LAYERS
        + ["dsv4_moe"] * (NUM_HIDDEN_LAYERS - NUM_HASH_LAYERS)
    )
    return SimpleNamespace(
        # Top-level shape
        hidden_size=4096,
        vocab_size=129280,
        num_hidden_layers=NUM_HIDDEN_LAYERS,
        tie_word_embeddings=False,

        # Attention (DSV4-Flash)
        num_attention_heads=64,
        num_key_value_heads=1,        # MQA
        head_dim=512,
        q_lora_rank=1024,
        o_groups=8,
        o_lora_rank=1024,
        sliding_window=128,
        # HCA at 128:1, CSA at 4:1
        compress_rates={
            "heavily_compressed_attention": 128,
            "compressed_sparse_attention":   4,
        },
        # Lightning Indexer (CSA only)
        index_n_heads=64,
        index_head_dim=128,
        index_topk=512,

        # MoE
        intermediate_size=2048,        # = moe_intermediate_size; shared expert reuses
        n_routed_experts=256,
        num_experts_per_tok=6,
        mlp_bias=False,
        scoring_func="sqrtsoftplus",
        routed_scaling_factor=1.5,

        # MHC (Manifold-constrained Hyper-Connections) — same shape as Pro:
        # hc_mult=4 parallel streams + 20 Sinkhorn iters per HyperConnection.
        use_mhc=True,
        hc_mult=4,
        hc_sinkhorn_iters=20,

        # Layer composition
        layer_types=layer_types,
        ffn_types=ffn_types,
    )


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--level", type=int, choices=[1, 2, 3], default=1,
        help="Report verbosity: 1=cross-hardware compact, 2=per-module table, "
             "3=aggregated by (kind, name).",
    )
    args = parser.parse_args()

    config = build_v4_flash_config()

    from collections import Counter
    layer_breakdown = Counter(config.layer_types)
    ffn_breakdown   = Counter(config.ffn_types)
    print(f"DeepSeek-V4-Flash composition  ({config.num_hidden_layers} layers, hidden={config.hidden_size})")
    print(f"  layer_types: {dict(layer_breakdown)}")
    print(f"  ffn_types  : {dict(ffn_breakdown)}")
    print()

    # Illustrative custom chip — same numbers as dsv4_pro for direct A/B.
    # fp4 ~ 4 PFLOPS placeholder, not stored (no torch.float4 dtype key).
    my_device = CustomHardware(
        name="my-device",
        hbm_bandwidth=8e12,
        fp16=1000e12,
        bf16=1000e12,                 # industry convention: BF16 = FP16
        fp8=2000e12,
    )

    cases = [
        ("prefill (seq=8192)",          lambda hw: roofline_prefill     (config, batch=1, seq_len=8192,                    hardware=hw)),
        ("mini-prefill (Q=512, P=8192)", lambda hw: roofline_mini_prefill(config, batch=1, chunk_len=512, kv_cache_len=8192, hardware=hw)),
        ("decode (cache=32768)",         lambda hw: roofline_decode      (config, batch=1, kv_cache_len=32768,              hardware=hw)),
    ]
    hardware_options = ["H100", "Ascend950PR", my_device]

    if args.level == 1:
        for case_name, runner in cases:
            reports = [runner(hw) for hw in hardware_options]
            print(format_comparison(reports, title=case_name))
            print()
    else:
        for case_name, runner in cases:
            for hw in hardware_options:
                r = runner(hw)
                print(r.format(level=args.level))
                print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
