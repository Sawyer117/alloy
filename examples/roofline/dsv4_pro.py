"""Roofline analysis for DeepSeek-V4-Pro on H100, Ascend910C, and a custom device.

Three reporting levels:

  * ``--level 1`` (default) — cross-hardware compact: shared FLOPs/bytes/AI
    per case + one row per hardware showing time, throughput, bottleneck.
    Best for "is it fast enough on this device?".
  * ``--level 2`` — full per-module table per (case, hardware), with bound
    ('C' compute / 'M' memory) and %time columns. Best for "which layer is
    the bottleneck?".
  * ``--level 3`` — per (case, hardware), rows aggregated by ``(kind, name)``
    so e.g. all 58 ``ffn:dsv4_moe`` layers collapse into one row showing
    combined contribution. Best for architectural decisions.


V4-Pro architecture (from
https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/config.json):

  61 hidden layers, hidden_size=7168, head_dim=512, MQA (num_kv_heads=1)
  q_lora_rank=1536, o_groups=16, o_lora_rank=1024
  Lightning Indexer: index_n_heads=64, index_head_dim=128, index_topk=1024

  Per-layer attention flavor driven by ``compress_ratios``:
    ratio 128 -> dsv4_hca_attention   (heavy compression, sliding + 1 entry / 128 tokens)
    ratio   4 -> dsv4_csa_attention   (CSA + Lightning Indexer top-1024)
    ratio   0 -> dsv4_sliding_attention (sliding only, no long-range path)

  FFN: first 3 layers use hash routing (``dsv4_hash_moe``), rest use topk
  (``dsv4_moe``); 384 routed experts + 1 always-on shared expert,
  num_experts_per_tok=6, moe_intermediate_size=3072.

Total params: ~671 B. Per-forward HBM traffic ~ 1.3 TB for the weights alone
in bf16, which dominates decode latency on every accelerator below ~10 TB/s
HBM.

Run::

    python -m alloy.examples.roofline.dsv4_pro                # default --level 1
    python -m alloy.examples.roofline.dsv4_pro --level 2
    python -m alloy.examples.roofline.dsv4_pro --level 3
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
# V4-Pro config (verbatim from the HF repo) -> alloy roofline-compatible form
# --------------------------------------------------------------------------- #


# 62 entries; the last one (0) is for the next-token-prediction head, not a
# real hidden layer. Take the first 61 for the model proper.
COMPRESS_RATIOS = [
    128, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4,
    128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128,
    4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4,
    128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128,
    4, 0,
]
NUM_HIDDEN_LAYERS = 61
NUM_HASH_LAYERS = 3

_RATIO_TO_LAYER_TYPE = {
    128: "dsv4_hca_attention",
    4:   "dsv4_csa_attention",
    0:   "dsv4_sliding_attention",
}


def build_v4_pro_config():
    layer_types = [_RATIO_TO_LAYER_TYPE[r] for r in COMPRESS_RATIOS[:NUM_HIDDEN_LAYERS]]
    ffn_types = (
        ["dsv4_hash_moe"] * NUM_HASH_LAYERS
        + ["dsv4_moe"] * (NUM_HIDDEN_LAYERS - NUM_HASH_LAYERS)
    )
    return SimpleNamespace(
        # Top-level shape
        hidden_size=7168,
        vocab_size=129280,
        num_hidden_layers=NUM_HIDDEN_LAYERS,
        tie_word_embeddings=False,

        # Attention (DSV4-Pro)
        num_attention_heads=128,
        num_key_value_heads=1,        # MQA
        head_dim=512,
        q_lora_rank=1536,
        o_groups=16,
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
        index_topk=1024,

        # MoE
        intermediate_size=3072,        # = moe_intermediate_size; shared expert reuses
        n_routed_experts=384,
        num_experts_per_tok=6,
        mlp_bias=False,
        scoring_func="sqrtsoftplus",
        routed_scaling_factor=2.5,

        # MHC (Manifold-constrained Hyper-Connections) — V4-Pro carries
        # hc_mult=4 parallel residual streams and runs 20 Sinkhorn iters
        # per HyperConnection site (2 sites per layer + 1 final HyperHead).
        use_mhc=True,
        hc_mult=4,
        hc_sinkhorn_iters=20,

        # Layer composition
        layer_types=layer_types,
        ffn_types=ffn_types,
    )


# --------------------------------------------------------------------------- #
# Hardware: two presets + one custom-built device
# --------------------------------------------------------------------------- #


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--level", type=int, choices=[1, 2, 3], default=1,
        help="Report verbosity: 1=cross-hardware compact, 2=per-module table, "
             "3=aggregated by (kind, name).",
    )
    args = parser.parse_args()

    config = build_v4_pro_config()

    # Sanity: count layer flavors so the user sees what they're roofline-ing
    from collections import Counter
    layer_breakdown = Counter(config.layer_types)
    ffn_breakdown   = Counter(config.ffn_types)
    print(f"DeepSeek-V4-Pro composition  ({config.num_hidden_layers} layers, hidden={config.hidden_size})")
    print(f"  layer_types: {dict(layer_breakdown)}")
    print(f"  ffn_types  : {dict(ffn_breakdown)}")
    print()

    # A custom chip — illustrative placeholders. fp4 throughput (4 PFLOPS,
    # 2x fp8) is shown in the comment but not stored in peak_flops because
    # alloy roofline runs in bf16 here and torch has no float4 dtype key.
    my_device = CustomHardware(
        name="my-device",
        hbm_bandwidth=8e12,
        fp16=1000e12,                 # 1 PFLOPS FP16
        bf16=1000e12,                 # 1 PFLOPS BF16 (industry convention: BF16 = FP16)
        fp8=2000e12,                  # 2 PFLOPS FP8
        # fp4 ~ 4 PFLOPS (twice fp8) — placeholder, not stored.
    )

    # Three serving modes at production-scale shapes
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
