"""Hand-computed attention spec tests for all 3 DSV4 layer types.

For one fixed small config + input shape, this file decomposes flops and
bytes into the discrete contributions documented in the spec, asserts each
total, and lets the spec output be checked piece-by-piece. Three independent
test cases (sliding / HCA / CSA) verify the effective-KV-len abstraction.

Run::

    python -m alloy.tests.test_roofline_attention_specs
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from alloy.roofline import SPEC_REGISTRY, get_spec
from alloy.roofline.specs_attention import DSV4AttentionSpec


# --------------------------------------------------------------------------- #
# Shared fixture: small DSV4 config with all knobs needed for HCA/CSA/sliding.
# --------------------------------------------------------------------------- #


def _config():
    """Minimal but complete DSV4 config for attention spec testing.

    hidden=128, n_heads=4, head_dim=32 (so n_heads*head_dim = 128).
    sliding_window=64 — pick T=32 for "T <= W" so KV_len = T regime.
    """
    return SimpleNamespace(
        hidden_size=128,
        num_attention_heads=4,
        head_dim=32,
        q_lora_rank=64,
        o_groups=2,
        o_lora_rank=64,
        sliding_window=64,
        index_n_heads=4,
        index_head_dim=32,
        index_topk=8,
        compress_rates={
            "compressed_sparse_attention": 4,
            "heavily_compressed_attention": 8,
        },
    )


# B=1, T=32, H=128 -> n_tokens=32
B, T, H = 1, 32, 128
N = B * T
ES = 2  # bf16


def _projection_flops_expected():
    """q_a + q_b + kv (MQA single head) + o_a (grouped equiv) + o_b."""
    return (
        2 * N * 128 * 64        # q_a_proj 524,288
        + 2 * N * 64 * 128      # q_b_proj (qlr * (nh*hd) = 64 * 128) 524,288
        + 2 * N * 128 * 32      # kv_proj (single head, head_dim=32) 262,144
        + 2 * N * 128 * 64      # o_a_proj (nh*hd * o_lora_rank = 128 * 64) 524,288
        + 2 * N * 128 * 128     # o_b_proj (og*olr * hidden = 128 * 128) 1,048,576
    )  # total 2,883,584


def _projection_weight_bytes_expected():
    return (
        128 * 64 * ES           # q_a 16,384
        + 64 * 128 * ES         # q_b 16,384
        + 128 * 32 * ES         # kv 8,192
        + 128 * 64 * ES         # o_a (block-diagonal: total = nh*hd*olr) 16,384
        + 128 * 128 * ES        # o_b 32,768
        + 4 * ES                # sinks 8
    )  # total 90,120


# --------------------------------------------------------------------------- #
# Sliding-only attention
# --------------------------------------------------------------------------- #


def test_sliding_attention():
    """KV_len = min(T=32, W=64) = 32. No compressor, no indexer.

      proj flops          = 2,883,584
      SDPA  4*B*nh*T*KV_len*hd
                          = 4*1*4*32*32*32     =   524,288
      total flops                              = 3,407,872

      proj weights        = 90,120
      KV act bytes  = B*1*KV_len*hd*ES = 1*32*32*2 = 2,048
      act_in + act_out   = 2 * N*H*ES = 2*32*128*2  = 16,384
      total bytes                              = 108,552
    """
    spec = DSV4AttentionSpec("dsv4_sliding_attention")
    config = _config()

    flops = spec.flops((B, T, H), config)
    expected_flops = _projection_flops_expected() + 4 * B * 4 * T * 32 * 32
    assert flops == expected_flops, f"flops {flops} != {expected_flops}"

    b = spec.bytes((B, T, H), config, dtype=torch.bfloat16)
    expected_b = _projection_weight_bytes_expected() + 1 * 32 * 32 * ES + 2 * N * H * ES
    assert b == expected_b, f"bytes {b} != {expected_b}"

    print(f"[ok] sliding attention: flops={flops:,} bytes={b:,}")


# --------------------------------------------------------------------------- #
# HCA attention: sliding + heavy compressed entries
# --------------------------------------------------------------------------- #


def test_hca_attention():
    """KV_len = min(32, 64) + ceil(32/8) = 32 + 4 = 36.

      proj flops          = 2,883,584
      SDPA at KV_len=36   = 4*1*4*32*36*32     =   589,824
      HCA compressor proj = 2 * 2*32*128*32    =   524,288
      total flops                              = 3,997,696

      proj weights        = 90,120
      HCA compr weights:
        kv_proj  = 128*32*2 = 8,192
        gate_proj= 128*32*2 = 8,192
        pos_bias = 8*32*2   =   512
        kv_norm  = 32*2     =    64
        subtotal            = 16,960
      KV act = 1*36*32*2    =  2,304
      act_in + act_out      = 16,384
      total bytes           = 125,768
    """
    spec = DSV4AttentionSpec("dsv4_hca_attention")
    config = _config()

    expected_kv_len = 32 + math.ceil(32 / 8)
    assert expected_kv_len == 36

    flops = spec.flops((B, T, H), config)
    expected_flops = (
        _projection_flops_expected()
        + 4 * B * 4 * T * expected_kv_len * 32  # SDPA
        + 2 * 2 * T * 128 * 32                   # HCA compressor (kv_proj + gate_proj)
    )
    assert flops == expected_flops, f"flops {flops} != {expected_flops}"

    b = spec.bytes((B, T, H), config, dtype=torch.bfloat16)
    expected_b = (
        _projection_weight_bytes_expected()
        + (128 * 32 * ES + 128 * 32 * ES + 8 * 32 * ES + 32 * ES)  # HCA compressor weights
        + 1 * expected_kv_len * 32 * ES                             # KV act
        + 2 * N * H * ES                                            # act_in + act_out
    )
    assert b == expected_b, f"bytes {b} != {expected_b}"

    print(f"[ok] HCA attention: flops={flops:,} bytes={b:,}")


# --------------------------------------------------------------------------- #
# CSA attention: sliding + Lightning Indexer top-k entries
# --------------------------------------------------------------------------- #


def test_csa_attention():
    """KV_len = min(32, 64) + index_topk = 32 + 8 = 40.

      proj flops             = 2,883,584
      SDPA at KV_len=40      = 4*1*4*32*40*32     =    655,360
      CSA compressor proj    = 2 * 2*32*128*64    =  1,048,576
                                          (2*hd=64)
      Lightning Indexer:
        idx_kv_proj  = 2*32*128*64    =   524,288
        idx_gate_proj= 2*32*128*64    =   524,288
        idx_q_b_proj = 2*32*64*128    =   524,288
        weights_proj = 2*32*128*4     =    32,768
        score matmul = 2*1*32*4*32*8  =    65,536
                       (B*T*nih*ihd*n_compressed; n_compressed=ceil(32/4)=8)
        subtotal indexer              = 1,671,168
      total flops                     = 6,258,688

      proj weights              = 90,120
      CSA compressor weights:
        kv_proj  = 128*64*2     = 16,384
        gate_proj= 128*64*2     = 16,384
        pos_bias = 4*64*2       =    512
        kv_norm  = 32*2         =     64
        subtotal                = 33,344
      Indexer weights:
        kv_proj   = 128*64*2    = 16,384
        gate_proj = 128*64*2    = 16,384
        q_b_proj  = 64*128*2    = 16,384
        weights_proj=128*4*2    =  1,024
        pos_bias  = 4*64*2      =    512
        kv_norm   = 32*2        =     64
        subtotal                = 50,752
      KV act = 1*40*32*2        =  2,560
      act_in + act_out          = 16,384
      total bytes               = 193,160
    """
    spec = DSV4AttentionSpec("dsv4_csa_attention")
    config = _config()

    expected_kv_len = 32 + 8
    n_compressed = math.ceil(T / 4)
    assert n_compressed == 8

    # FLOPs decomposition
    proj_flops = _projection_flops_expected()
    sdpa_flops = 4 * B * 4 * T * expected_kv_len * 32
    csa_compressor_flops = 2 * 2 * T * 128 * 64  # 2*hd output dim
    indexer_flops = (
        2 * N * 128 * 64        # idx kv_proj
        + 2 * N * 128 * 64      # idx gate_proj
        + 2 * N * 64 * 128      # idx q_b_proj
        + 2 * N * 128 * 4       # weights_proj
        + 2 * B * T * 4 * 32 * n_compressed  # score matmul
    )
    expected_flops = proj_flops + sdpa_flops + csa_compressor_flops + indexer_flops

    flops = spec.flops((B, T, H), config)
    assert flops == expected_flops, f"flops {flops} != {expected_flops}"

    # Bytes decomposition
    proj_w = _projection_weight_bytes_expected()
    csa_comp_w = (
        128 * 64 * ES + 128 * 64 * ES + 4 * 64 * ES + 32 * ES
    )
    indexer_w = (
        128 * 64 * ES   # kv
        + 128 * 64 * ES # gate
        + 64 * 128 * ES # q_b
        + 128 * 4 * ES  # weights_proj
        + 4 * 64 * ES   # pos_bias
        + 32 * ES       # kv_norm
    )
    kv_act = 1 * expected_kv_len * 32 * ES
    act_in_out = 2 * N * H * ES
    expected_b = proj_w + csa_comp_w + indexer_w + kv_act + act_in_out

    b = spec.bytes((B, T, H), config, dtype=torch.bfloat16)
    assert b == expected_b, f"bytes {b} != {expected_b}"

    print(f"[ok] CSA attention: flops={flops:,} bytes={b:,}")


# --------------------------------------------------------------------------- #
# Long-context regime: confirm Lightning Indexer dominates at large T
# --------------------------------------------------------------------------- #


def test_csa_indexer_dominates_long_context():
    """At T=4096 the indexer score matmul (~T²/csa_rate) should outpace main
    SDPA (~T*KV_len) by a meaningful margin — this is the practical "indexer
    is heavy" claim that motivated treating it as a separate component."""
    config = _config()
    config.sliding_window = 256
    big_T = 4096

    spec = DSV4AttentionSpec("dsv4_csa_attention")
    flops_csa = spec.flops((1, big_T, 128), config)

    spec_sliding = DSV4AttentionSpec("dsv4_sliding_attention")
    flops_sliding = spec_sliding.flops((1, big_T, 128), config)

    # CSA should be substantially more expensive than sliding at long T
    # because the indexer adds T*T/csa_rate compute that grows quadratically.
    diff_ratio = flops_csa / flops_sliding
    assert diff_ratio > 1.5, (
        f"At T={big_T}, expected CSA flops >> sliding flops, got ratio {diff_ratio:.2f}"
    )
    print(f"[ok] CSA/sliding flops ratio at T={big_T}: {diff_ratio:.2f}x "
          f"({flops_csa:,} vs {flops_sliding:,})")


# --------------------------------------------------------------------------- #
# Wrong-shape guardrail
# --------------------------------------------------------------------------- #


def test_attention_spec_rejects_2d_in_shape():
    spec = DSV4AttentionSpec("dsv4_sliding_attention")
    config = _config()
    try:
        spec.flops((32, 128), config)
    except ValueError:
        print("[ok] attention spec rejects 2D in_shape")
        return
    raise AssertionError("expected ValueError on 2D in_shape")


def test_unknown_layer_type_rejected():
    try:
        DSV4AttentionSpec("definitely_not_dsv4")
    except ValueError:
        print("[ok] DSV4AttentionSpec rejects unknown layer_type")
        return
    raise AssertionError("expected ValueError on unknown layer_type")


# --------------------------------------------------------------------------- #
# Registration
# --------------------------------------------------------------------------- #


def test_all_attention_names_registered():
    expected = {"dsv4_sliding_attention", "dsv4_hca_attention", "dsv4_csa_attention"}
    missing = expected - set(SPEC_REGISTRY)
    assert not missing, f"missing attention spec registrations: {missing}"
    for name in expected:
        spec = get_spec(name)
        assert isinstance(spec, DSV4AttentionSpec)
        assert spec.layer_type == name
    print(f"[ok] all attention names registered: {sorted(expected)}")


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #


def main() -> int:
    test_sliding_attention()
    test_hca_attention()
    test_csa_attention()
    test_csa_indexer_dominates_long_context()
    test_attention_spec_rejects_2d_in_shape()
    test_unknown_layer_type_rejected()
    test_all_attention_names_registered()
    print("\nAll DSV4 attention spec tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
