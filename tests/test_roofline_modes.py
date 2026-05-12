"""Hand-computed tests for prefill / mini-prefill / decode modes.

Verifies the three core behaviors that distinguish modes:

  1. **Effective KV length scales with cache** — sliding/HCA/CSA layer types
     all see longer KV when ``kv_cache_len > 0`` (up to sliding_window cap).
  2. **Convenience wrappers compose** — ``roofline_prefill(seq=T)`` is
     equivalent to ``roofline_mini_prefill(chunk=T, cache=0)``.
  3. **Mode label inference** — ``RooflineReport.mode`` correctly identifies
     prefill / decode / mini-prefill from ``(query_len, kv_cache_len)``.

Plus a few cross-cutting sanity checks:

  * Decode at large cache is memory-bound on typical hardware (low AI)
  * Prefill of T tokens has more flops than decode @ kv_cache=T
  * Lightning Indexer's score-matmul flops grow with kv_cache_len for CSA

Run::

    python -m alloy.tests.test_roofline_modes
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from alloy.roofline import (
    roofline,
    roofline_decode,
    roofline_mini_prefill,
    roofline_prefill,
)
from alloy.roofline.specs_attention import (
    DSV4AttentionSpec,
    _effective_kv_len,
    _indexer_flops,
)


# --------------------------------------------------------------------------- #
# Shared config (matches the attention-spec test)
# --------------------------------------------------------------------------- #


def _config():
    return SimpleNamespace(
        # AlloyConfig-like fields
        hidden_size=128,
        vocab_size=1024,
        num_hidden_layers=2,
        layer_types=["dsv4_hca_attention", "dsv4_csa_attention"],
        ffn_types=["dsv4_moe", "dsv4_moe"],
        # Attention
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
        # MoE
        intermediate_size=64,
        n_routed_experts=8,
        num_experts_per_tok=2,
        mlp_bias=False,
        tie_word_embeddings=False,
    )


# --------------------------------------------------------------------------- #
# Effective KV length: prefill vs mini-prefill vs decode
# --------------------------------------------------------------------------- #


def test_effective_kv_len_sliding():
    """Sliding KV_len = min(Q+P, sliding_window). Caps cleanly at W."""
    config = _config()
    # prefill: T=32 < W=64  -> KV_len = 32
    assert _effective_kv_len(32, "dsv4_sliding_attention", config, kv_cache_len=0) == 32
    # mini-prefill: Q=16, P=32  -> total=48 < W=64 -> KV_len = 48
    assert _effective_kv_len(16, "dsv4_sliding_attention", config, kv_cache_len=32) == 48
    # decode at large cache: Q=1, P=4096 -> total=4097 > W=64 -> KV_len = 64 (capped)
    assert _effective_kv_len(1, "dsv4_sliding_attention", config, kv_cache_len=4096) == 64
    print("[ok] effective_kv_len sliding (caps at W)")


def test_effective_kv_len_hca():
    """HCA KV_len = sliding cap + ceil((Q+P)/hca_rate=8) compressed entries."""
    config = _config()
    # prefill T=32: sliding=32, compressed=ceil(32/8)=4 -> 36
    assert _effective_kv_len(32, "dsv4_hca_attention", config, kv_cache_len=0) == 36
    # decode Q=1, P=4096: sliding=64 (capped), compressed=ceil(4097/8)=513 -> 577
    assert _effective_kv_len(1, "dsv4_hca_attention", config, kv_cache_len=4096) == 577
    print("[ok] effective_kv_len HCA (compressed grows with cache)")


def test_effective_kv_len_csa():
    """CSA KV_len = sliding cap + index_topk (constant gate; doesn't grow)."""
    config = _config()
    assert _effective_kv_len(32, "dsv4_csa_attention", config, kv_cache_len=0) == 32 + 8
    # Even at huge cache, CSA's effective KV is bounded by index_topk
    assert _effective_kv_len(1, "dsv4_csa_attention", config, kv_cache_len=4096) == 64 + 8
    print("[ok] effective_kv_len CSA (gated by index_topk)")


# --------------------------------------------------------------------------- #
# Lightning Indexer: score matmul grows with cache
# --------------------------------------------------------------------------- #


def test_indexer_flops_grow_with_cache():
    """The score matmul (q vs cached compressed) is O(Q * (Q+P)/csa_rate).
    For decode (Q=1), score flops = 2 * B * 1 * nih * ihd * ceil(P/csa_rate)
    grows linearly with P. Other indexer flops (projections) scale only with
    Q (= 1 in decode) and stay tiny."""
    config = _config()
    # decode Q=1, ascending cache
    flops_p0 = _indexer_flops(1, 1, config, kv_cache_len=0)
    flops_p64 = _indexer_flops(1, 1, config, kv_cache_len=64)
    flops_p4096 = _indexer_flops(1, 1, config, kv_cache_len=4096)

    # Score-matmul portion: 2 * 1 * 1 * 4 * 32 * ceil((1+P)/4)
    score_p0 = 2 * 1 * 1 * 4 * 32 * math.ceil(1 / 4)        #   256
    score_p64 = 2 * 1 * 1 * 4 * 32 * math.ceil((1+64)/4)    # 4,224
    score_p4096 = 2 * 1 * 1 * 4 * 32 * math.ceil((1+4096)/4)  # 262,464

    # Non-score (projection) flops are constant w.r.t. P
    non_score_p0 = flops_p0 - score_p0
    non_score_p64 = flops_p64 - score_p64
    non_score_p4096 = flops_p4096 - score_p4096
    assert non_score_p0 == non_score_p64 == non_score_p4096, (
        f"projection flops should be constant: {non_score_p0} {non_score_p64} {non_score_p4096}"
    )
    # Score flops should match hand-compute
    assert flops_p4096 - non_score_p4096 == score_p4096
    print(f"[ok] indexer score flops grow with cache: "
          f"P=0->{score_p0:,}, P=64->{score_p64:,}, P=4096->{score_p4096:,}")


# --------------------------------------------------------------------------- #
# Convenience wrapper composition
# --------------------------------------------------------------------------- #


def test_prefill_equals_zero_cache_mini_prefill():
    """roofline_prefill(seq=T) must equal roofline_mini_prefill(chunk=T, cache=0).
    They take different argument structures but reduce to the same call."""
    config = _config()
    prefill = roofline_prefill(config, batch=1, seq_len=32, hardware="A100")
    mini = roofline_mini_prefill(config, batch=1, chunk_len=32, kv_cache_len=0, hardware="A100")
    assert prefill.total_flops == mini.total_flops
    assert prefill.total_bytes == mini.total_bytes
    # mode label differs (prefill vs mini-prefill at cache=0 is awkward edge case;
    # we treat cache=0 as prefill regardless of which wrapper called)
    assert prefill.mode == "prefill"
    assert mini.mode == "prefill"  # cache=0 -> prefill, even from mini-prefill wrapper
    print("[ok] roofline_prefill(T) == roofline_mini_prefill(T, cache=0)")


def test_decode_equals_query_one():
    """roofline_decode(cache=P) == roofline(query_len=1, kv_cache_len=P)."""
    config = _config()
    decode_wrap = roofline_decode(config, batch=1, kv_cache_len=4096, hardware="A100")
    decode_core = roofline(config, batch=1, query_len=1, kv_cache_len=4096, hardware="A100")
    assert decode_wrap.total_flops == decode_core.total_flops
    assert decode_wrap.total_bytes == decode_core.total_bytes
    assert decode_wrap.mode == "decode"
    assert decode_core.mode == "decode"
    print("[ok] roofline_decode(P) == roofline(query=1, cache=P)")


# --------------------------------------------------------------------------- #
# Mode-aware behaviors
# --------------------------------------------------------------------------- #


def test_decode_at_large_cache_is_memory_bound():
    """Decode is the canonical memory-bound regime — model weights dominate
    HBM traffic relative to the tiny single-query compute."""
    config = _config()
    decode = roofline_decode(config, batch=1, kv_cache_len=4096, hardware="A100")
    assert decode.bottleneck == "memory", (
        f"decode at cache=4096 expected memory-bound, got {decode.bottleneck} "
        f"(AI={decode.arithmetic_intensity:.1f}, compute={decode.compute_time_s*1e6:.2f}us, "
        f"memory={decode.memory_time_s*1e6:.2f}us)"
    )
    # AI should be very low (close to 1) — decode reads ~ as many bytes as it does flops
    assert decode.arithmetic_intensity < 5, (
        f"decode AI expected << prefill, got {decode.arithmetic_intensity:.1f}"
    )
    print(f"[ok] decode @ cache=4096: AI={decode.arithmetic_intensity:.1f}, bottleneck=memory")


def test_prefill_more_compute_than_decode():
    """Prefill of T tokens does T queries × KV_len attention compute,
    so its flops dwarf a single-token decode at the same cache state."""
    config = _config()
    prefill = roofline_prefill(config, batch=1, seq_len=4096, hardware="A100")
    decode = roofline_decode(config, batch=1, kv_cache_len=4096, hardware="A100")
    ratio = prefill.total_flops / decode.total_flops
    # Prefill should be hundreds to thousands of times more compute
    assert ratio > 100, f"prefill/decode flop ratio expected >>100, got {ratio:.1f}"
    print(f"[ok] prefill flops / decode flops = {ratio:.1f}x")


def test_mini_prefill_between_prefill_and_decode():
    """Mini-prefill flops should sit between decode (Q=1) and full prefill (Q=T)."""
    config = _config()
    decode = roofline_decode(config, batch=1, kv_cache_len=2048, hardware="A100")
    mini = roofline_mini_prefill(config, batch=1, chunk_len=512, kv_cache_len=2048, hardware="A100")
    prefill = roofline_prefill(config, batch=1, seq_len=2560, hardware="A100")
    assert decode.total_flops < mini.total_flops < prefill.total_flops, (
        f"expected decode < mini < prefill: {decode.total_flops} vs {mini.total_flops} "
        f"vs {prefill.total_flops}"
    )
    print(f"[ok] flops ordering: decode({decode.total_flops:,}) < "
          f"mini({mini.total_flops:,}) < prefill({prefill.total_flops:,})")


# --------------------------------------------------------------------------- #
# Mode label inference
# --------------------------------------------------------------------------- #


def test_mode_label_inference():
    """RooflineReport.mode should correctly identify the regime."""
    config = _config()

    # Prefill: cache=0
    r = roofline(config, batch=1, query_len=64, kv_cache_len=0)
    assert r.mode == "prefill"

    # Decode: query=1, cache>0
    r = roofline(config, batch=1, query_len=1, kv_cache_len=512)
    assert r.mode == "decode"

    # Mini-prefill: query>1, cache>0
    r = roofline(config, batch=1, query_len=128, kv_cache_len=512)
    assert r.mode == "mini-prefill"

    # Edge case: query=1 with cache=0 — counts as prefill (cold start, single token)
    r = roofline(config, batch=1, query_len=1, kv_cache_len=0)
    assert r.mode == "prefill"

    print("[ok] mode label inference (prefill/decode/mini-prefill)")


def test_mode_label_in_report_str():
    """The pretty-printed report header reflects the mode."""
    config = _config()
    s_prefill = str(roofline_prefill(config, batch=1, seq_len=64))
    s_decode = str(roofline_decode(config, batch=1, kv_cache_len=512))
    s_mini = str(roofline_mini_prefill(config, batch=1, chunk_len=128, kv_cache_len=512))
    assert "prefill (B=1, Q=64)" in s_prefill
    assert "decode (B=1, cache=512)" in s_decode
    assert "mini-prefill (B=1, Q=128, cache=512)" in s_mini
    print("[ok] report header shows correct mode label")


# --------------------------------------------------------------------------- #
# DSV4AttentionSpec: kv_cache_len changes flops appropriately
# --------------------------------------------------------------------------- #


def test_attention_spec_uses_kv_cache_len():
    """DSV4 attention spec flops/bytes should change when kv_cache_len changes."""
    config = _config()
    spec = DSV4AttentionSpec("dsv4_hca_attention")
    # Same in_shape, different cache. Query=1 isolates the kv_cache_len effect.
    flops_cache0 = spec.flops((1, 1, 128), config, kv_cache_len=0)
    flops_cache4096 = spec.flops((1, 1, 128), config, kv_cache_len=4096)
    # SDPA flops scale with KV_len; KV_len jumps from ~1 to ~577 (sliding cap + compressed)
    assert flops_cache4096 > flops_cache0
    bytes_cache0 = spec.bytes((1, 1, 128), config, dtype=torch.bfloat16, kv_cache_len=0)
    bytes_cache4096 = spec.bytes((1, 1, 128), config, dtype=torch.bfloat16, kv_cache_len=4096)
    # KV-act bytes also grow with effective KV_len
    assert bytes_cache4096 > bytes_cache0
    print(f"[ok] attention spec sees kv_cache_len: "
          f"flops {flops_cache0:,} -> {flops_cache4096:,}, "
          f"bytes {bytes_cache0:,} -> {bytes_cache4096:,}")


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #


def main() -> int:
    test_effective_kv_len_sliding()
    test_effective_kv_len_hca()
    test_effective_kv_len_csa()
    test_indexer_flops_grow_with_cache()
    test_prefill_equals_zero_cache_mini_prefill()
    test_decode_equals_query_one()
    test_decode_at_large_cache_is_memory_bound()
    test_prefill_more_compute_than_decode()
    test_mini_prefill_between_prefill_and_decode()
    test_mode_label_inference()
    test_mode_label_in_report_str()
    test_attention_spec_uses_kv_cache_len()
    print("\nAll roofline mode tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
