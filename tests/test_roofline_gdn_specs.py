"""Hand-computed tests for the Gated DeltaNet (qwen3_5_gdn) spec.

GDN has no public FLOPs reference (flame uses the wrong generic transformer
formula; fla has none) so we can't ground-truth against another tool. These
tests instead check **internal consistency** of the spec across modes and
config knobs:

  1. Hand-computed numerical equality for a small known config (one round
     each through the algebraic formulas).
  2. Mode-dispatch boundary: query_len <= 64 -> recurrent, > 64 -> chunk.
  3. Per-token FLOPs is **constant** in chunk mode (linear-attention
     property — the headline difference vs softmax attention's T-quadratic).
  4. Recurrent state bytes are **independent** of kv_cache_len (the
     headline GDN advantage at long context vs softmax KV cache).
  5. State I/O **doubles** when kv_cache_len > 0 (read added on top of
     the always-present write).
  6. Decode (q_len=1) is dominated by linear projections + state read,
     yielding very low AI (memory-bound regime).
  7. fla docstring sanity: param count for a balanced config (where
     ``key_dim = 0.75*H, value_dim = 1.5*H``) lands near ``6*H^2``.

Run::

    python -m alloy.tests.test_roofline_gdn_specs
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from alloy.roofline import SPEC_REGISTRY, get_spec
from alloy.roofline.specs_gdn import GDN_CHUNK_SIZE, Qwen35GDNSpec


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


def _config():
    """Small GDN config (alloy field names) for byte-exact hand-compute."""
    return SimpleNamespace(
        hidden_size=64,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_key_head_dim=16,
        linear_value_head_dim=32,    # expand_v=2 effectively
        linear_conv_kernel_dim=4,
    )


# Derived dims (cached for hand-compute reuse)
H = 64
H_k = 2     # num_k_heads
H_v = 2     # num_v_heads
K = 16      # head_k_dim
V = 32      # head_v_dim
KD = H_k * K     # key_dim = 32
VD = H_v * V     # value_dim = 64
CONV_DIM = 2 * KD + VD  # 128
CONV_SIZE = 4
ES = 2  # bf16


def _linear_flops_per_token():
    """Sum of all 5 linear projections, hand-counted."""
    return (
        2 * H * (2 * KD + VD)   # in_proj_qkv:  H -> 2*KD + VD
        + 2 * H * VD            # in_proj_z:    H -> VD
        + 2 * H * H_v           # in_proj_b
        + 2 * H * H_v           # in_proj_a
        + 2 * VD * H            # out_proj:     VD -> H
    )


def _linear_weight_bytes():
    return (
        H * (2 * KD + VD) * ES
        + H * VD * ES
        + H * H_v * ES
        + H * H_v * ES
        + VD * H * ES
    )


def _conv_flops_per_token():
    return 2 * CONV_SIZE * CONV_DIM


def _conv_weight_bytes():
    return CONV_SIZE * CONV_DIM * ES


def _other_weight_bytes():
    """RMSNormGated gain (V) + A_log + dt_bias (per v-head, fp32)."""
    return V * ES + 2 * H_v * 4


def _state_size():
    return 1 * H_v * K * V * ES   # batch=1


def _conv_state_size():
    return 1 * (CONV_SIZE - 1) * CONV_DIM * ES  # batch=1


# --------------------------------------------------------------------------- #
# Hand-computed numerical tests
# --------------------------------------------------------------------------- #


def test_decode_flops_and_bytes():
    """Decode: query_len=1, kv_cache_len=4096 -> recurrent mode, full state I/O.

    flops = linear + conv + 1 * H_v * 7 * K * V
          = 33,280 + 1,024 + 7,168 = 41,472

    bytes(bf16):
      weights  = linear (33,280) + conv (1,024) + norm (64) + A_log/dt_bias (16)
               = 34,384
      state_io = 2*state_size = 2*2,048 = 4,096   (read + write, cache>0)
      conv_state_io = 2*(3*128*2) = 1,536
      act      = 2*(1*64*2) = 256
      total    = 40,272
    """
    spec = Qwen35GDNSpec()
    cfg = _config()

    flops = spec.flops((1, 1, H), cfg, kv_cache_len=4096)
    expected_flops = (
        _linear_flops_per_token() * 1   # n_tokens=1
        + _conv_flops_per_token() * 1
        + 1 * H_v * 7 * K * V             # recurrent (q_len=1 < chunk_size)
    )
    assert flops == expected_flops, f"flops {flops} != {expected_flops}"

    bytes_ = spec.bytes((1, 1, H), cfg, dtype=torch.bfloat16, kv_cache_len=4096)
    expected_bytes = (
        _linear_weight_bytes()
        + _conv_weight_bytes()
        + _other_weight_bytes()
        + 2 * _state_size()              # read + write (cache_len > 0)
        + 2 * _conv_state_size()
        + 2 * 1 * H * ES                  # act_in + act_out, n_tokens=1
    )
    assert bytes_ == expected_bytes, f"bytes {bytes_} != {expected_bytes}"

    print(f"[ok] decode: flops={flops:,}, bytes={bytes_:,}")


def test_prefill_chunk_flops_and_bytes():
    """Prefill q_len=128, cache=0 -> chunk mode, state write only.

    chunk attn per token per head: 2*C*(3*K + 2*V) + 4*K*V
                                  = 2*64*(48 + 64) + 4*16*32
                                  = 128*112 + 2048
                                  = 14,336 + 2,048 = 16,384
    × n_tokens=128, × H_v=2: 4,194,304

    flops = 4,259,840 (linear*128) + 131,072 (conv*128) + 4,194,304 (chunk attn)
          = 8,585,216

    bytes:
      weights         = 34,384
      state_io        = state_size only (write, no read)         = 2,048
      conv_state_io   = conv_state_size only                     = 768
      act             = 2 * 128 * 64 * 2                         = 32,768
      total                                                      = 69,968
    """
    spec = Qwen35GDNSpec()
    cfg = _config()
    n = 128

    flops = spec.flops((1, n, H), cfg, kv_cache_len=0)
    expected_flops = (
        _linear_flops_per_token() * n
        + _conv_flops_per_token() * n
        + n * H_v * (2 * GDN_CHUNK_SIZE * (3 * K + 2 * V) + 4 * K * V)
    )
    assert flops == expected_flops, f"flops {flops} != {expected_flops}"

    bytes_ = spec.bytes((1, n, H), cfg, dtype=torch.bfloat16, kv_cache_len=0)
    expected_bytes = (
        _linear_weight_bytes()
        + _conv_weight_bytes()
        + _other_weight_bytes()
        + _state_size()                # write only
        + _conv_state_size()
        + 2 * n * H * ES                # act_in + act_out
    )
    assert bytes_ == expected_bytes, f"bytes {bytes_} != {expected_bytes}"
    print(f"[ok] prefill (q={n}): flops={flops:,}, bytes={bytes_:,}")


def test_mini_prefill_flops_unchanged_bytes_grow():
    """Mini-prefill (q=128, cache=1024) vs prefill (q=128, cache=0):

      flops: identical (cache_len doesn't change compute for GDN — state is
             a fixed-size summary, doesn't grow with prefix)
      bytes: state I/O grows by state_size + conv_state_size (reads added)
    """
    spec = Qwen35GDNSpec()
    cfg = _config()

    flops_prefill = spec.flops((1, 128, H), cfg, kv_cache_len=0)
    flops_mini = spec.flops((1, 128, H), cfg, kv_cache_len=1024)
    assert flops_prefill == flops_mini, (
        f"GDN compute should not depend on kv_cache_len: {flops_prefill} != {flops_mini}"
    )

    b_prefill = spec.bytes((1, 128, H), cfg, dtype=torch.bfloat16, kv_cache_len=0)
    b_mini = spec.bytes((1, 128, H), cfg, dtype=torch.bfloat16, kv_cache_len=1024)
    expected_bytes_diff = _state_size() + _conv_state_size()
    assert b_mini - b_prefill == expected_bytes_diff, (
        f"state read should add {expected_bytes_diff} bytes, got diff {b_mini - b_prefill}"
    )
    print(f"[ok] mini-prefill: flops same, bytes +{b_mini - b_prefill} (state read)")


# --------------------------------------------------------------------------- #
# Mode dispatch boundary
# --------------------------------------------------------------------------- #


def test_mode_dispatch_at_chunk_size_boundary():
    """Boundary: q_len <= GDN_CHUNK_SIZE (64) -> recurrent; > 64 -> chunk.

    Per token per head:
      recurrent: 7*K*V             = 7*16*32 = 3,584
      chunk:     2*C*(3K+2V) + 4*K*V = 16,384
    Chunk does ~4.5× more attention compute per token at this small config
    (the numerical break-even depends on K, V, C — chunk wins on bytes by
    enabling matmul efficiency, not on flop count).
    """
    spec = Qwen35GDNSpec()
    cfg = _config()

    # At q_len = chunk_size: still recurrent (boundary is strict >)
    f_at_boundary = spec.flops((1, GDN_CHUNK_SIZE, H), cfg)
    expected_at = (
        _linear_flops_per_token() * GDN_CHUNK_SIZE
        + _conv_flops_per_token() * GDN_CHUNK_SIZE
        + GDN_CHUNK_SIZE * H_v * 7 * K * V    # recurrent
    )
    assert f_at_boundary == expected_at, "q_len = chunk_size should still be recurrent"

    # At q_len = chunk_size + 1: switches to chunk
    f_above = spec.flops((1, GDN_CHUNK_SIZE + 1, H), cfg)
    expected_above = (
        _linear_flops_per_token() * (GDN_CHUNK_SIZE + 1)
        + _conv_flops_per_token() * (GDN_CHUNK_SIZE + 1)
        + (GDN_CHUNK_SIZE + 1) * H_v
            * (2 * GDN_CHUNK_SIZE * (3 * K + 2 * V) + 4 * K * V)
    )
    assert f_above == expected_above, "q_len > chunk_size should be chunk mode"
    print(f"[ok] mode boundary at q_len={GDN_CHUNK_SIZE}: recurrent;  "
          f"q_len={GDN_CHUNK_SIZE+1}: chunk")


# --------------------------------------------------------------------------- #
# Linear-attention invariants
# --------------------------------------------------------------------------- #


def test_chunk_mode_flops_linear_in_query_len():
    """The headline GDN property: flops/token is constant in chunk mode.

    Total flops should scale exactly linearly with query_len when both
    operating points are above the chunk boundary. Pick q=128 and q=512,
    expect total flops to scale 4×."""
    spec = Qwen35GDNSpec()
    cfg = _config()

    f_128 = spec.flops((1, 128, H), cfg, kv_cache_len=0)
    f_512 = spec.flops((1, 512, H), cfg, kv_cache_len=0)
    assert f_512 == f_128 * 4, (
        f"chunk-mode flops should be linear in q_len: 4*{f_128:,} != {f_512:,}"
    )
    print(f"[ok] chunk-mode flops linear in query_len: 128->{f_128:,}, 512->{f_512:,} (4x)")


def test_state_bytes_independent_of_cache_length():
    """The headline GDN advantage: state size is **constant** in cache_len.

    Compare decode at cache=0 (cold) vs cache=4096 vs cache=65536. The
    spec.bytes total may differ (state read added when cache>0), but the
    state size itself stays at H_v * K * V * es regardless of cache_len."""
    spec = Qwen35GDNSpec()
    cfg = _config()

    # Same query (q_len=1), three cache_len values
    b_cold = spec.bytes((1, 1, H), cfg, dtype=torch.bfloat16, kv_cache_len=0)
    b_4k = spec.bytes((1, 1, H), cfg, dtype=torch.bfloat16, kv_cache_len=4096)
    b_64k = spec.bytes((1, 1, H), cfg, dtype=torch.bfloat16, kv_cache_len=65536)
    # cache>0 adds exactly state_size + conv_state_size (read), regardless of cache_len value
    delta_4k = b_4k - b_cold
    delta_64k = b_64k - b_cold
    expected_delta = _state_size() + _conv_state_size()
    assert delta_4k == expected_delta == delta_64k, (
        f"state I/O should be cache-length independent: 4k->{delta_4k}, 64k->{delta_64k}, "
        f"expected {expected_delta}"
    )
    print(f"[ok] state I/O constant in cache_len (4k, 64k both add {expected_delta} B)")


def test_decode_is_memory_bound():
    """Decode regime: linear projections dominate flops, weights dominate
    bytes, AI ~= ratio of (flops/token) / (bytes/token). At q_len=1 with
    weights >> per-token flop count, AI should be small (memory-bound)."""
    spec = Qwen35GDNSpec()
    cfg = _config()

    flops = spec.flops((1, 1, H), cfg, kv_cache_len=4096)
    bytes_ = spec.bytes((1, 1, H), cfg, dtype=torch.bfloat16, kv_cache_len=4096)
    ai = flops / bytes_
    # For our small config, AI should be very low — much less than typical
    # compute/bandwidth ratios (A100 bf16 ratio ~= 156).
    assert ai < 5, f"GDN decode AI expected << 156, got {ai:.2f}"
    print(f"[ok] decode AI = {ai:.2f} (memory-bound regime)")


# --------------------------------------------------------------------------- #
# fla docstring sanity check (informational)
# --------------------------------------------------------------------------- #


def test_param_count_balanced_config_near_6h_squared():
    """fla docstring claims ~6*H^2 params per layer for a balanced config:

      key_dim = 0.75*H,  value_dim = 1.5*H,  use_gate=True

    Build a config that satisfies this and check the spec's weight bytes
    land near 6*H^2 * sizeof(dtype). Tolerance is wide (±15%) — the
    docstring lumps small terms (conv, A_log, dt_bias, norm, b/a-proj)
    into "ignorably small", we count them all explicitly.
    """
    H_test = 256
    cfg = SimpleNamespace(
        hidden_size=H_test,
        linear_num_key_heads=4,
        linear_num_value_heads=4,
        linear_key_head_dim=int(0.75 * H_test) // 4,   # key_dim = 0.75*H
        linear_value_head_dim=int(1.5 * H_test) // 4,  # value_dim = 1.5*H
        linear_conv_kernel_dim=4,
    )
    spec = Qwen35GDNSpec()
    # bytes() includes activations + state I/O; isolate weights only by
    # subtracting those out. Easier path: re-derive from helpers.
    from alloy.roofline.specs_gdn import (
        _conv_weight_bytes,
        _linear_proj_weight_bytes,
        _read_dims,
    )
    d = _read_dims(cfg)
    weight_bytes = _linear_proj_weight_bytes(d, es=2) + _conv_weight_bytes(d, es=2)
    weight_bytes += d["head_v_dim"] * 2 + 2 * d["num_v_heads"] * 4
    weight_params = weight_bytes / 2  # bf16 -> 2 B/param (mostly)

    expected_params = 6 * H_test * H_test
    ratio = weight_params / expected_params
    assert 0.85 <= ratio <= 1.15, (
        f"docstring ~6*H^2 check: got {weight_params:,} params vs expected {expected_params:,} "
        f"(ratio {ratio:.3f}, expected within [0.85, 1.15])"
    )
    print(f"[ok] balanced-config param count ~= 6*H^2: "
          f"{weight_params:,} ~= {expected_params:,} (ratio {ratio:.3f})")


# --------------------------------------------------------------------------- #
# Registration
# --------------------------------------------------------------------------- #


def test_registered_under_qwen3_5_gdn_name():
    spec = get_spec("qwen3_5_gdn")
    assert spec is not None, "qwen3_5_gdn should be registered"
    assert isinstance(spec, Qwen35GDNSpec)
    print("[ok] qwen3_5_gdn registered")


# --------------------------------------------------------------------------- #
# Wrong-shape guardrail
# --------------------------------------------------------------------------- #


def test_rejects_non_3d_in_shape():
    spec = Qwen35GDNSpec()
    cfg = _config()
    try:
        spec.flops((128, 64), cfg)
    except ValueError:
        print("[ok] rejects 2D in_shape")
        return
    raise AssertionError("expected ValueError on 2D in_shape")


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #


def main() -> int:
    test_decode_flops_and_bytes()
    test_prefill_chunk_flops_and_bytes()
    test_mini_prefill_flops_unchanged_bytes_grow()
    test_mode_dispatch_at_chunk_size_boundary()
    test_chunk_mode_flops_linear_in_query_len()
    test_state_bytes_independent_of_cache_length()
    test_decode_is_memory_bound()
    test_param_count_balanced_config_near_6h_squared()
    test_registered_under_qwen3_5_gdn_name()
    test_rejects_non_3d_in_shape()
    print("\nAll Qwen35GDNSpec tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
