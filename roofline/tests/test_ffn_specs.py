"""Hand-computed FFN spec tests.

Each test computes expected FLOPs / bytes by hand from the spec's documented
formula, then asserts the spec output matches byte-exactly. Covers:

  * :class:`SwiGLUMLPSpec` for ``qwen3_mlp`` and bias-on (DSV4 shared expert
    flavor) and shared_expert_intermediate (Qwen3.5 shared expert flavor)
  * :class:`DSV4MoESpec` for ``dsv4_moe`` (topk router) and ``dsv4_hash_moe``
    (hash router with ``tid2eid`` lookup)
  * :class:`Qwen35MoESpec` for ``qwen3_5_moe`` (with gated shared expert)
  * Registry: all four names registered after ``import alloy.roofline``.

Run::

    python -m alloy.tests.test_roofline_ffn_specs
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from alloy.roofline import SPEC_REGISTRY, get_spec
from alloy.roofline.specs_ffn import (
    DSV4MoESpec,
    Qwen35MoESpec,
    SwiGLUMLPSpec,
)


# --------------------------------------------------------------------------- #
# SwiGLUMLPSpec — qwen3_mlp / DSV4 shared / Qwen3.5 shared variants
# --------------------------------------------------------------------------- #


def test_swiglu_qwen3_mlp_no_bias():
    """qwen3_mlp: hidden=128, intermediate=256, bias=False, on [2, 32, 128]:

    n_tokens = 64
    flops = 6*64*128*256 + 5*64*256 = 12,582,912 + 81,920 = 12,664,832
    bytes(bf16):
      weights = 3*128*256*2 = 196,608
      act_in  = 64*128*2    =  16,384
      act_out = 64*128*2    =  16,384
      total   =              229,376
    """
    spec = SwiGLUMLPSpec()
    config = SimpleNamespace(intermediate_size=256, mlp_bias=False)

    flops = spec.flops((2, 32, 128), config)
    expected_flops = 6 * 64 * 128 * 256 + 5 * 64 * 256
    assert flops == expected_flops, f"flops {flops} != {expected_flops}"

    b = spec.bytes((2, 32, 128), config, dtype=torch.bfloat16)
    expected_b = 3 * 128 * 256 * 2 + 64 * 128 * 2 + 64 * 128 * 2
    assert b == expected_b, f"bytes {b} != {expected_b}"

    print(f"[ok] SwiGLU qwen3_mlp (no bias): flops={flops:,} bytes={b:,}")


def test_swiglu_with_bias():
    """DSV4 shared expert flavor — same as above but with mlp_bias=True:

    flops += 2*64*256 + 64*128 = 32,768 + 8,192 = 40,960
    bytes += (2*256 + 128) * 2 = 1,280
    """
    spec = SwiGLUMLPSpec()
    config = SimpleNamespace(intermediate_size=256, mlp_bias=True)

    flops = spec.flops((2, 32, 128), config)
    expected_flops = 6 * 64 * 128 * 256 + 5 * 64 * 256 + (2 * 64 * 256 + 64 * 128)
    assert flops == expected_flops

    b = spec.bytes((2, 32, 128), config, dtype=torch.bfloat16)
    expected_b = 3 * 128 * 256 * 2 + (2 * 256 + 128) * 2 + 64 * 128 * 2 + 64 * 128 * 2
    assert b == expected_b

    print(f"[ok] SwiGLU bias=True: flops={flops:,} bytes={b:,}")


def test_swiglu_alt_intermediate_attr():
    """Qwen3.5 shared expert flavor — reads shared_expert_intermediate_size."""
    spec = SwiGLUMLPSpec(intermediate_attr="shared_expert_intermediate_size")
    config = SimpleNamespace(shared_expert_intermediate_size=512)

    # Should read 512, not error on missing intermediate_size
    flops = spec.flops((2, 32, 128), config)
    expected_flops = 6 * 64 * 128 * 512 + 5 * 64 * 512
    assert flops == expected_flops
    print(f"[ok] SwiGLU shared_expert_intermediate_size: flops={flops:,}")


def test_swiglu_bias_default_false_when_attr_missing():
    """If config doesn't have mlp_bias attribute, SwiGLUMLPSpec defaults to bias=False
    (uses getattr default). Verifies qwen3_mlp's hardcoded-no-bias case."""
    spec = SwiGLUMLPSpec()
    config = SimpleNamespace(intermediate_size=256)  # no mlp_bias
    flops = spec.flops((2, 32, 128), config)
    # Same as the no-bias case
    expected = 6 * 64 * 128 * 256 + 5 * 64 * 256
    assert flops == expected
    print("[ok] SwiGLU defaults bias=False on missing attr")


# --------------------------------------------------------------------------- #
# DSV4MoESpec — topk and hash variants
# --------------------------------------------------------------------------- #


def _dsv4_moe_config():
    """Small DSV4 MoE config: H=128, I=64, E=8, K=2, no mlp_bias."""
    return SimpleNamespace(
        intermediate_size=64,
        n_routed_experts=8,
        num_experts_per_tok=2,
        mlp_bias=False,
    )


def test_dsv4_moe_topk():
    """DSV4 MoE topk on input [1, 16, 128] (n_tokens=16):

    flops:
      router_matmul = 2 * 16 * 128 * 8                     = 32,768
      router_score  = 5 * 16 * 8                           =    640
      router_norm   = 2 * 16 * 2                           =     64
      per_expert    = 6*1*128*64 + 5*1*64                  = 49,472
      routed        = 16 * 2 * 49,472                      = 1,583,104
      shared        = 6*16*128*64 + 5*16*64                = 791,552
      add           = 16 * 128                             = 2,048
      total                                                = 2,410,176

    bytes (bf16):
      router_w (matmul) = 8 * 128 * 2                      = 2,048
      e_bias            = 8 * 2                            =    16
      unique_experts    = min(8, 16*2)                     = 8
      per_expert_w      = 3 * 128 * 64 * 2                 = 49,152
      routed_w          = 8 * 49,152                       = 393,216
      shared_w          =                                  = 49,152
      act               = 2 * 16 * 128 * 2                 = 8,192
      total                                                = 452,624
    """
    spec = DSV4MoESpec(is_hash=False)
    config = _dsv4_moe_config()

    flops = spec.flops((1, 16, 128), config)
    expected_flops = (
        2 * 16 * 128 * 8       # router matmul
        + 5 * 16 * 8           # score
        + 2 * 16 * 2           # norm
        + 16 * 2 * (6 * 128 * 64 + 5 * 64)   # routed
        + (6 * 16 * 128 * 64 + 5 * 16 * 64)  # shared
        + 16 * 128             # add
    )
    assert flops == expected_flops, f"flops {flops} != {expected_flops}"

    b = spec.bytes((1, 16, 128), config, dtype=torch.bfloat16)
    expected_b = (
        8 * 128 * 2            # router weight
        + 8 * 2                # e_score_correction_bias (topk)
        + 8 * (3 * 128 * 64 * 2)  # routed expert weights
        + 3 * 128 * 64 * 2     # shared expert weights
        + 16 * 128 * 2 * 2     # act_in + act_out
    )
    assert b == expected_b, f"bytes {b} != {expected_b}"

    print(f"[ok] dsv4_moe (topk): flops={flops:,} bytes={b:,}")


def test_dsv4_moe_hash():
    """DSV4 hash variant: router buffer is tid2eid (int64), not e_score bias.

    bytes diff vs topk:
      add tid2eid:    16 * 2 * 8                           = 256
      remove e_bias:  -8 * 2                               = -16
      net diff: +240 bytes
    flops: identical to topk variant.
    """
    spec_topk = DSV4MoESpec(is_hash=False)
    spec_hash = DSV4MoESpec(is_hash=True)
    config = _dsv4_moe_config()

    flops_topk = spec_topk.flops((1, 16, 128), config)
    flops_hash = spec_hash.flops((1, 16, 128), config)
    assert flops_topk == flops_hash, "hash and topk flops should match"

    b_topk = spec_topk.bytes((1, 16, 128), config, dtype=torch.bfloat16)
    b_hash = spec_hash.bytes((1, 16, 128), config, dtype=torch.bfloat16)
    expected_diff = 16 * 2 * 8 - 8 * 2  # +tid2eid -e_bias
    assert b_hash - b_topk == expected_diff, (
        f"diff {b_hash - b_topk} != {expected_diff}"
    )

    print(f"[ok] dsv4_hash_moe: bytes={b_hash:,} (diff vs topk = {b_hash-b_topk}: tid2eid)")


def test_dsv4_moe_decode_regime():
    """N=1 (decode): unique_experts = min(E, top_k) = top_k, not E.

    With H=128, I=64, E=8, K=2, n_tokens=1:
      unique_experts = min(8, 1*2) = 2 (not 8)
      routed_w = 2 * (3*128*64*2) = 49,152
    """
    spec = DSV4MoESpec(is_hash=False)
    config = _dsv4_moe_config()
    b = spec.bytes((1, 1, 128), config, dtype=torch.bfloat16)
    expected_routed_w = 2 * (3 * 128 * 64 * 2)  # only 2 experts loaded
    expected = (
        8 * 128 * 2            # router weight
        + 8 * 2                # e_bias
        + expected_routed_w
        + 3 * 128 * 64 * 2     # shared expert
        + 1 * 128 * 2 * 2      # act_in + act_out
    )
    assert b == expected, f"bytes {b} != {expected}"
    print(f"[ok] dsv4_moe decode regime (N=1): only top_k experts loaded, bytes={b:,}")


# --------------------------------------------------------------------------- #
# Qwen35MoESpec
# --------------------------------------------------------------------------- #


def test_qwen35_moe():
    """Qwen3.5 MoE on input [1, 16, 128]:

    Config: H=128, moe_inter=64, shared_inter=128, E=8, K=2.

    flops:
      router_matmul = 2 * 16 * 128 * 8                     =    32,768
      router_softmax= 5 * 16 * 8                           =       640
      router_norm   = 2 * 16 * 2                           =        64
      per_expert    = 6*1*128*64 + 5*1*64                  =    49,472
      routed        = 16 * 2 * 49,472                      = 1,583,104
      shared        = 6*16*128*128 + 5*16*128              = 1,583,104
      shared_gate   = 2*16*128 + 3*16 + 16*128             =     6,192
      add           = 16 * 128                             =     2,048
      total                                                = 3,207,920

    bytes (bf16):
      router_w     = 8 * 128 * 2                           =   2,048
      routed_w     = 8 * (3*128*64*2)                      = 393,216
      shared_w     = 3 * 128 * 128 * 2                     =  98,304
      gate_w       = 128 * 2                               =     256
      act          = 2 * 16 * 128 * 2                      =   8,192
      total                                                = 502,016
    """
    spec = Qwen35MoESpec()
    config = SimpleNamespace(
        moe_intermediate_size=64,
        shared_expert_intermediate_size=128,
        num_experts=8,
        num_experts_per_tok=2,
    )

    flops = spec.flops((1, 16, 128), config)
    expected_flops = (
        2 * 16 * 128 * 8                         # router matmul
        + 5 * 16 * 8                             # softmax
        + 2 * 16 * 2                             # norm
        + 16 * 2 * (6 * 128 * 64 + 5 * 64)       # routed
        + (6 * 16 * 128 * 128 + 5 * 16 * 128)    # shared (using shared_inter)
        + (2 * 16 * 128 + 3 * 16 + 16 * 128)     # shared gate
        + 16 * 128                               # final add
    )
    assert flops == expected_flops, f"flops {flops} != {expected_flops}"

    b = spec.bytes((1, 16, 128), config, dtype=torch.bfloat16)
    expected_b = (
        8 * 128 * 2                # router weight
        + 8 * (3 * 128 * 64 * 2)   # routed expert weights (moe_inter)
        + 3 * 128 * 128 * 2        # shared expert weights (shared_inter)
        + 128 * 2                  # shared expert gate weight [H]
        + 16 * 128 * 2 * 2         # act_in + act_out
    )
    assert b == expected_b, f"bytes {b} != {expected_b}"

    print(f"[ok] qwen3_5_moe: flops={flops:,} bytes={b:,}")


# --------------------------------------------------------------------------- #
# Registry verification
# --------------------------------------------------------------------------- #


def test_all_ffn_names_registered():
    """All four FFN names should be registered after ``import alloy.roofline``."""
    expected_names = {"qwen3_mlp", "dsv4_moe", "dsv4_hash_moe", "qwen3_5_moe"}
    missing = expected_names - set(SPEC_REGISTRY)
    assert not missing, f"missing FFN spec registrations: {missing}"

    # Spot check types
    assert isinstance(get_spec("qwen3_mlp"), SwiGLUMLPSpec)
    assert isinstance(get_spec("dsv4_moe"), DSV4MoESpec)
    assert get_spec("dsv4_moe").is_hash is False
    assert isinstance(get_spec("dsv4_hash_moe"), DSV4MoESpec)
    assert get_spec("dsv4_hash_moe").is_hash is True
    assert isinstance(get_spec("qwen3_5_moe"), Qwen35MoESpec)

    print(f"[ok] all FFN names registered: {sorted(expected_names)}")


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #


def main() -> int:
    test_swiglu_qwen3_mlp_no_bias()
    test_swiglu_with_bias()
    test_swiglu_alt_intermediate_attr()
    test_swiglu_bias_default_false_when_attr_missing()
    test_dsv4_moe_topk()
    test_dsv4_moe_hash()
    test_dsv4_moe_decode_regime()
    test_qwen35_moe()
    test_all_ffn_names_registered()
    print("\nAll FFN spec tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
