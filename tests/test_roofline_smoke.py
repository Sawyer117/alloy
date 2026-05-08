"""Hello-world roofline test — verify spec API + end-to-end analyze() call.

Hand-computed numbers for a small known shape, checked against the spec /
analyzer output. Exercises:

  * :func:`dtype_size` for common dtypes
  * :class:`LinearSpec` flops / bytes / out_shape, with and without bias
  * dim-mismatch validation
  * spec registry: register / get / override / fail-open / strict
  * type check on register_spec
  * :class:`Hardware` presets present and well-formed
  * End-to-end :func:`roofline` on a tiny ``SimpleNamespace`` config with
    fake mixer/ffn names registered to :class:`LinearSpec` instances.

Run::

    python -m alloy.tests.test_roofline_smoke
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from alloy.roofline import (
    A100,
    ASCEND_910B,
    ASCEND_910B1,
    ASCEND_910C,
    H100,
    PRESETS,
    Hardware,
    LinearSpec,
    RooflineSpec,
    SPEC_REGISTRY,
    dtype_size,
    get_hardware,
    get_spec,
    register_spec,
    roofline,
)


# --------------------------------------------------------------------------- #
# dtype + LinearSpec unit tests
# --------------------------------------------------------------------------- #


def test_dtype_size():
    assert dtype_size(torch.float32) == 4
    assert dtype_size(torch.bfloat16) == 2
    assert dtype_size(torch.float16) == 2
    assert dtype_size(torch.float64) == 8
    print("[ok] dtype_size")


def test_linear_spec_flops_no_bias():
    """Linear(128 -> 256) on input [B=2, T=32, 128]:
        n_tokens = 64;  flops = 2 * 64 * 128 * 256 = 4,194,304."""
    spec = LinearSpec(128, 256, bias=False)
    flops = spec.flops((2, 32, 128))
    expected = 2 * 64 * 128 * 256
    assert flops == expected, f"{flops} != {expected}"
    print(f"[ok] LinearSpec flops (no bias): {flops:,}")


def test_linear_spec_flops_with_bias():
    spec = LinearSpec(128, 256, bias=True)
    flops = spec.flops((2, 32, 128))
    expected = 2 * 64 * 128 * 256 + 64 * 256
    assert flops == expected
    print(f"[ok] LinearSpec flops (bias):    {flops:,}")


def test_linear_spec_bytes_bf16():
    """bf16 (2 B/elt) for Linear(128 -> 256), input [2, 32, 128]:
        weights  = 128*256*2 = 65,536
        act_in   =  64*128*2 = 16,384
        act_out  =  64*256*2 = 32,768
        total    =            114,688  (no bias)."""
    spec = LinearSpec(128, 256)
    b = spec.bytes((2, 32, 128), dtype=torch.bfloat16)
    expected = 128 * 256 * 2 + 64 * 128 * 2 + 64 * 256 * 2
    assert b == expected, f"{b} != {expected}"
    print(f"[ok] LinearSpec bytes (bf16):    {b:,}")


def test_linear_spec_bytes_fp32_doubles_bf16():
    spec = LinearSpec(128, 256)
    b_bf16 = spec.bytes((2, 32, 128), dtype=torch.bfloat16)
    b_fp32 = spec.bytes((2, 32, 128), dtype=torch.float32)
    assert b_fp32 == 2 * b_bf16
    print(f"[ok] LinearSpec bytes scale: bf16={b_bf16:,} fp32={b_fp32:,}")


def test_linear_spec_out_shape():
    spec = LinearSpec(128, 256)
    assert spec.out_shape((2, 32, 128)) == (2, 32, 256)
    print("[ok] LinearSpec out_shape")


def test_linear_spec_dim_mismatch_raises():
    spec = LinearSpec(128, 256)
    try:
        spec.flops((2, 32, 64))
    except ValueError:
        print("[ok] LinearSpec dim-mismatch raises ValueError")
        return
    raise AssertionError("expected ValueError on dim mismatch")


# --------------------------------------------------------------------------- #
# Registry tests
# --------------------------------------------------------------------------- #


def test_register_and_get_spec():
    name = "__test_dummy__"
    SPEC_REGISTRY.pop(name, None)

    spec = LinearSpec(64, 128)
    register_spec(name, spec)
    assert get_spec(name) is spec

    try:
        register_spec(name, LinearSpec(64, 128))
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError on duplicate without override")

    spec2 = LinearSpec(64, 128)
    register_spec(name, spec2, override=True)
    assert get_spec(name) is spec2

    SPEC_REGISTRY.pop(name)
    print("[ok] register_spec / get_spec / override")


def test_get_spec_missing_warns():
    name = "__definitely_not_registered__"
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = get_spec(name, strict=False)
        assert result is None
        assert any("No RooflineSpec for" in str(x.message) for x in w)
    print("[ok] get_spec(missing, strict=False) warns + returns None")


def test_get_spec_missing_strict_raises():
    name = "__definitely_not_registered_2__"
    try:
        get_spec(name, strict=True)
    except KeyError as e:
        assert name in str(e)
        print("[ok] get_spec(missing, strict=True) raises KeyError")
        return
    raise AssertionError("expected KeyError")


def test_register_spec_type_check():
    try:
        register_spec("__bad__", "not a spec")  # type: ignore[arg-type]
    except TypeError:
        print("[ok] register_spec rejects non-RooflineSpec")
        return
    raise AssertionError("expected TypeError")


# --------------------------------------------------------------------------- #
# Hardware presets sanity
# --------------------------------------------------------------------------- #


def test_hardware_presets():
    for name, hw in PRESETS.items():
        assert isinstance(hw, Hardware)
        assert hw.hbm_bandwidth > 0
        assert torch.bfloat16 in hw.peak_flops
        assert hw.peak_flops[torch.bfloat16] > 0
    assert get_hardware("A100") is A100
    assert get_hardware("H100") is H100
    assert get_hardware("Ascend910B") is ASCEND_910B
    assert get_hardware("Ascend910B1") is ASCEND_910B1
    assert get_hardware("Ascend910C") is ASCEND_910C
    # Backward-compat: "Ascend910B" alias points at the 910B1 instance
    assert ASCEND_910B is ASCEND_910B1
    # 910C compute is exactly 2x 910B1 across all dtypes (sanity check)
    for dt, v in ASCEND_910B1.peak_flops.items():
        assert ASCEND_910C.peak_flops[dt] == 2 * v, (
            f"910C peak {dt} should be 2x 910B1: {ASCEND_910C.peak_flops[dt]} vs {2*v}"
        )
    # 910B1 vector throughput is set; NVIDIA presets leave it None
    assert ASCEND_910B1.peak_vector_flops is not None
    assert ASCEND_910B1.get_peak_vector_flops(torch.float16) == 24 * 1e12
    assert A100.peak_vector_flops is None
    assert A100.get_peak_vector_flops(torch.float16) is None
    # Pass-through Hardware instance
    custom = Hardware(name="custom", peak_flops={torch.bfloat16: 1e12}, hbm_bandwidth=1e9)
    assert get_hardware(custom) is custom
    try:
        get_hardware("Nonexistent-9000")
    except KeyError:
        pass
    else:
        raise AssertionError("expected KeyError on unknown preset")
    print("[ok] hardware presets")


# --------------------------------------------------------------------------- #
# End-to-end: roofline() on a tiny SimpleNamespace config
# --------------------------------------------------------------------------- #


def test_roofline_end_to_end():
    """Tiny synthetic 'model': 2 layers, each layer is 1 mixer-as-Linear + 1
    ffn-as-Linear, all hidden=128. Verifies aggregation and the embedding /
    lm_head top-level contributions.

    Hand-compute (bf16, 2 B/elt; B=1, T=64, H=128, V=1024):
      embed:    0 flops; 2 * 1*64*128 * 2                                = 32,768 B
      Linear(H,H) per call:
          flops = 2*64*128*128         = 2,097,152
          bytes = 128*128*2 + 64*128*2 + 64*128*2 = 32,768+16,384+16,384 = 65,536 B
      4 layer-Linears: flops = 8,388,608; bytes = 262,144 B
      final RMSNorm on [1, 64, 128]:
          flops = 4 * 8192                                               =     32,768
          bytes = 128*2 + 2*8192*2                                       =     33,024 B
      lm_head (H -> V):
          flops = 2*64*128*1024                                          = 16,777,216
          bytes = (128*1024 + 64*128 + 64*1024) * 2                      = 409,600 B
      TOTAL flops = 25,198,592;  TOTAL bytes = 737,536 B
    """
    mixer_name = "__hello_mixer__"
    ffn_name = "__hello_ffn__"
    SPEC_REGISTRY.pop(mixer_name, None)
    SPEC_REGISTRY.pop(ffn_name, None)
    register_spec(mixer_name, LinearSpec(128, 128))
    register_spec(ffn_name, LinearSpec(128, 128))

    try:
        config = SimpleNamespace(
            layer_types=[mixer_name, mixer_name],
            ffn_types=[ffn_name, ffn_name],
            hidden_size=128,
            vocab_size=1024,
            tie_word_embeddings=False,
        )
        report = roofline(config, batch=1, query_len=64, dtype=torch.bfloat16, hardware="A100")

        # Per-module hand-computed expectations
        per_linear_flops = 2 * 64 * 128 * 128
        per_linear_bytes = 128 * 128 * 2 + 64 * 128 * 2 + 64 * 128 * 2
        embed_bytes = 2 * 64 * 128 * 2
        norm_flops = 4 * 1 * 64 * 128
        norm_bytes = 128 * 2 + 2 * 1 * 64 * 128 * 2
        lm_head_flops = 2 * 64 * 128 * 1024
        lm_head_bytes = (128 * 1024 + 64 * 128 + 64 * 1024) * 2

        expected_flops = 4 * per_linear_flops + norm_flops + lm_head_flops
        expected_bytes = embed_bytes + 4 * per_linear_bytes + norm_bytes + lm_head_bytes

        assert report.total_flops == expected_flops, (
            f"flops {report.total_flops} != {expected_flops}"
        )
        assert report.total_bytes == expected_bytes, (
            f"bytes {report.total_bytes} != {expected_bytes}"
        )

        # Module count: 1 embed + 2*(mixer+ffn) + 1 norm + 1 lm_head = 7
        assert len(report.modules) == 7, f"module count {len(report.modules)} != 7"

        # Bottleneck: tiny model is heavily memory-bound on A100
        assert report.bottleneck == "memory"
        assert report.compute_time_s > 0
        assert report.memory_time_s > 0

        print(f"[ok] roofline end-to-end: "
              f"flops={report.total_flops:,} bytes={report.total_bytes:,} "
              f"AI={report.arithmetic_intensity:.1f} bottleneck={report.bottleneck}")
        print()
        print(report)
        print()
    finally:
        SPEC_REGISTRY.pop(mixer_name, None)
        SPEC_REGISTRY.pop(ffn_name, None)


def test_roofline_strict_missing_raises():
    config = SimpleNamespace(
        layer_types=["__never_registered_mixer__"],
        ffn_types=["__never_registered_ffn__"],
        hidden_size=128,
        vocab_size=1024,
        tie_word_embeddings=False,
    )
    try:
        roofline(config, batch=1, query_len=8, strict=True)
    except KeyError:
        print("[ok] roofline(strict=True) raises on missing spec")
        return
    raise AssertionError("expected KeyError under strict=True")


def test_roofline_fail_open_marks_unknown():
    """Default strict=False: missing specs -> 'unknown' kind, 0 contribution."""
    config = SimpleNamespace(
        layer_types=["__never_registered_mixer__"],
        ffn_types=["__never_registered_ffn__"],
        hidden_size=128,
        vocab_size=1024,
        tie_word_embeddings=False,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # fail-open emits warnings
        report = roofline(config, batch=1, query_len=8, strict=False)
    unknown = [m for m in report.modules if m.kind == "unknown"]
    assert len(unknown) == 2, f"expected 2 unknown stats, got {len(unknown)}"
    assert all(m.flops == 0 and m.bytes == 0 for m in unknown)
    # Embedding + lm_head still contribute
    assert report.total_flops > 0
    print("[ok] roofline(strict=False) marks missing specs as 'unknown'")


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #


def main() -> int:
    test_dtype_size()
    test_linear_spec_flops_no_bias()
    test_linear_spec_flops_with_bias()
    test_linear_spec_bytes_bf16()
    test_linear_spec_bytes_fp32_doubles_bf16()
    test_linear_spec_out_shape()
    test_linear_spec_dim_mismatch_raises()
    test_register_and_get_spec()
    test_get_spec_missing_warns()
    test_get_spec_missing_strict_raises()
    test_register_spec_type_check()
    test_hardware_presets()
    test_roofline_end_to_end()
    test_roofline_strict_missing_raises()
    test_roofline_fail_open_marks_unknown()
    print("\nAll roofline smoke tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
