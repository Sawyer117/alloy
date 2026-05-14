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
    CustomHardware,
    H100,
    PRESETS,
    Hardware,
    LinearSpec,
    RooflineSpec,
    SPEC_REGISTRY,
    dtype_size,
    format_comparison,
    get_hardware,
    get_spec,
    register_spec,
    roofline,
    roofline_decode,
    roofline_prefill,
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


def test_custom_hardware_factory():
    """CustomHardware(name=..., bf16=..., hbm_bandwidth=...) returns a Hardware
    that behaves identically to one constructed via Hardware(...) directly."""
    # Build same H100-like spec two ways and compare
    via_factory = CustomHardware(
        name="my-h100",
        hbm_bandwidth=3.35e12,
        bf16=989e12,
        fp16=989e12,
        fp32=67e12,
    )
    via_direct = Hardware(
        name="my-h100",
        peak_flops={
            torch.bfloat16: 989e12,
            torch.float16: 989e12,
            torch.float32: 67e12,
        },
        hbm_bandwidth=3.35e12,
    )
    assert isinstance(via_factory, Hardware)
    assert via_factory.name == via_direct.name
    assert via_factory.hbm_bandwidth == via_direct.hbm_bandwidth
    assert via_factory.peak_flops == via_direct.peak_flops
    assert via_factory.peak_vector_flops is None  # not provided
    print("[ok] CustomHardware factory matches direct Hardware construction")


def test_custom_hardware_with_vector_unit():
    """Specifying any vector_* kwarg populates peak_vector_flops."""
    hw = CustomHardware(
        name="my-ascend",
        hbm_bandwidth=2e12,
        bf16=400e12,
        vector_fp16=24e12,
        vector_fp32=12e12,
    )
    assert hw.peak_vector_flops is not None
    assert hw.get_peak_vector_flops(torch.float16) == 24e12
    assert hw.get_peak_vector_flops(torch.float32) == 12e12
    assert hw.get_peak_vector_flops(torch.bfloat16) is None  # not specified
    print("[ok] CustomHardware vector unit populates peak_vector_flops")


def test_custom_hardware_works_through_roofline():
    """End-to-end: build a custom hardware, pass to roofline(), verify the
    result uses our custom numbers."""
    custom = CustomHardware(
        name="dummy-fast-chip",
        hbm_bandwidth=10e12,    # 10 TB/s (very high)
        bf16=2000e12,           # 2 PFLOPS BF16 (very high)
    )
    config = SimpleNamespace(
        layer_types=[], ffn_types=[],
        hidden_size=128, vocab_size=1024, tie_word_embeddings=False,
    )
    report = roofline(config, batch=1, query_len=8, dtype=torch.bfloat16, hardware=custom)
    assert report.hardware is custom
    assert report.hardware.name == "dummy-fast-chip"
    # compute_time = total_flops / 2e15 (very fast)
    # memory_time = total_bytes / 1e13 (very fast too)
    # both should be sub-microsecond for this tiny model
    assert report.roofline_time_s < 1e-6
    print(f"[ok] CustomHardware end-to-end: roofline_time = {report.roofline_time_s*1e9:.1f} ns")


def test_custom_hardware_roundtrips_through_get_hardware():
    """get_hardware should pass-through Hardware instances unchanged."""
    custom = CustomHardware(name="x", hbm_bandwidth=1e12, bf16=100e12)
    assert get_hardware(custom) is custom
    print("[ok] get_hardware passes through CustomHardware")


def test_custom_hardware_fp8_when_available():
    """fp8 kwarg works when torch has float8_e4m3fn; raises a helpful error
    on older torch."""
    if hasattr(torch, "float8_e4m3fn"):
        hw = CustomHardware(name="h100-fp8", hbm_bandwidth=3.35e12, fp8=1979e12)
        assert hw.peak_flops[torch.float8_e4m3fn] == 1979e12
        print("[ok] CustomHardware fp8 stored under torch.float8_e4m3fn")
    else:
        try:
            CustomHardware(name="x", hbm_bandwidth=1e12, fp8=1000e12)
        except ValueError as e:
            assert "fp8" in str(e).lower()
            print(f"[ok] CustomHardware fp8 raises on old torch: {e}")
            return
        raise AssertionError("expected ValueError on fp8 with old torch")


def test_tokens_per_sec_and_per_module_bound():
    """Verify tokens_per_sec / time_per_token math + per-module bound returns
    'C' or 'M' (never 'unknown')."""
    mixer_name = "__bound_test_mixer__"
    SPEC_REGISTRY.pop(mixer_name, None)
    register_spec(mixer_name, LinearSpec(128, 128))
    try:
        config = SimpleNamespace(
            layer_types=[mixer_name],
            ffn_types=[mixer_name],
            hidden_size=128, vocab_size=1024, tie_word_embeddings=False,
        )
        report = roofline_prefill(config, batch=1, seq_len=64, hardware="A100")
        # tokens_per_sec * roofline_time = batch * query_len = 64
        assert abs(report.tokens_per_sec * report.roofline_time_s - 64) < 1e-6
        assert abs(report.time_per_token_s - report.roofline_time_s / 64) < 1e-12
        # Every module should produce a defined bound (C or M, never '?')
        bounds = [report._module_bound(m) for m in report.modules]
        assert all(b in ("C", "M") for b in bounds), f"unexpected bounds: {bounds}"
        # Per-module times should be non-negative
        times = [report._module_time_s(m) for m in report.modules]
        assert all(t >= 0 for t in times)
        print(f"[ok] tokens_per_sec={report.tokens_per_sec:.0f}  "
              f"per-module bounds={'/'.join(set(bounds))}")
    finally:
        SPEC_REGISTRY.pop(mixer_name, None)


def test_format_levels_run_with_markers():
    """Each format level + format_comparison should produce a non-empty string
    containing a level-specific marker. We don't pin exact layout."""
    mixer_name = "__fmt_test_mixer__"
    SPEC_REGISTRY.pop(mixer_name, None)
    register_spec(mixer_name, LinearSpec(128, 128))
    try:
        config = SimpleNamespace(
            layer_types=[mixer_name, mixer_name],
            ffn_types=[mixer_name, mixer_name],
            hidden_size=128, vocab_size=1024, tie_word_embeddings=False,
        )
        rep_h100 = roofline_prefill(config, batch=1, seq_len=64, hardware="H100")
        rep_a100 = roofline_prefill(config, batch=1, seq_len=64, hardware="A100")

        s1 = rep_h100.format(level=1)
        s2 = rep_h100.format(level=2)
        s3 = rep_h100.format(level=3)
        scmp = format_comparison([rep_h100, rep_a100], title="prefill seq=64")

        # Level 1: single-line summary, must mention tokens/sec
        assert "tok/s" in s1 and "\n" not in s1, "level=1 should be one line with tok/s"
        # Level 2: per-module table marker '%t' header column + bound legend
        assert "%t" in s2 and "compute-bound" in s2, "level=2 missing bound/percent markers"
        # Level 3: aggregated table — header has 'count' column + same bound legend
        assert "count" in s3 and "compute-bound" in s3, "level=3 missing aggregated marker"
        # format_comparison: title + multiple hardware names
        assert "prefill seq=64" in scmp and "H100-SXM5" in scmp and "A100-80GB-SXM" in scmp

        # invalid level raises
        try:
            rep_h100.format(level=99)
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError on invalid level")
        print("[ok] format(level=1/2/3) and format_comparison emit expected markers")
    finally:
        SPEC_REGISTRY.pop(mixer_name, None)


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
# MHC (Manifold-constrained Hyper-Connections) wiring
# --------------------------------------------------------------------------- #


def _mhc_test_config(*, use_mhc: bool, n_layers: int = 2):
    """Tiny config with the smallest registered mixer + ffn names. We don't
    care about the per-layer math here, only whether MHC adds the
    ``mhc:hc_attn`` / ``mhc:hc_ffn`` / ``mhc:hc_head`` accounting rows."""
    register_spec("_mhc_test_mixer", LinearSpec(in_features=64, out_features=64),
                  override=True)
    register_spec("_mhc_test_ffn", LinearSpec(in_features=64, out_features=64),
                  override=True)
    return SimpleNamespace(
        hidden_size=64,
        vocab_size=128,
        num_hidden_layers=n_layers,
        tie_word_embeddings=False,
        layer_types=["_mhc_test_mixer"] * n_layers,
        ffn_types=["_mhc_test_ffn"] * n_layers,
        use_mhc=use_mhc,
        hc_mult=4,
        hc_sinkhorn_iters=20,
    )


def test_mhc_off_no_residual_machinery() -> None:
    """With ``use_mhc=False`` the analyzer must not emit any mhc:* rows."""
    cfg = _mhc_test_config(use_mhc=False, n_layers=3)
    report = roofline(cfg, batch=1, query_len=8, kv_cache_len=0, hardware="H100")
    mhc_modules = [m for m in report.modules if m.kind == "mhc"]
    assert mhc_modules == [], (
        f"mhc rows should be absent when use_mhc=False; got {len(mhc_modules)}: "
        f"{[m.name for m in mhc_modules]}"
    )
    print("[ok] use_mhc=False emits zero mhc:* rows")


def test_mhc_on_emits_hc_per_layer_and_head_once() -> None:
    """With ``use_mhc=True`` and N layers, expect:
      * N ``mhc:hc_attn`` rows
      * N ``mhc:hc_ffn`` rows
      * exactly 1 ``mhc:hc_head`` row (post-loop, before final norm).
    """
    n = 5
    cfg = _mhc_test_config(use_mhc=True, n_layers=n)
    report = roofline(cfg, batch=1, query_len=8, kv_cache_len=0, hardware="H100")
    by_name = {}
    for m in (m for m in report.modules if m.kind == "mhc"):
        by_name.setdefault(m.name, []).append(m)
    assert len(by_name.get("hc_attn", [])) == n, f"expected {n} hc_attn rows, got {len(by_name.get('hc_attn', []))}"
    assert len(by_name.get("hc_ffn", [])) == n, f"expected {n} hc_ffn rows, got {len(by_name.get('hc_ffn', []))}"
    assert len(by_name.get("hc_head", [])) == 1, "exactly one hc_head row expected"
    # All MHC rows must have nonzero flops + bytes; zero would mean the spec misfired.
    for rows in by_name.values():
        for r in rows:
            assert r.flops > 0 and r.bytes > 0, f"mhc row {r.name} has zero flops/bytes"
    print(f"[ok] use_mhc=True emits {n} hc_attn + {n} hc_ffn + 1 hc_head rows with nonzero costs")


def test_mhc_on_strictly_more_flops_than_off() -> None:
    """Toggling MHC on must add non-trivial total FLOPs and bytes — sanity
    check that the spec values flow into the report totals."""
    cfg_off = _mhc_test_config(use_mhc=False, n_layers=3)
    cfg_on = _mhc_test_config(use_mhc=True, n_layers=3)
    r_off = roofline(cfg_off, batch=1, query_len=8, kv_cache_len=0, hardware="H100")
    r_on = roofline(cfg_on, batch=1, query_len=8, kv_cache_len=0, hardware="H100")
    assert r_on.total_flops > r_off.total_flops, "MHC on should add FLOPs"
    assert r_on.total_bytes > r_off.total_bytes, "MHC on should add bytes"
    print("[ok] enabling MHC strictly grows total_flops and total_bytes")


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
    test_custom_hardware_factory()
    test_custom_hardware_with_vector_unit()
    test_custom_hardware_works_through_roofline()
    test_custom_hardware_roundtrips_through_get_hardware()
    test_custom_hardware_fp8_when_available()
    test_tokens_per_sec_and_per_module_bound()
    test_format_levels_run_with_markers()
    test_roofline_end_to_end()
    test_roofline_strict_missing_raises()
    test_roofline_fail_open_marks_unknown()
    test_mhc_off_no_residual_machinery()
    test_mhc_on_emits_hc_per_layer_and_head_once()
    test_mhc_on_strictly_more_flops_than_off()
    print("\nAll roofline smoke tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
