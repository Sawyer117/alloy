"""Integration test: ``alloy.integrations.hf_npu_binder`` bridge — core
cross-cutting behaviour (no per-model knowledge).

Covers:
  - ``activate(model, prefer=<str>)`` resolves each operator's intent via
    ``hf_npu_binder.DEFAULTS`` (smart broadcast — not a naive same-string set).
  - ``activate(model, prefer="auto")`` consults DEFAULTS per operator and
    never broadcasts the literal ``"auto"`` string into IMPL_REGISTRY (which
    would never resolve).
  - ``activate(model, prefer=<dict>)`` takes literal per-module overrides,
    no DEFAULTS translation.
  - The binder package itself stays alloy-unaware (no
    ``register_implementation`` / ``IMPL_REGISTRY`` leak into ``hf_npu_binder``).

Per-model wiring tests live in sibling files
(``test_hf_npu_binder_qwen3_5_moe.py``, ``test_hf_npu_binder_deepseek_v4.py``).

If ``hf_npu_binder`` is not installed, the whole file no-ops (printed SKIP).

Pure CPU torch.
"""
from __future__ import annotations

try:
    import hf_npu_binder  # noqa: F401  -- bridge dep; skip if absent
except ImportError:
    import pytest

    pytest.skip("hf_npu_binder not installed", allow_module_level=True)

# Importing the bridge has the side effect of registering binder backends.
import alloy.integrations.hf_npu_binder as bridge


def _fake_model():
    """A model-like object with an empty config — enough for ``activate``,
    which only reads/writes ``_*_implementation`` attributes on it."""
    return type("FakeModel", (), {"config": type("FakeConfig", (), {})()})()


def test_activate_broadcast() -> None:
    """String-broadcast resolves each operator's intent via
    ``hf_npu_binder.DEFAULTS``, NOT a naive same-string set. Operators
    without a kernel for the requested intent are translated to a
    working fallback in their DEFAULTS entry.

    Today's DEFAULTS:
      - qwen3_5_moe.chunk_gated_delta_rule + experts both ship "flash",
        so an intent of ``"flash"`` lands literally.
      - deepseek_v4.{sparse_flash_attention, compressed_attention,
        hyper_connection} have no dedicated flash backend; their
        DEFAULTS route ``"flash"`` to ``"triton"`` (the vendored
        MindSpeed kernels via BHSD adapter / MHC sinkhorn-pre_bmm).
    """
    model = _fake_model()
    chosen = bridge.activate(model, prefer="flash")
    expected = {
        "_qwen3_5_gdn_implementation":    "flash",
        "_experts_implementation":        "flash",
        "_dsv4_csa_implementation":       "triton",  # SFA: flash -> triton
        "_dsv4_hca_implementation":       "triton",  # compressed_attn: flash -> triton
        "_dsv4_sliding_implementation":   "triton",  # compressed_attn: flash -> triton
        "_dsv4_mhc_implementation":       "triton",  # hyper_connection: flash -> triton
    }
    assert chosen == expected, chosen
    cfg = model.config
    for field, impl in expected.items():
        assert getattr(cfg, field) == impl, (field, getattr(cfg, field), impl)


def test_activate_auto_per_operator_recommendation() -> None:
    """``activate(model, "auto")`` consults each operator's recommended
    impl from ``hf_npu_binder.DEFAULTS`` — never blindly broadcasts
    "auto" as a literal string (which would never resolve in
    ``IMPL_REGISTRY``).

    Auto recommendations per DEFAULTS (evidence-based, see binder
    __init__ docstring): GDN auto=triton, experts auto=flash. DSV4
    attention surfaces auto=torch — triton-ascend isn't a measured
    speed win on toy configs and adds bf16 drift through stacked
    layers, so 'auto' opts out by default; users get triton via
    explicit ``activate(prefer='triton')`` or ``ascendc`` for the
    genuine CSA fast path once CANN ships ``aclnnSparseAttnSharedkv``.
    """
    model = _fake_model()
    chosen = bridge.activate(model, prefer="auto")
    # Each value must be an actual impl name registered in IMPL_REGISTRY,
    # not the literal "auto".
    assert "auto" not in chosen.values(), chosen
    assert chosen["_qwen3_5_gdn_implementation"] == "triton"
    assert chosen["_experts_implementation"] == "flash"
    assert chosen["_dsv4_csa_implementation"] == "torch"
    assert chosen["_dsv4_hca_implementation"] == "torch"
    assert chosen["_dsv4_sliding_implementation"] == "torch"
    assert chosen["_dsv4_mhc_implementation"] == "torch"


def test_activate_explicit_mapping_with_bare_module_key() -> None:
    """Mapping form takes literal values — no DEFAULTS translation —
    since the user named the impl explicitly. Bare module keys
    (``"qwen3_5_gdn"``) and fully-qualified field names
    (``"_qwen3_5_gdn_implementation"``) are both accepted."""
    model = _fake_model()
    chosen = bridge.activate(model, prefer={"qwen3_5_gdn": "triton"})
    assert chosen == {"_qwen3_5_gdn_implementation": "triton"}
    assert getattr(model.config, "_qwen3_5_gdn_implementation") == "triton"


def test_no_alloy_dep_inside_binder() -> None:
    """The binder import path must remain alloy-unaware — only the bridge
    knows alloy. Quick sanity check: the binder's top-level package
    namespace doesn't expose any 'register_implementation' or 'IMPL_REGISTRY'.
    """
    import hf_npu_binder as binder
    assert not hasattr(binder, "register_implementation"), (
        "binder should not surface alloy's register_implementation"
    )
    assert not hasattr(binder, "IMPL_REGISTRY"), (
        "binder should not surface alloy's IMPL_REGISTRY"
    )


_TESTS = [
    test_activate_broadcast,
    test_activate_auto_per_operator_recommendation,
    test_activate_explicit_mapping_with_bare_module_key,
    test_no_alloy_dep_inside_binder,
]


def main() -> int:
    failed = 0
    for fn in _TESTS:
        try:
            fn()
            print(f"  OK  {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL {fn.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  ERR  {fn.__name__}: {type(e).__name__}: {e}")
    if failed:
        print(f"\n{failed}/{len(_TESTS)} test(s) failed.")
        return 1
    print(f"\nAll {len(_TESTS)} bridge core tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
