"""Integration test: ``alloy.integrations.hf_npu_binder`` bridge —
deepseek_v4 family wiring.

Covers the CSA (Compressed Sparse Attention) layer's dispatch surface:

  - ``"dsv4_csa.attention"`` has a registered ``"torch"`` impl (alloy's
    own ``_torch_csa_attention`` — the eager fallback that wraps
    ``_eager_attention_with_sinks``).
  - The bridge registers the binder's
    ``deepseek_v4.sparse_flash_attention.triton`` and ``.ascendc``
    adapters as the ``"triton"`` / ``"ascendc"`` impls under the same
    dispatch key.
  - ``activate(model, prefer=...)`` writes a ``_dsv4_csa_implementation``
    field on the config that resolves to a working impl name for every
    intent in ``hf_npu_binder.DEFAULTS["deepseek_v4.sparse_flash_attention"]``
    (auto / flash / triton -> ``"triton"``; ascendc -> ``"ascendc"``;
    torch -> ``"torch"``).

Cross-cutting bridge tests (DEFAULTS schema, intent broadcast,
binder-side hygiene) live in ``test_hf_npu_binder_bridge.py``.

If ``hf_npu_binder`` is not installed, the whole file no-ops (printed
SKIP).

Pure CPU torch.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

try:
    import hf_npu_binder  # noqa: F401  -- bridge dep; bail out below if absent
except ImportError:
    print("SKIP — hf_npu_binder not installed; deepseek_v4 wiring not exercised.")
    sys.exit(0)

from alloy.modules.attention.dsv4_attention import _torch_csa_attention
from alloy.modules.registry import get_implementation, list_implementations

# Importing the bridge has the side effect of registering binder backends.
import alloy.integrations.hf_npu_binder as bridge


def _fake_model():
    return type("FakeModel", (), {"config": type("FakeConfig", (), {})()})()


def test_dsv4_csa_attention_torch_registered() -> None:
    """alloy registers its own ``_torch_csa_attention`` under the
    ``"dsv4_csa.attention"`` dispatch key at module import time. The
    bridge does *not* override this — it leaves the torch impl alone."""
    impls = list_implementations("dsv4_csa")
    assert "dsv4_csa.attention" in impls, (
        f"missing 'dsv4_csa.attention' from alloy IMPL_REGISTRY; "
        f"got dsv4_csa keys: {list(impls)}"
    )
    assert "torch" in impls["dsv4_csa.attention"], (
        f"expected 'torch' impl on dsv4_csa.attention; "
        f"got: {sorted(impls['dsv4_csa.attention'])}"
    )


def test_dsv4_csa_torch_is_alloy_native() -> None:
    """The registered ``torch`` callable for CSA attention is alloy's own
    ``_torch_csa_attention``, NOT a binder symbol. This is the eager
    fallback that runs the scatter-bias mask + sink-softmax math in
    pure torch — byte-exact correct on its own."""
    fn = get_implementation("dsv4_csa.attention", "torch")
    assert fn is _torch_csa_attention, (
        f"dsv4_csa.attention:torch should be alloy's _torch_csa_attention, "
        f"got {fn!r}"
    )


def test_dsv4_csa_triton_registered() -> None:
    """The binder's ``sparse_flash_attention.triton`` adapter (BHSD-to-SBND
    permute + combined-topk construction over the vendored MindSpeed
    kernel) is registered under ``dsv4_csa.attention:triton``."""
    from hf_npu_binder.deepseek_v4 import sparse_flash_attention as _hf_sfa

    impls = list_implementations("dsv4_csa")
    csa_attn_impls = set(impls.get("dsv4_csa.attention", {}))
    assert "triton" in csa_attn_impls, (
        f"dsv4_csa.attention:triton not registered; "
        f"got: {sorted(csa_attn_impls)}"
    )
    fn = get_implementation("dsv4_csa.attention", "triton")
    assert fn is _hf_sfa.triton, (
        f"dsv4_csa.attention:triton should be hf_npu_binder's triton adapter, "
        f"got {fn!r}"
    )


def test_dsv4_csa_ascendc_registered() -> None:
    """The binder's ``sparse_flash_attention.ascendc`` adapter (wrap CANN's
    ``aclnnSparseAttnSharedkv`` op) is registered under
    ``dsv4_csa.attention:ascendc``. It needs a CANN release that ships
    the aclnn op in ``libopapi.so`` to actually run, but registration
    works on any environment — the import is cheap and the aclnn lookup
    is deferred to the first forward call."""
    from hf_npu_binder.deepseek_v4 import sparse_flash_attention as _hf_sfa

    fn = get_implementation("dsv4_csa.attention", "ascendc")
    assert fn is _hf_sfa.ascendc, (
        f"dsv4_csa.attention:ascendc should be hf_npu_binder's ascendc adapter, "
        f"got {fn!r}"
    )


def test_activate_writes_dsv4_csa_field() -> None:
    """``activate(model, prefer=...)`` includes the DSV4 CSA field in its
    broadcast and writes the impl that DEFAULTS recommends for each
    intent."""
    expected = {
        "auto":    "triton",
        "flash":   "triton",
        "triton":  "triton",
        "ascendc": "ascendc",
        "torch":   "torch",
    }
    for intent, want_impl in expected.items():
        model = _fake_model()
        chosen = bridge.activate(model, prefer=intent)
        assert "_dsv4_csa_implementation" in chosen, (
            f"activate(prefer={intent!r}) did not touch _dsv4_csa_implementation"
        )
        got = chosen["_dsv4_csa_implementation"]
        assert got == want_impl, (
            f"intent={intent!r} -> impl={got!r}; expected {want_impl!r} per "
            f"binder DEFAULTS"
        )


_TESTS = [
    test_dsv4_csa_attention_torch_registered,
    test_dsv4_csa_torch_is_alloy_native,
    test_dsv4_csa_triton_registered,
    test_dsv4_csa_ascendc_registered,
    test_activate_writes_dsv4_csa_field,
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
    print(f"\nAll {len(_TESTS)} deepseek_v4 wiring tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
