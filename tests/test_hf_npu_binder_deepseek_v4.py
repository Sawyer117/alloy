"""Integration test: ``alloy.integrations.hf_npu_binder`` bridge —
deepseek_v4 family wiring.

Covers the CSA (Compressed Sparse Attention) layer's dispatch surface:

  - ``"dsv4_csa.attention"`` has a registered ``"torch"`` impl (alloy's
    own ``_torch_csa_attention`` — the eager fallback that wraps
    ``_eager_attention_with_sinks``).
  - The binder's ``deepseek_v4.sparse_flash_attention.triton`` adapter
    is NOT registered (Phase 2: kernel port is in
    ``hf_npu_binder/deepseek_v4/sparse_flash_attention.py`` but the
    BHSD-to-SBHD wrapper + sliding-window combining is pending). A
    config request of ``_dsv4_csa_implementation = "triton"`` therefore
    falls back to ``"torch"`` via
    ``get_implementation(..., fallback="torch")`` instead of crashing
    with NotImplementedError on first forward.
  - ``activate(model, prefer=...)`` writes a ``_dsv4_csa_implementation``
    field on the config that resolves to a working callable for every
    intent in ``hf_npu_binder.DEFAULTS["deepseek_v4.sparse_flash_attention"]``
    (today: auto / flash / triton all map to ``"torch"``).

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


def test_dsv4_csa_triton_not_registered_yet() -> None:
    """The binder's ``sparse_flash_attention.triton`` (alloy adapter) is
    Phase 2 work and currently raises NotImplementedError if called. The
    bridge deliberately does not register it into alloy's IMPL_REGISTRY
    so that requests for ``triton`` quietly fall back to ``torch`` via
    ``get_implementation(..., fallback="torch")`` rather than crashing.

    Once Phase 2 lands (BHSD-to-SBHD wrapper + sliding + SFA combining),
    update this test to assert ``"triton"`` IS in the impl set and
    points to the binder symbol.
    """
    impls = list_implementations("dsv4_csa")
    csa_attn_impls = set(impls.get("dsv4_csa.attention", {}))
    assert "triton" not in csa_attn_impls, (
        f"dsv4_csa.attention:triton appears registered; binder adapter is "
        f"Phase 2 NotImplementedError. Update this test when the adapter ships."
    )
    # And ``get_implementation`` falls back cleanly:
    fn = get_implementation("dsv4_csa.attention", "triton", fallback="torch")
    assert fn is _torch_csa_attention, (
        "fallback chain should resolve to the registered torch impl"
    )


def test_activate_writes_dsv4_csa_field() -> None:
    """``activate(model, prefer=...)`` includes the DSV4 CSA field in its
    broadcast and writes a working impl name (``"torch"`` until the SFA
    adapter ships)."""
    for intent in ("auto", "flash", "triton"):
        model = _fake_model()
        chosen = bridge.activate(model, prefer=intent)
        assert "_dsv4_csa_implementation" in chosen, (
            f"activate(prefer={intent!r}) did not touch _dsv4_csa_implementation"
        )
        impl = chosen["_dsv4_csa_implementation"]
        # While the binder triton adapter is pending, every intent maps
        # to "torch" per DEFAULTS. Update this when the adapter ships.
        assert impl == "torch", (
            f"intent={intent!r} -> impl={impl!r}; expected 'torch' while "
            f"the SFA adapter is pending in binder"
        )


_TESTS = [
    test_dsv4_csa_attention_torch_registered,
    test_dsv4_csa_torch_is_alloy_native,
    test_dsv4_csa_triton_not_registered_yet,
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
