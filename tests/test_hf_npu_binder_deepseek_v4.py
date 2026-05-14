"""Integration test: ``alloy.integrations.hf_npu_binder`` bridge —
deepseek_v4 family wiring.

Covers all three DSV4 attention dispatch surfaces (registered through
alloy's ``IMPL_REGISTRY`` so accelerator kernels can swap in per-layer-type):

  - ``"dsv4_csa.attention"``     (Lightning-Indexer SFA + sliding)
  - ``"dsv4_hca.attention"``     (heavily-compressed KV + sliding, no indexer)
  - ``"dsv4_sliding.attention"`` (pure sliding window)

For each key, the torch ``"torch"`` impl is alloy's shared
``_torch_dsv4_attention`` (wraps ``_eager_attention_with_sinks``). The
binder bridge registers:

  - ``dsv4_csa.attention``     : ``triton`` (SFA + indexer adapter),
                                 ``ascendc`` (aclnnSparseAttnSharedkv)
  - ``dsv4_hca.attention``     : ``triton`` (compressed_attention adapter)
  - ``dsv4_sliding.attention`` : ``triton`` (compressed_attention adapter)

``activate(model, prefer=...)`` writes ``_dsv4_{csa,hca,sliding}_implementation``
fields on the config that resolve through ``hf_npu_binder.DEFAULTS`` for
the respective binder operator key.

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

from alloy.modules.attention.dsv4_attention import _torch_dsv4_attention
from alloy.modules.registry import get_implementation, list_implementations

# Importing the bridge has the side effect of registering binder backends.
import alloy.integrations.hf_npu_binder as bridge


def _fake_model():
    return type("FakeModel", (), {"config": type("FakeConfig", (), {})()})()


# ---------------------------------------------------------------------------
# torch impl: shared ``_torch_dsv4_attention`` registered under all 3 keys
# ---------------------------------------------------------------------------
def test_dsv4_torch_registered_for_all_three_layer_types() -> None:
    """alloy registers its own ``_torch_dsv4_attention`` under each of
    the three DSV4 attention dispatch keys at module import time. The
    bridge does *not* override these — it leaves the torch impls alone."""
    for kind in ("dsv4_csa", "dsv4_hca", "dsv4_sliding"):
        impls = list_implementations(kind)
        key = f"{kind}.attention"
        assert key in impls, (
            f"missing {key!r} from alloy IMPL_REGISTRY; got {kind} keys: {list(impls)}"
        )
        assert "torch" in impls[key], (
            f"expected 'torch' impl on {key}; got: {sorted(impls[key])}"
        )


def test_dsv4_torch_is_alloy_native_for_all_three() -> None:
    """The registered ``torch`` callable for every DSV4 attention key is
    alloy's own shared ``_torch_dsv4_attention``, NOT a binder symbol.
    This is the eager fallback that runs sink-softmax math in pure
    torch — byte-exact correct on its own. The same callable serves
    all three keys because the body (delegating to
    ``_eager_attention_with_sinks``) is layer-type-agnostic."""
    for key in ("dsv4_csa.attention", "dsv4_hca.attention", "dsv4_sliding.attention"):
        fn = get_implementation(key, "torch")
        assert fn is _torch_dsv4_attention, (
            f"{key}:torch should be alloy's _torch_dsv4_attention, got {fn!r}"
        )


# ---------------------------------------------------------------------------
# binder triton / ascendc registrations
# ---------------------------------------------------------------------------
def test_dsv4_csa_triton_registered() -> None:
    """The binder's ``sparse_flash_attention.triton`` adapter (BHSD-to-SBND
    permute + combined-topk over the vendored MindSpeed SFA kernel +
    Lightning-Indexer picks) is registered under
    ``dsv4_csa.attention:triton``."""
    from hf_npu_binder.deepseek_v4 import sparse_flash_attention as _hf_sfa

    impls = list_implementations("dsv4_csa")
    csa_attn_impls = set(impls.get("dsv4_csa.attention", {}))
    assert "triton" in csa_attn_impls, (
        f"dsv4_csa.attention:triton not registered; got: {sorted(csa_attn_impls)}"
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


def test_dsv4_hca_triton_registered() -> None:
    """The binder's ``compressed_attention.triton`` adapter (BHSD-to-SBND
    permute + sliding-window + see-all-compressed topk over the same
    vendored MindSpeed SFA kernel) is registered under
    ``dsv4_hca.attention:triton``. HCA differs from CSA in that there
    is no Lightning Indexer — the compressed-KV range is attended in
    full (matching alloy's torch reference, which pads
    ``attention_mask`` with 0 over the compressed columns)."""
    from hf_npu_binder.deepseek_v4 import compressed_attention as _hf_ca

    fn = get_implementation("dsv4_hca.attention", "triton")
    assert fn is _hf_ca.triton, (
        f"dsv4_hca.attention:triton should be hf_npu_binder's compressed_attention.triton, "
        f"got {fn!r}"
    )


def test_dsv4_sliding_triton_registered() -> None:
    """Sliding-only layers share the binder ``compressed_attention.triton``
    entry; the adapter recognises ``compressed_seq_len == 0`` and
    constructs only the sliding-window topk (no compressed range)."""
    from hf_npu_binder.deepseek_v4 import compressed_attention as _hf_ca

    fn = get_implementation("dsv4_sliding.attention", "triton")
    assert fn is _hf_ca.triton, (
        f"dsv4_sliding.attention:triton should be hf_npu_binder's compressed_attention.triton, "
        f"got {fn!r}"
    )


def test_dsv4_mhc_torch_registered() -> None:
    """alloy's ``_torch_hyper_connection`` is registered under
    ``dsv4_mhc.hyper_connection:torch`` at module-import time. The bridge
    doesn't override it — it leaves the torch impl alone and adds binder
    entries alongside."""
    impls = list_implementations("dsv4_mhc")
    key = "dsv4_mhc.hyper_connection"
    assert key in impls, f"missing {key!r} from IMPL_REGISTRY; got: {list(impls)}"
    assert "torch" in impls[key], (
        f"expected 'torch' impl on {key}; got: {sorted(impls[key])}"
    )
    fn = get_implementation(key, "torch")
    from alloy.modeling_alloy import _torch_hyper_connection
    assert fn is _torch_hyper_connection, (
        f"{key}:torch should be alloy's _torch_hyper_connection, got {fn!r}"
    )


def test_dsv4_mhc_triton_registered() -> None:
    """The binder's ``hyper_connection.triton`` adapter (composes
    rmsnorm_without_weight + sinkhorn + pre_bmm vendored kernels) is
    registered under ``dsv4_mhc.hyper_connection:triton``. The entry's
    hc_mult=4 guard is checked at first forward, not registration."""
    from hf_npu_binder.deepseek_v4 import hyper_connection as _hf_hc

    fn = get_implementation("dsv4_mhc.hyper_connection", "triton")
    assert fn is _hf_hc.triton, (
        f"dsv4_mhc.hyper_connection:triton should be hf_npu_binder's "
        f"hyper_connection.triton, got {fn!r}"
    )


# ---------------------------------------------------------------------------
# activate() broadcasts to all four DSV4 fields (3 attn + 1 mhc)
# ---------------------------------------------------------------------------
def test_activate_writes_all_dsv4_fields() -> None:
    """``activate(model, prefer=...)`` covers all three DSV4 dispatch
    surfaces. Each one resolves through its own DEFAULTS entry — CSA
    has an ``ascendc`` option (separate aclnn op), HCA / sliding share
    ``deepseek_v4.compressed_attention`` and only ship triton / torch.

    Note on ``auto``: as of binder 0.0.4, ``auto`` for both DSV4
    attention surfaces resolves to ``torch`` (not ``triton``) because
    measured triton-ascend speed on toy configs isn't a win over
    torch_npu eager and the wrapper adds bf16 drift. ``triton`` is the
    explicit-opt-in intent for environments that want the kernel path;
    ``ascendc`` is the genuine production fast path once CANN ships
    aclnnSparseAttnSharedkv. See binder DEFAULTS for the rationale.
    """
    # Each intent -> expected impl name per layer type
    expected_per_intent: dict[str, dict[str, str]] = {
        "auto": {
            # auto -> torch on all DSV4 surfaces (intentional; triton
            # isn't a measured speed win at toy config and adds bf16
            # drift; ascendc is opt-in only)
            "_dsv4_csa_implementation":     "torch",
            "_dsv4_hca_implementation":     "torch",
            "_dsv4_sliding_implementation": "torch",
            "_dsv4_mhc_implementation":     "torch",
        },
        "flash": {
            "_dsv4_csa_implementation":     "triton",   # no flash; triton is closest
            "_dsv4_hca_implementation":     "triton",
            "_dsv4_sliding_implementation": "triton",
            "_dsv4_mhc_implementation":     "triton",
        },
        "triton": {
            "_dsv4_csa_implementation":     "triton",
            "_dsv4_hca_implementation":     "triton",
            "_dsv4_sliding_implementation": "triton",
            "_dsv4_mhc_implementation":     "triton",
        },
        "ascendc": {
            "_dsv4_csa_implementation":     "ascendc",
            # HCA / sliding don't have a compressed_attention ascendc entry
            # yet (the aclnnSparseAttnSharedkv op fuses sliding+sparse+sink
            # in one kernel — HCA path through it is a future port). Fall
            # back to torch for now so activate() doesn't write a name
            # that resolves to nothing.
            "_dsv4_hca_implementation":     "torch",
            "_dsv4_sliding_implementation": "torch",
            # MHC has no ascendc port (would need aclnn op for the full
            # sinkhorn + bmm chain). Falls back to torch.
            "_dsv4_mhc_implementation":     "torch",
        },
        "torch": {
            "_dsv4_csa_implementation":     "torch",
            "_dsv4_hca_implementation":     "torch",
            "_dsv4_sliding_implementation": "torch",
            "_dsv4_mhc_implementation":     "torch",
        },
    }
    for intent, want_fields in expected_per_intent.items():
        model = _fake_model()
        chosen = bridge.activate(model, prefer=intent)
        for field, want_impl in want_fields.items():
            assert field in chosen, (
                f"activate(prefer={intent!r}) did not touch {field!r}"
            )
            got = chosen[field]
            assert got == want_impl, (
                f"intent={intent!r} field={field!r} -> impl={got!r}; "
                f"expected {want_impl!r} per binder DEFAULTS"
            )


_TESTS = [
    test_dsv4_torch_registered_for_all_three_layer_types,
    test_dsv4_torch_is_alloy_native_for_all_three,
    test_dsv4_csa_triton_registered,
    test_dsv4_csa_ascendc_registered,
    test_dsv4_hca_triton_registered,
    test_dsv4_sliding_triton_registered,
    test_dsv4_mhc_torch_registered,
    test_dsv4_mhc_triton_registered,
    test_activate_writes_all_dsv4_fields,
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
