"""Integration test: ``alloy.integrations.hf_npu_binder`` bridge —
qwen3_5_moe family wiring.

Covers:
  - The bridge import registers ``"triton"`` / ``"flash"`` impls under
    every ``qwen3_5_gdn.*`` alloy key (chunk_rule / recurrent_rule /
    causal_conv1d).
  - The registered callables are byte-identical to the binder's symbols
    (no copies / no wrappers) — so identity tests downstream are
    meaningful.
  - ``experts.flash`` is registered into HF's ``ALL_EXPERTS_FUNCTIONS``
    table (the MoE experts forward is a whole-block dispatch, not split
    into per-op alloy sub-functions; it doesn't go through alloy's
    IMPL_REGISTRY).
  - A subsequently-constructed ``Qwen35GatedDeltaNet`` routes its
    sub-functions to the binder callables after ``activate``.

Cross-cutting bridge tests (DEFAULTS broadcast, intent translation,
binder-side hygiene) live in ``test_hf_npu_binder_bridge.py``.

If ``hf_npu_binder`` is not installed, the whole file no-ops (printed
SKIP).

Pure CPU torch.
"""
from __future__ import annotations

try:
    import hf_npu_binder  # noqa: F401  -- bridge dep; skip if absent
except ImportError:
    import pytest

    pytest.skip("hf_npu_binder not installed", allow_module_level=True)

from alloy import AlloyConfig
from alloy.modules.attention.qwen3_5_gdn import Qwen35GatedDeltaNet
from alloy.modules.registry import get_implementation, list_implementations
from transformers.integrations.moe import ALL_EXPERTS_FUNCTIONS

# Importing the bridge has the side effect of registering binder backends.
import alloy.integrations.hf_npu_binder as bridge  # noqa: F401
from hf_npu_binder.qwen3_5_moe import (
    causal_conv1d as _hf_causal_conv1d,
    chunk_gated_delta_rule as _hf_chunk_gdr,
    experts as _hf_experts,
    fused_recurrent_gated_delta_rule as _hf_recurrent_gdr,
)


def _gdn_config(**override) -> AlloyConfig:
    """A minimal AlloyConfig sufficient to construct ``Qwen35GatedDeltaNet``.
    GDN reads its sub-function impl handles from ``config._qwen3_5_gdn_implementation``
    at ``__init__``, so we need a real AlloyConfig instance with those
    attribute setters working."""
    cfg = AlloyConfig(
        vocab_size=128,
        hidden_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        intermediate_size=128,
        max_position_embeddings=64,
        layer_types=["qwen3_5_gdn"],
        ffn_types=["qwen3_mlp"],
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_conv_kernel_dim=4,
    )
    for k, v in override.items():
        setattr(cfg, k, v)
    return cfg


def test_bridge_registers_triton_and_flash() -> None:
    impls = list_implementations("qwen3_5_gdn")
    expected_keys = {
        "qwen3_5_gdn.chunk_rule",
        "qwen3_5_gdn.recurrent_rule",
        "qwen3_5_gdn.causal_conv1d",
    }
    assert expected_keys <= set(impls), f"missing alloy keys: {expected_keys - set(impls)}"
    for k in expected_keys:
        names = set(impls[k])
        assert {"torch", "triton", "flash"} <= names, (
            f"{k}: expected torch/triton/flash, got {names}"
        )


def test_bridge_callables_are_binder_originals() -> None:
    """The registered callables must be exactly the binder's symbols — not
    copies, not wrappers — so identity tests downstream are meaningful.
    """
    assert get_implementation("qwen3_5_gdn.chunk_rule", "triton") is _hf_chunk_gdr.triton
    assert get_implementation("qwen3_5_gdn.chunk_rule", "flash")  is _hf_chunk_gdr.flash
    assert get_implementation("qwen3_5_gdn.recurrent_rule", "triton") is _hf_recurrent_gdr.triton
    assert get_implementation("qwen3_5_gdn.recurrent_rule", "flash")  is _hf_recurrent_gdr.flash
    assert get_implementation("qwen3_5_gdn.causal_conv1d", "triton") is _hf_causal_conv1d.triton
    assert get_implementation("qwen3_5_gdn.causal_conv1d", "flash")  is _hf_causal_conv1d.flash


def test_bridge_registers_experts_into_hf_table() -> None:
    """The whole MoE experts forward (permute + gmm + swiglu + gmm + unpermute)
    is one HF dispatch entry — registered into ``ALL_EXPERTS_FUNCTIONS``,
    not into alloy's IMPL_REGISTRY.
    """
    assert "flash" in ALL_EXPERTS_FUNCTIONS, (
        f"expected 'flash' in HF ALL_EXPERTS_FUNCTIONS after bridge import; "
        f"got {sorted(ALL_EXPERTS_FUNCTIONS)}"
    )
    assert ALL_EXPERTS_FUNCTIONS["flash"] is _hf_experts.flash, (
        "HF table 'flash' entry is not the binder callable — wiring drift"
    )


def test_constructed_layer_routes_to_binder() -> None:
    cfg = _gdn_config()
    model = type("FakeModel", (), {"config": cfg})()
    bridge.activate(model, prefer="flash")

    layer = Qwen35GatedDeltaNet(cfg, layer_idx=0)
    assert layer._chunk_rule_fn      is _hf_chunk_gdr.flash
    assert layer._recurrent_rule_fn  is _hf_recurrent_gdr.flash
    assert layer._causal_conv1d_fn   is _hf_causal_conv1d.flash


_TESTS = [
    test_bridge_registers_triton_and_flash,
    test_bridge_callables_are_binder_originals,
    test_bridge_registers_experts_into_hf_table,
    test_constructed_layer_routes_to_binder,
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
    print(f"\nAll {len(_TESTS)} qwen3_5_moe wiring tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
