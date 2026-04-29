"""Integration test: ``alloy.integrations.hf_npu_binder`` bridge.

Verifies the bridge behaviour when the optional ``hf_npu_binder`` package is
installed:

  - The bridge import registers binder backends ("triton" / "flash") into
    ``alloy.modules.registry.IMPL_REGISTRY`` under the canonical alloy keys.
  - ``activate(model, prefer=...)`` writes the right ``_<key>_implementation``
    fields on ``model.config``.
  - A subsequently-constructed ``Qwen35GatedDeltaNet`` routes its sub-functions
    to the binder callables, not alloy's torch defaults.

If ``hf_npu_binder`` is not installed, the whole file no-ops (printed SKIP) —
the bridge is genuinely optional.

Pure CPU torch.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

try:
    import hf_npu_binder  # noqa: F401  -- bridge dep; bail out below if absent
except ImportError:
    print("SKIP — hf_npu_binder not installed; bridge integration not exercised.")
    sys.exit(0)

from alloy import AlloyConfig
from alloy.modules.attention.qwen3_5_gdn import Qwen35GatedDeltaNet
from alloy.modules.registry import list_implementations

# Importing the bridge has the side effect of registering binder backends.
import alloy.integrations.hf_npu_binder as bridge
from hf_npu_binder.qwen3_5_moe import (
    causal_conv1d as _hf_causal_conv1d,
    chunk_gated_delta_rule as _hf_chunk_gdr,
    fused_recurrent_gated_delta_rule as _hf_recurrent_gdr,
)


def _gdn_config(**override) -> AlloyConfig:
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
    from alloy.modules.registry import get_implementation

    assert get_implementation("qwen3_5_gdn.chunk_rule", "triton") is _hf_chunk_gdr.triton
    assert get_implementation("qwen3_5_gdn.chunk_rule", "flash")  is _hf_chunk_gdr.flash
    assert get_implementation("qwen3_5_gdn.recurrent_rule", "triton") is _hf_recurrent_gdr.triton
    assert get_implementation("qwen3_5_gdn.recurrent_rule", "flash")  is _hf_recurrent_gdr.flash
    assert get_implementation("qwen3_5_gdn.causal_conv1d", "triton") is _hf_causal_conv1d.triton
    assert get_implementation("qwen3_5_gdn.causal_conv1d", "flash")  is _hf_causal_conv1d.flash


def test_activate_broadcast() -> None:
    cfg = _gdn_config()
    model = type("FakeModel", (), {"config": cfg})()
    chosen = bridge.activate(model, prefer="flash")
    assert chosen == {"_qwen3_5_gdn_implementation": "flash"}, chosen
    assert getattr(cfg, "_qwen3_5_gdn_implementation") == "flash"


def test_activate_explicit_mapping_with_bare_module_key() -> None:
    cfg = _gdn_config()
    model = type("FakeModel", (), {"config": cfg})()
    chosen = bridge.activate(model, prefer={"qwen3_5_gdn": "triton"})
    assert chosen == {"_qwen3_5_gdn_implementation": "triton"}
    assert getattr(cfg, "_qwen3_5_gdn_implementation") == "triton"


def test_constructed_layer_routes_to_binder() -> None:
    cfg = _gdn_config()
    model = type("FakeModel", (), {"config": cfg})()
    bridge.activate(model, prefer="flash")

    layer = Qwen35GatedDeltaNet(cfg, layer_idx=0)
    assert layer._chunk_rule_fn      is _hf_chunk_gdr.flash
    assert layer._recurrent_rule_fn  is _hf_recurrent_gdr.flash
    assert layer._causal_conv1d_fn   is _hf_causal_conv1d.flash


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
    test_bridge_registers_triton_and_flash,
    test_bridge_callables_are_binder_originals,
    test_activate_broadcast,
    test_activate_explicit_mapping_with_bare_module_key,
    test_constructed_layer_routes_to_binder,
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
    print(f"\nAll {len(_TESTS)} bridge tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
