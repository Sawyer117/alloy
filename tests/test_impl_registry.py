"""Smoke-test the implementation registry that lets external fast-path
packages (e.g. hf-npu-binder) plug NPU / triton / flash kernels into alloy
without modifying alloy's modules.

Verifies:
  - register_implementation enforces the '<module>.<sub_fn>' dot-form.
  - get_implementation raises clearly on unknown module / unknown impl.
  - get_implementation falls back when ``fallback=`` is provided.
  - list_implementations filters by prefix.
  - Qwen35GatedDeltaNet picks ``"torch"`` impls by default and routes to a
    registered override when ``config._qwen3_5_gdn_implementation`` is set.
  - AlloyConfig.to_json_string strips leading-underscore runtime fields.

Hardware-agnostic: pure CPU torch.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from alloy import AlloyConfig
from alloy.modules.attention.qwen3_5_gdn import (
    Qwen35GatedDeltaNet,
    _torch_chunk_gated_delta_rule,
    _torch_recurrent_gated_delta_rule,
    _torch_causal_conv1d_update,
)
from alloy.modules.registry import (
    IMPL_REGISTRY,
    get_implementation,
    list_implementations,
    register_implementation,
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


def test_register_requires_dot_key() -> None:
    try:
        register_implementation("no_dot_here", "torch", lambda: None)
    except ValueError as e:
        assert "<module>.<sub_fn>" in str(e), e
        return
    raise AssertionError("expected ValueError for non-dot key")


def test_register_rejects_duplicate_without_override() -> None:
    key = "_test_dup.fn"
    register_implementation(key, "x", lambda: 1)
    try:
        register_implementation(key, "x", lambda: 2)
    except ValueError as e:
        assert "override=True" in str(e), e
    else:
        raise AssertionError("expected ValueError on duplicate without override")
    register_implementation(key, "x", lambda: 3, override=True)  # override path
    assert get_implementation(key, "x")() == 3
    del IMPL_REGISTRY[key]


def test_get_unknown_module_raises() -> None:
    try:
        get_implementation("does_not_exist.fn", "torch")
    except KeyError as e:
        assert "does_not_exist.fn" in str(e), e
        return
    raise AssertionError("expected KeyError for unknown module key")


def test_get_unknown_impl_raises_without_fallback() -> None:
    try:
        get_implementation("qwen3_5_gdn.chunk_rule", "made_up_impl")
    except KeyError as e:
        assert "made_up_impl" in str(e) and "Available" in str(e), e
        return
    raise AssertionError("expected KeyError for unknown impl name")


def test_get_falls_back_with_warning() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fn = get_implementation("qwen3_5_gdn.chunk_rule", "made_up_impl", fallback="torch")
    assert fn is _torch_chunk_gated_delta_rule
    assert any("falling back to 'torch'" in str(w.message) for w in caught), (
        f"expected fallback warning, got: {[str(w.message) for w in caught]}"
    )


def test_list_implementations_filters_by_prefix() -> None:
    all_impls = list_implementations()
    assert "qwen3_5_gdn.chunk_rule" in all_impls
    assert "torch" in all_impls["qwen3_5_gdn.chunk_rule"]

    gdn_only = list_implementations("qwen3_5_gdn")
    assert all(k.startswith("qwen3_5_gdn") for k in gdn_only)
    assert {"qwen3_5_gdn.chunk_rule", "qwen3_5_gdn.recurrent_rule",
            "qwen3_5_gdn.causal_conv1d"} <= set(gdn_only)


def test_default_dispatch_uses_torch_impls() -> None:
    cfg = _gdn_config()
    layer = Qwen35GatedDeltaNet(cfg, layer_idx=0)
    assert layer._chunk_rule_fn is _torch_chunk_gated_delta_rule
    assert layer._recurrent_rule_fn is _torch_recurrent_gated_delta_rule
    assert layer._causal_conv1d_fn is _torch_causal_conv1d_update


def test_override_dispatch_via_config_field() -> None:
    sentinel = object()

    def fake_chunk_rule(*args, **kwargs):
        return sentinel  # pragma: no cover — only the identity is checked

    register_implementation("qwen3_5_gdn.chunk_rule", "fake", fake_chunk_rule)
    try:
        cfg = _gdn_config(_qwen3_5_gdn_implementation="fake")
        layer = Qwen35GatedDeltaNet(cfg, layer_idx=0)
        # chunk_rule should be the override; the others fall back to torch
        # because no "fake" was registered for them.
        assert layer._chunk_rule_fn is fake_chunk_rule
        assert layer._recurrent_rule_fn is _torch_recurrent_gated_delta_rule
        assert layer._causal_conv1d_fn is _torch_causal_conv1d_update
    finally:
        del IMPL_REGISTRY["qwen3_5_gdn.chunk_rule"]["fake"]


def test_runtime_field_not_in_json() -> None:
    cfg = _gdn_config(_qwen3_5_gdn_implementation="flash")
    blob = cfg.to_json_string()
    assert "_qwen3_5_gdn_implementation" not in blob, (
        "runtime field _qwen3_5_gdn_implementation must not appear in serialized config.json"
    )
    # And round-trip is still clean — _qwen3_5_gdn_implementation defaults to "torch"
    # on the rebuilt config because it was filtered out.
    import json as _json
    rebuilt = AlloyConfig(**_json.loads(blob))
    assert getattr(rebuilt, "_qwen3_5_gdn_implementation", "torch") == "torch"


_TESTS = [
    test_register_requires_dot_key,
    test_register_rejects_duplicate_without_override,
    test_get_unknown_module_raises,
    test_get_unknown_impl_raises_without_fallback,
    test_get_falls_back_with_warning,
    test_list_implementations_filters_by_prefix,
    test_default_dispatch_uses_torch_impls,
    test_override_dispatch_via_config_field,
    test_runtime_field_not_in_json,
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
        except Exception as e:  # noqa: BLE001 — surface unexpected failures verbatim
            failed += 1
            print(f"  ERR  {fn.__name__}: {type(e).__name__}: {e}")
    if failed:
        print(f"\n{failed}/{len(_TESTS)} test(s) failed.")
        return 1
    print(f"\nAll {len(_TESTS)} registry tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
