"""Bridge: ``hf-npu-binder`` kernel package → alloy ``IMPL_REGISTRY``.

Importing this module (opt-in, never auto-loaded by alloy core):

  1. Imports ``hf_npu_binder`` (raises a clear ``ImportError`` if not installed).
  2. Registers each backend the binder ships under the canonical alloy
     ``<module_key>.<sub_function>`` keys.
  3. Exposes :func:`activate` — a thin sugar that sets
     ``_<module_key>_implementation`` fields on a model's config so that
     dispatch picks the binder backends at module ``__init__`` time.

The binder package itself is consumer-agnostic — it does not know alloy
exists. All knowledge of how alloy's registry is shaped lives in this
file. When alloy adds new modules with binder-served sub-functions, this
bridge picks up the new keys; binder gains new backends by adding files
under its own ``hf_npu_binder/<family>/`` tree, and this bridge wires them.
"""
from __future__ import annotations

from typing import Mapping

import hf_npu_binder
from hf_npu_binder.qwen3_5_moe import (
    causal_conv1d as _hf_causal_conv1d,
    chunk_gated_delta_rule as _hf_chunk_gdr,
    fused_recurrent_gated_delta_rule as _hf_recurrent_gdr,
)

from alloy.modules.registry import IMPL_REGISTRY, register_implementation


# ---------------------------------------------------------------------------
# Registration: HF-named binder callables → alloy registry keys.
#
# alloy's registry uses source-coupled keys (``qwen3_5_gdn``); binder uses
# HF-canonical names (``qwen3_5_moe.chunk_gated_delta_rule``). The bridge is
# the one place that knows the mapping.
# ---------------------------------------------------------------------------
_QWEN3_5_GDN_BINDINGS: tuple[tuple[str, str, object], ...] = (
    # (alloy_key, impl_name, callable)
    ("qwen3_5_gdn.chunk_rule",     "triton", _hf_chunk_gdr.triton),
    ("qwen3_5_gdn.chunk_rule",     "flash",  _hf_chunk_gdr.flash),
    ("qwen3_5_gdn.recurrent_rule", "triton", _hf_recurrent_gdr.triton),
    ("qwen3_5_gdn.recurrent_rule", "flash",  _hf_recurrent_gdr.flash),
    ("qwen3_5_gdn.causal_conv1d",  "triton", _hf_causal_conv1d.triton),
    ("qwen3_5_gdn.causal_conv1d",  "flash",  _hf_causal_conv1d.flash),
)


def _register_all() -> None:
    for alloy_key, impl_name, fn in _QWEN3_5_GDN_BINDINGS:
        # ``override=True`` tolerates re-import during interactive sessions
        # and avoids a hard crash if a user's environment somehow runs the
        # bridge twice. Backends here come from a single binder version so
        # the override is identity-on-equal in normal use.
        register_implementation(alloy_key, impl_name, fn, override=True)


_register_all()


# ---------------------------------------------------------------------------
# activate(): sugar for setting _<module_key>_implementation on model.config
# ---------------------------------------------------------------------------
def _binder_module_keys() -> set[str]:
    """alloy module keys for which binder has registered at least one impl.

    Reads ``IMPL_REGISTRY`` after registration has run, so this is naturally
    in sync with whatever this bridge currently wires.
    """
    keys: set[str] = set()
    for key, impls in IMPL_REGISTRY.items():
        if "." not in key:
            continue
        if any(name != "torch" for name in impls):
            keys.add(key.split(".", 1)[0])
    return keys


def _field_name(module_key: str) -> str:
    return f"_{module_key}_implementation"


def activate(model, prefer: str | Mapping[str, str]) -> dict[str, str]:
    """Set fast-path selection fields on ``model.config``.

    Args:
        model: any object with a ``.config`` attribute (typically an
            ``AlloyForCausalLM``).
        prefer: either
            - a single backend name (``"flash"`` / ``"triton"`` / ``"torch"`` / ...)
              applied to every alloy module that has at least one binder impl, OR
            - a mapping ``{"qwen3_5_gdn": "flash", ...}`` (or fully-qualified
              field names ``"_qwen3_5_gdn_implementation"``) for explicit choices.

    Returns:
        A dict ``{field_name: chosen_impl}`` describing what was set.

    Notes:
        Fields starting with ``_`` are filtered out of
        ``AlloyConfig.to_json_string`` so they never leak into ``config.json``
        on ``save_pretrained``.

        For modules already constructed before ``activate`` is called, the
        per-instance ``self._chunk_rule_fn`` etc. attributes were resolved
        at ``__init__`` and are NOT updated. To switch a live model,
        reconstruct it after calling ``activate``, or set the per-instance
        callable directly via :func:`alloy.modules.registry.get_implementation`.
    """
    if not hasattr(model, "config"):
        raise TypeError(
            f"activate(model, ...) expects an object with a `.config` attribute, "
            f"got {type(model).__name__}"
        )
    config = model.config

    if isinstance(prefer, str):
        chosen: dict[str, str] = {
            _field_name(mk): prefer for mk in _binder_module_keys()
        }
    else:
        chosen = {}
        for k, v in prefer.items():
            field = k if k.startswith("_") and k.endswith("_implementation") else _field_name(k)
            chosen[field] = v

    for field, impl in chosen.items():
        setattr(config, field, impl)

    return chosen


__all__ = ["activate", "hf_npu_binder"]
__version__ = hf_npu_binder.__version__
