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
    experts as _hf_experts,
    fused_recurrent_gated_delta_rule as _hf_recurrent_gdr,
)

# alloy's own per-module dispatch table (GDN sub-functions live here).
from alloy.modules.registry import register_implementation

# HuggingFace's MoE experts dispatch table. alloy's ``_Experts`` is wrapped by
# ``@use_experts_implementation`` and reads ``config._experts_implementation``,
# so the binder's whole-experts fast path plugs in here, not into alloy's
# IMPL_REGISTRY.
from transformers.integrations.moe import ALL_EXPERTS_FUNCTIONS


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

# (hf_table_key, callable). The MoE experts forward is **whole-block** —
# permute + GMM + swiglu + GMM + unpermute happens in one HF dispatch
# entry, not split into per-op alloy sub-functions. alloy already wraps
# its ``_Experts`` with ``@use_experts_implementation``, so registering
# the binder callable here is sufficient.
_HF_EXPERTS_BINDINGS: tuple[tuple[str, object], ...] = (
    ("flash", _hf_experts.flash),
)

# Config field names that ``activate(prefer="<backend>")`` will broadcast a
# backend choice across. Each entry corresponds to one dispatch surface the
# bridge has registered into above. Future module additions append here.
_ACTIVATABLE_FIELDS: list[str] = []


def _register_all() -> None:
    for alloy_key, impl_name, fn in _QWEN3_5_GDN_BINDINGS:
        # ``override=True`` tolerates re-import during interactive sessions
        # and avoids a hard crash if a user's environment somehow runs the
        # bridge twice. Backends here come from a single binder version so
        # the override is identity-on-equal in normal use.
        register_implementation(alloy_key, impl_name, fn, override=True)
    _ACTIVATABLE_FIELDS.append("_qwen3_5_gdn_implementation")

    for hf_key, fn in _HF_EXPERTS_BINDINGS:
        # ALL_EXPERTS_FUNCTIONS is a dict-like; assignment is the public form
        # of register and tolerates rebinding cleanly.
        ALL_EXPERTS_FUNCTIONS[hf_key] = fn
    _ACTIVATABLE_FIELDS.append("_experts_implementation")


_register_all()


# ---------------------------------------------------------------------------
# activate(): sugar for setting _<module_key>_implementation on model.config
# ---------------------------------------------------------------------------
def _normalise_field(k: str) -> str:
    """Accept either a fully-qualified field name (``"_qwen3_5_gdn_implementation"``)
    or a bare module key (``"qwen3_5_gdn"`` / ``"experts"``).
    """
    if k.startswith("_") and k.endswith("_implementation"):
        return k
    return f"_{k}_implementation"


def activate(model, prefer: str | Mapping[str, str]) -> dict[str, str]:
    """Set fast-path selection fields on ``model.config``.

    Args:
        model: any object with a ``.config`` attribute (typically an
            ``AlloyForCausalLM``).
        prefer: either
            - a single backend name (``"flash"`` / ``"triton"`` / ``"torch"`` / ...)
              broadcast to every dispatch surface this bridge has wired up
              (currently: alloy's ``_qwen3_5_gdn_implementation`` and
              HF's ``_experts_implementation``), OR
            - a mapping ``{"qwen3_5_gdn": "flash", "experts": "flash", ...}``
              (or fully-qualified field names) for explicit choices.

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
        HF's experts dispatch reads the field on every forward, so flipping
        ``_experts_implementation`` post-construction works.
    """
    if not hasattr(model, "config"):
        raise TypeError(
            f"activate(model, ...) expects an object with a `.config` attribute, "
            f"got {type(model).__name__}"
        )
    config = model.config

    if isinstance(prefer, str):
        chosen: dict[str, str] = {field: prefer for field in _ACTIVATABLE_FIELDS}
    else:
        chosen = {_normalise_field(k): v for k, v in prefer.items()}

    for field, impl in chosen.items():
        setattr(config, field, impl)

    return chosen


__all__ = ["activate", "hf_npu_binder"]
__version__ = hf_npu_binder.__version__
