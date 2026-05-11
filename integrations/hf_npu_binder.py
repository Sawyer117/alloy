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
from alloy.modules.registry import DEFAULT_IMPL, register_implementation

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

# DSV4 CSA SFA: kernel port from MindSpeed-LLM is pending (the binder's
# ``deepseek_v4.sparse_flash_attention.triton`` raises NotImplementedError
# until then). We deliberately do NOT add it to the bindings tuple here —
# leaving it unregistered means a config request of
# ``_dsv4_csa_implementation="triton"`` resolves to ``"torch"`` via
# ``get_implementation(..., fallback="torch")``, instead of crashing with
# NotImplementedError on first CSA forward. The activatable field is still
# wired below so ``activate(model, "auto")`` reaches it; DEFAULTS' DSV4 entry
# (which maps every intent to ``"torch"`` while the port is pending) is what
# makes that resolve cleanly.
_DEEPSEEK_V4_BINDINGS: tuple[tuple[str, str, object], ...] = ()

# Config field names that ``activate(prefer="<backend>")`` will broadcast a
# backend choice across, paired with the *primary* binder operator key the
# bridge looks up in ``hf_npu_binder.DEFAULTS`` to translate a user intent
# (``"auto"`` / ``"flash"`` / ...) into the actual impl name set on the
# field. The per-sub-function ``get_implementation(..., fallback="torch")``
# call inside each alloy module handles the case where one sub-function of
# a module has a kernel but a sibling does not.
_ACTIVATABLE_FIELDS: list[tuple[str, str]] = []  # (field_name, primary_op_key)


def _register_all() -> None:
    for alloy_key, impl_name, fn in _QWEN3_5_GDN_BINDINGS:
        # ``override=True`` tolerates re-import during interactive sessions
        # and avoids a hard crash if a user's environment somehow runs the
        # bridge twice. Backends here come from a single binder version so
        # the override is identity-on-equal in normal use.
        register_implementation(alloy_key, impl_name, fn, override=True)
    _ACTIVATABLE_FIELDS.append(
        ("_qwen3_5_gdn_implementation", "qwen3_5_moe.chunk_gated_delta_rule")
    )

    for hf_key, fn in _HF_EXPERTS_BINDINGS:
        # ALL_EXPERTS_FUNCTIONS is a dict-like; assignment is the public form
        # of register and tolerates rebinding cleanly.
        ALL_EXPERTS_FUNCTIONS[hf_key] = fn
    _ACTIVATABLE_FIELDS.append(("_experts_implementation", "qwen3_5_moe.experts"))

    for alloy_key, impl_name, fn in _DEEPSEEK_V4_BINDINGS:
        register_implementation(alloy_key, impl_name, fn, override=True)
    # DSV4 SFA: field is activatable even though the binder has no
    # callable to register yet — DEFAULTS' DSV4 entry routes every intent
    # to ``"torch"`` so ``activate(model, "...")`` still produces a
    # working configuration (alloy's own torch impl is what gets called).
    _ACTIVATABLE_FIELDS.append(
        ("_dsv4_csa_implementation", "deepseek_v4.sparse_flash_attention")
    )

    # Per-bare-module-key default impl, consulted by alloy modules when
    # the user has not set ``config._<module>_implementation`` explicitly.
    # Source of truth is binder's own ``DEFAULTS`` table — the package
    # that actually benchmarks these on hardware decides what "auto"
    # means for each module.
    binder_defaults: dict = getattr(hf_npu_binder, "DEFAULTS", {})
    DEFAULT_IMPL["qwen3_5_gdn"] = _resolve_intent(
        binder_defaults, "qwen3_5_moe.chunk_gated_delta_rule", "auto",
    )
    DEFAULT_IMPL["dsv4_csa"] = _resolve_intent(
        binder_defaults, "deepseek_v4.sparse_flash_attention", "auto",
    )


def _resolve_intent(
    defaults: dict, binder_op_key: str, intent: str,
) -> str:
    """Translate a user-facing intent (``"auto"`` / ``"flash"`` / ``"triton"``
    / ...) into the actual impl name binder recommends for the given
    operator. Falls back to ``"torch"`` if either the operator or the
    intent is missing from ``hf_npu_binder.DEFAULTS`` — that always
    resolves through ``get_implementation(..., fallback="torch")`` to a
    working callable, since every alloy dispatch surface registers a
    torch impl at module-import time.

    Tolerates the legacy ``dict[str, str]`` shape (single recommended
    impl per operator, no per-intent mapping) so older binder packages
    keep working.
    """
    entry = defaults.get(binder_op_key)
    if entry is None:
        return "torch"
    if isinstance(entry, str):
        return entry  # legacy: single recommended impl
    return entry.get(intent, "torch")


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
            - a single **intent** name (``"auto"`` / ``"flash"`` / ``"triton"``
              / ``"torch"`` / ...) broadcast across every dispatch surface
              this bridge has wired up. The actual impl name written to each
              field is resolved per-operator through
              ``hf_npu_binder.DEFAULTS`` — so an operator with no flash
              kernel gets the binder's recommended fallback (typically
              ``"triton"`` or ``"torch"``) instead of an unresolvable
              ``"flash"`` string, OR
            - a mapping ``{"qwen3_5_gdn": "flash", "experts": "flash", ...}``
              (or fully-qualified field names) for explicit per-module
              choices. Mapping values are taken **literally** — no
              DEFAULTS translation — since the user has named the impl.

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

    Example:
        >>> activate(model, "auto")  # binder picks the best per operator
        {'_qwen3_5_gdn_implementation': 'triton',
         '_experts_implementation': 'flash',
         '_dsv4_csa_implementation': 'torch'}        # SFA kernel pending
        >>> activate(model, "flash")  # broadcast; DSV4 has no flash, falls back
        {'_qwen3_5_gdn_implementation': 'flash',
         '_experts_implementation': 'flash',
         '_dsv4_csa_implementation': 'torch'}
        >>> activate(model, {"qwen3_5_gdn": "torch"})  # explicit override
        {'_qwen3_5_gdn_implementation': 'torch'}
    """
    if not hasattr(model, "config"):
        raise TypeError(
            f"activate(model, ...) expects an object with a `.config` attribute, "
            f"got {type(model).__name__}"
        )
    config = model.config
    binder_defaults: dict = getattr(hf_npu_binder, "DEFAULTS", {})

    if isinstance(prefer, str):
        intent = prefer
        chosen: dict[str, str] = {
            field: _resolve_intent(binder_defaults, op_key, intent)
            for field, op_key in _ACTIVATABLE_FIELDS
        }
    else:
        chosen = {_normalise_field(k): v for k, v in prefer.items()}

    for field, impl in chosen.items():
        setattr(config, field, impl)

    return chosen


__all__ = ["activate", "hf_npu_binder"]
__version__ = hf_npu_binder.__version__
