from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Callable, Literal

import torch.nn as nn


MaskKind = Literal["causal", "sliding", "linear"]


@dataclass(frozen=True)
class MixerEntry:
    cls: type[nn.Module]
    attr_name: str  # name under which the module is stored on the DecoderLayer (matches HF state_dict)
    mask_kind: MaskKind = "causal"


MIXER_REGISTRY: dict[str, MixerEntry] = {}
FFN_REGISTRY: dict[str, type[nn.Module]] = {}

# Two-level table of swappable per-module implementations:
#   IMPL_REGISTRY["<module_key>.<sub_function>"]["<impl_name>"] = callable
# The top-level key uses a dot to mirror the file layout of an external
# fast-path package (e.g. ``hf_npu_binder/models/qwen3_5_gdn/chunk_rule.py``
# corresponds to ``"qwen3_5_gdn.chunk_rule"``). The module that defines a
# dispatch site is responsible for registering the default ``"torch"`` impl
# at import time; external packages add ``"triton"`` / ``"flash"`` / etc.
# alongside without modifying alloy.
IMPL_REGISTRY: dict[str, dict[str, Callable]] = {}


# Per-module-key default backend, populated by external bridges (e.g.
# ``alloy.integrations.hf_npu_binder``). Empty by default — alloy modules
# that find no entry here fall back to ``"torch"``.
#
# Keys are the bare ``"<module_key>"`` part (e.g. ``"qwen3_5_gdn"``), not
# the dotted ``"<module_key>.<sub_function>"`` form used in IMPL_REGISTRY.
# Rationale: a single backend choice ("triton" / "flash" / "tilelang" / ...)
# typically covers all sub-functions of one module; the per-sub-function
# split is for callable lookup, not for default selection.
#
# Why not hardcode defaults inside alloy modules:
#   alloy is backend-agnostic — it knows which dispatch surfaces exist but
#   not which backend is best on a given hardware. Backend-specific
#   knowledge ("on Ascend 910B, triton beats flash") lives in the package
#   that ships those kernels (e.g. hf_npu_binder), and that package
#   declares its preferences via its own bridge populating this dict.
#   When a future module gets a tilelang backend, the new bridge writes
#   ``DEFAULT_IMPL["mamba3_ssm"] = "tilelang"`` and alloy core stays
#   unchanged.
DEFAULT_IMPL: dict[str, str] = {}


def register_mixer(
    name: str,
    attr_name: str,
    mask_kind: MaskKind = "causal",
) -> Callable[[type[nn.Module]], type[nn.Module]]:
    """
    Register a token-mixer (attention-like) module under a string key.

    `attr_name` determines the attribute on the DecoderLayer where the module is stored.
    For HF state_dict compatibility use:
      - "self_attn"   for full / sliding / gated GQA
      - "linear_attn" for linear-attention (e.g. Qwen35GatedDeltaNet)

    `mask_kind` declares which mask family this mixer needs at the model level:
      - "causal"  : standard causal mask
      - "sliding" : causal mask with a sliding window
      - "linear"  : 2D padding mask consumed inside the recurrent kernel
    """

    def _wrap(cls: type[nn.Module]) -> type[nn.Module]:
        if name in MIXER_REGISTRY:
            raise ValueError(f"Mixer '{name}' already registered")
        MIXER_REGISTRY[name] = MixerEntry(cls=cls, attr_name=attr_name, mask_kind=mask_kind)
        return cls

    return _wrap


def register_ffn(name: str) -> Callable[[type[nn.Module]], type[nn.Module]]:
    """Register a feed-forward (channel mixer) module under a string key.

    The FFN is always stored on the DecoderLayer as `self.mlp` to match
    HF state_dict layout (qwen3 / qwen3.5 / llama all use `.mlp`).
    """

    def _wrap(cls: type[nn.Module]) -> type[nn.Module]:
        if name in FFN_REGISTRY:
            raise ValueError(f"FFN '{name}' already registered")
        FFN_REGISTRY[name] = cls
        return cls

    return _wrap


def get_mixer(name: str) -> MixerEntry:
    if name not in MIXER_REGISTRY:
        raise KeyError(
            f"Unknown mixer '{name}'. Registered: {sorted(MIXER_REGISTRY)}. "
            f"Did you forget to import the module that defines it?"
        )
    return MIXER_REGISTRY[name]


def get_ffn(name: str) -> type[nn.Module]:
    if name not in FFN_REGISTRY:
        raise KeyError(
            f"Unknown FFN '{name}'. Registered: {sorted(FFN_REGISTRY)}. "
            f"Did you forget to import the module that defines it?"
        )
    return FFN_REGISTRY[name]


def register_implementation(
    key: str,
    impl_name: str,
    fn: Callable,
    *,
    override: bool = False,
) -> None:
    """Register a swappable implementation of a sub-function for some alloy module.

    ``key`` must be ``"<module_key>.<sub_function>"`` (e.g. ``"qwen3_5_gdn.chunk_rule"``).
    The dot-form is enforced so the registry stays grep-friendly and mirrors
    the file layout of external fast-path packages.

    ``impl_name`` is a short user-facing handle (``"torch"``, ``"triton"``,
    ``"flash"``, ``"npu_fused"``, ...).

    ``override=True`` is required to replace an existing entry — this catches
    accidental double registration when an external package is imported twice.
    """
    if "." not in key:
        raise ValueError(
            f"register_implementation key must be '<module>.<sub_fn>', got '{key}'. "
            f"Examples: 'qwen3_5_gdn.chunk_rule', 'qwen3_5_gdn.causal_conv1d'."
        )
    table = IMPL_REGISTRY.setdefault(key, {})
    if impl_name in table and not override:
        raise ValueError(
            f"Implementation '{impl_name}' already registered for {key}. "
            f"Pass override=True to replace."
        )
    table[impl_name] = fn


def get_implementation(
    key: str,
    impl_name: str,
    *,
    fallback: str | None = None,
) -> Callable:
    """Look up an implementation registered under ``key``.

    If ``impl_name`` is missing and ``fallback`` is provided, fall back to
    ``fallback`` (warning the caller). The dispatch site in an alloy module
    typically passes ``fallback="torch"`` so that picking a backend that hasn't
    registered every sub-function gracefully degrades to torch for the gaps,
    rather than blowing up at ``__init__`` time.
    """
    if key not in IMPL_REGISTRY:
        raise KeyError(
            f"No implementations registered for '{key}'. "
            f"Did you forget to import the module that defines it?"
        )
    table = IMPL_REGISTRY[key]
    if impl_name in table:
        return table[impl_name]
    if fallback is not None and fallback in table:
        warnings.warn(
            f"No '{impl_name}' implementation registered for {key}; "
            f"falling back to '{fallback}'. Available: {sorted(table)}.",
            stacklevel=2,
        )
        return table[fallback]
    raise KeyError(
        f"Unknown implementation '{impl_name}' for {key}. Available: {sorted(table)}."
    )


def list_implementations(prefix: str = "") -> dict[str, list[str]]:
    """Discovery helper. ``prefix=""`` lists everything; ``prefix="qwen3_5_gdn"``
    filters to one module's sub-functions.
    """
    return {
        k: sorted(v)
        for k, v in IMPL_REGISTRY.items()
        if k.startswith(prefix)
    }
