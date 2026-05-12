"""RooflineSpec — per-module FLOPs / bytes-movement contracts.

Each registered mixer or FFN name (the keys used by `modules.registry`) gets
a paired :class:`RooflineSpec` that computes the module's contribution to
roofline analysis from ``(input_shape, config, dtype)``. Specs encode the
module's *mathematical* operation under optimal-fusion assumptions, not its
current PyTorch op decomposition — they describe what theory says is
achievable, not what the current implementation actually does.

The spec registry is independent from `modules.registry`: a separate dict
with its own decorator-equivalent (:func:`register_spec`). Forgetting to
register a spec is fail-open by default — the analyzer warns, tags the
module as ``"unknown"``, contributes zero, and continues. Strict mode
(``alloy.roofline(..., strict=True)``) raises.

The fail-open default keeps roofline analysis useful during incremental
spec coverage; strict mode is for CI gates that enforce full coverage.
"""
from __future__ import annotations

import warnings
from abc import ABC, abstractmethod

import torch


# --------------------------------------------------------------------------- #
# Dtype helper
# --------------------------------------------------------------------------- #


_DTYPE_BYTES: dict[torch.dtype, int] = {
    torch.float32: 4,
    torch.bfloat16: 2,
    torch.float16: 2,
    torch.float64: 8,
    torch.int8: 1,
    torch.uint8: 1,
}


def dtype_size(dtype: torch.dtype) -> int:
    """Bytes per element for the given torch dtype.

    Falls back to ``torch.empty((), dtype=dtype).element_size()`` for any
    dtype not in the common-case table (e.g. fp8 variants on newer torch).
    """
    if dtype in _DTYPE_BYTES:
        return _DTYPE_BYTES[dtype]
    return torch.empty((), dtype=dtype).element_size()


# --------------------------------------------------------------------------- #
# Spec contract
# --------------------------------------------------------------------------- #


class RooflineSpec(ABC):
    """Per-module FLOPs / bytes / output-shape under optimal-fusion assumptions.

    Specs are pure functions: they don't construct any nn.Module, don't
    allocate tensors, don't run forward. A spec describes the *math* of
    the module under the assumption that the kernel implementation has
    perfect operator fusion (e.g. FlashAttention-style single kernel for
    softmax(QK^T)V; fused gate-up-down in SwiGLU). This produces the
    roofline upper bound — real implementations land at or below it; the
    gap is the optimization opportunity.

    Three required methods:

    * :meth:`flops` — theoretical FLOPs (one mul-add counted as 2). Same
      number regardless of dtype.
    * :meth:`bytes` — total HBM traffic. Optimal-fusion assumptions:
      weights reloaded once per forward, activations crossing fusion
      boundaries written once and read once, no recomputation.
    * :meth:`out_shape` — output shape given input shape. Default is
      identity (most LLM modules preserve ``[B, T, hidden_size]``);
      shape-changing modules (downsamplers, vision patch embeds, etc.)
      override.
    """

    @abstractmethod
    def flops(self, in_shape: tuple[int, ...], config, **kwargs) -> int:
        """Theoretical FLOPs for this module's forward.

        ``**kwargs`` is reserved for forward-compatible context — currently
        ``kv_cache_len`` (used by attention specs in mini-prefill / decode
        modes) is the only consumer. Specs that don't care about context
        accept and ignore it.
        """
        ...

    @abstractmethod
    def bytes(self, in_shape: tuple[int, ...], config, dtype: torch.dtype, **kwargs) -> int:
        ...

    def out_shape(self, in_shape: tuple[int, ...], config, **kwargs) -> tuple[int, ...]:
        return in_shape


# --------------------------------------------------------------------------- #
# Registry — independent from modules.registry
# --------------------------------------------------------------------------- #


SPEC_REGISTRY: dict[str, RooflineSpec] = {}


def register_spec(name: str, spec: RooflineSpec, *, override: bool = False) -> None:
    """Bind a :class:`RooflineSpec` to a registered mixer / FFN name.

    Names should match the keys used by ``register_mixer`` / ``register_ffn``
    in ``modules.registry``. ``override=True`` is required to replace an
    existing entry — catches accidental double registration when the same
    spec module is imported twice.
    """
    if not isinstance(spec, RooflineSpec):
        raise TypeError(
            f"register_spec expects a RooflineSpec instance, got {type(spec).__name__}."
        )
    if name in SPEC_REGISTRY and not override:
        raise ValueError(
            f"Spec for '{name}' already registered. Pass override=True to replace."
        )
    SPEC_REGISTRY[name] = spec


def get_spec(name: str, *, strict: bool = False) -> RooflineSpec | None:
    """Look up a spec by name.

    ``strict=True`` raises ``KeyError`` if missing. Default ``False`` warns
    once and returns ``None`` — the analyzer then tags the module as
    ``"unknown"`` in the report so the gap is visible without crashing
    partial coverage. CI gates should pass ``strict=True`` to enforce
    spec coverage for new modules.
    """
    if name in SPEC_REGISTRY:
        return SPEC_REGISTRY[name]
    if strict:
        raise KeyError(
            f"No RooflineSpec registered for '{name}'. "
            f"Registered: {sorted(SPEC_REGISTRY)}."
        )
    warnings.warn(
        f"No RooflineSpec for '{name}' — counted as 0 in the analysis. "
        f"Registered specs: {sorted(SPEC_REGISTRY)}.",
        stacklevel=3,
    )
    return None


# --------------------------------------------------------------------------- #
# Generic specs — building blocks for model-specific composite specs.
# --------------------------------------------------------------------------- #


class RMSNormSpec(RooflineSpec):
    """Spec for RMSNorm: ``y = x * rsqrt(mean(x**2) + eps) * weight``.

    FLOPs: ~4 per element (square, mean accum, rsqrt+eps, mul-back-with-weight).
    The constant is a tight approximation; matmul costs around it dominate by
    100-1000x in normal LLM forward, so even a 2x error here is below the
    noise floor.

    Bytes (optimal fusion): read input + read weight (small ``[H]`` vector) +
    write output. No constructor args — reads ``hidden`` from ``in_shape[-1]``.
    """

    def flops(self, in_shape: tuple[int, ...], config=None, **kwargs) -> int:
        n_elements = 1
        for d in in_shape:
            n_elements *= d
        return 4 * n_elements

    def bytes(self, in_shape: tuple[int, ...], config=None, dtype: torch.dtype = torch.bfloat16, **kwargs) -> int:
        n_elements = 1
        for d in in_shape:
            n_elements *= d
        es = dtype_size(dtype)
        hidden = in_shape[-1]
        weight = hidden * es
        act_in = n_elements * es
        act_out = n_elements * es
        return weight + act_in + act_out


class LinearSpec(RooflineSpec):
    """Spec for a single ``nn.Linear`` projection.

    FLOPs: ``2 * batch_tokens * in_features * out_features`` (mul + add per
    weight per token); bias adds ``batch_tokens * out_features`` extra adds,
    typically negligible.

    Bytes (optimal fusion):
      * weights:  ``in_features * out_features * dtype_size``
      * bias:     ``out_features * dtype_size`` (when present)
      * act_in:   ``batch_tokens * in_features * dtype_size``
      * act_out:  ``batch_tokens * out_features * dtype_size``

    Constructor args fix the projection dims so a ``LinearSpec`` instance
    describes one specific linear — model-specific composite specs build on
    top of this by holding a list of ``LinearSpec`` instances and summing
    their contributions.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False) -> None:
        if in_features <= 0 or out_features <= 0:
            raise ValueError(
                f"Linear dims must be positive, got ({in_features}, {out_features})."
            )
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias

    def _check_in_shape(self, in_shape: tuple[int, ...]) -> int:
        if in_shape[-1] != self.in_features:
            raise ValueError(
                f"LinearSpec({self.in_features}, {self.out_features}) called with "
                f"in_shape last-dim={in_shape[-1]}; mismatch."
            )
        n_tokens = 1
        for d in in_shape[:-1]:
            n_tokens *= d
        return n_tokens

    def flops(self, in_shape: tuple[int, ...], config=None, **kwargs) -> int:
        n_tokens = self._check_in_shape(in_shape)
        macs = n_tokens * self.in_features * self.out_features
        bias_flops = n_tokens * self.out_features if self.bias else 0
        return 2 * macs + bias_flops

    def bytes(self, in_shape: tuple[int, ...], config=None, dtype: torch.dtype = torch.bfloat16, **kwargs) -> int:
        n_tokens = self._check_in_shape(in_shape)
        es = dtype_size(dtype)
        weights = self.in_features * self.out_features * es
        bias_bytes = self.out_features * es if self.bias else 0
        act_in = n_tokens * self.in_features * es
        act_out = n_tokens * self.out_features * es
        return weights + bias_bytes + act_in + act_out

    def out_shape(self, in_shape: tuple[int, ...], config=None, **kwargs) -> tuple[int, ...]:
        if in_shape[-1] != self.in_features:
            raise ValueError(
                f"LinearSpec out_shape: expected last dim {self.in_features}, "
                f"got {in_shape[-1]}."
            )
        return (*in_shape[:-1], self.out_features)


__all__ = [
    "dtype_size",
    "RooflineSpec",
    "SPEC_REGISTRY",
    "register_spec",
    "get_spec",
    "LinearSpec",
    "RMSNormSpec",
]
