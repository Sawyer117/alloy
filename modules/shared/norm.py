"""Source-coupled RMSNorm variants.

alloy follows HF's per-model class convention — even when two models'
RMSNorms are mathematically equivalent, they get separate class names so
state_dict provenance, debug prints, and source-of-truth references are
unambiguous. The math is small enough that the duplication cost is
trivial.

Variants currently shipped:

  - :class:`Qwen3RMSNorm` — ones-init, ``y = w * rms(x)``. Used by
    Qwen3 / LLaMA-style models.
  - :class:`Qwen35RMSNorm` — zero-init, unit-offset, ``y = (1+w) * rms(x)``.
    Used by Qwen3.5 / Qwen3-Next / DeepSeek-V3.
  - :class:`Qwen35RMSNormGated` — variant used inside Qwen3.5
    GatedDeltaNet: applies an extra ``silu(gate)`` multiplication after
    the standard ones-init RMS step.

Backward compat: :func:`RMSNorm` is a factory function (was a class) that
maps the legacy ``unit_offset=True/False`` kwarg to the matching named
class. ``RMSNormGated`` is an alias for ``Qwen35RMSNormGated``.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class Qwen3RMSNorm(nn.Module):
    """Standard ones-init RMSNorm: ``y = w * rms(x)``.

    Used by Qwen3 dense models and any qwen3-flavoured port. ``weight`` is
    initialised to ones so the layer starts as identity at the residual
    scale of the model.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.variance_epsilon = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        x_fp32 = hidden_states.to(torch.float32)
        variance = x_fp32.pow(2).mean(-1, keepdim=True)
        x_normed = x_fp32 * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * x_normed.to(input_dtype)

    def _alloy_init_weights(self, init_std: float) -> None:
        del init_std
        nn.init.ones_(self.weight)

    def extra_repr(self) -> str:
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class Qwen35RMSNorm(nn.Module):
    """Unit-offset zero-init RMSNorm: ``y = (1 + w) * rms(x)``.

    Used by Qwen3.5 / Qwen3-Next / DeepSeek-V3. ``weight`` is initialised
    to zeros so the ``(1 + w)`` factor starts as identity, which lets the
    surrounding layer initialise stably without an extra scale parameter.
    The ``(1 + w)`` multiplication is computed in fp32 to dodge bf16 noise
    on small weight magnitudes; the cast back to ``input_dtype`` happens
    at the end.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.variance_epsilon = eps
        self.weight = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        x_fp32 = hidden_states.to(torch.float32)
        variance = x_fp32.pow(2).mean(-1, keepdim=True)
        x_normed = x_fp32 * torch.rsqrt(variance + self.variance_epsilon)
        output = x_normed * (1.0 + self.weight.float())
        return output.to(input_dtype)

    def _alloy_init_weights(self, init_std: float) -> None:
        del init_std
        nn.init.zeros_(self.weight)

    def extra_repr(self) -> str:
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}, unit_offset=True"


class Qwen35RMSNormGated(nn.Module):
    """Gated RMSNorm used by Qwen3.5 GatedDeltaNet (port of
    ``Qwen3_5MoeRMSNormGated``).

    Normalises ``hidden_states`` with a ones-init weight, multiplies by
    ``silu(gate)`` to gate the output. Despite the qwen3.5 family name,
    this variant uses ones-init (not unit-offset zero-init) — that's how
    the upstream class is defined.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        hidden_states = self.weight * hidden_states.to(input_dtype)
        hidden_states = hidden_states * F.silu(gate.to(torch.float32))
        return hidden_states.to(input_dtype)

    def _alloy_init_weights(self, init_std: float) -> None:
        del init_std
        nn.init.ones_(self.weight)


# --------------------------------------------------------------------------- #
# Backward-compat shims
# --------------------------------------------------------------------------- #
# Earlier alloy code constructed RMSNorm with a parametric ``unit_offset``
# kwarg. Keep those callers working: ``RMSNorm`` is now a factory that
# returns the matching named class. New code should construct
# ``Qwen3RMSNorm`` / ``Qwen35RMSNorm`` directly. The factory returns an
# already-constructed module so the call site doesn't change.
def RMSNorm(hidden_size: int, eps: float = 1e-6, unit_offset: bool = False) -> nn.Module:
    """Deprecated factory: prefer constructing :class:`Qwen3RMSNorm` /
    :class:`Qwen35RMSNorm` directly. Kept so existing callers keep working
    without churn.
    """
    cls = Qwen35RMSNorm if unit_offset else Qwen3RMSNorm
    return cls(hidden_size, eps)


# Old name; new code should use Qwen35RMSNormGated directly.
RMSNormGated = Qwen35RMSNormGated
