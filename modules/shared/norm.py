from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class RMSNorm(nn.Module):
    """Parameterized RMSNorm covering both qwen3 and qwen3.5 styles.

    - unit_offset=False (qwen3-style):  out = w * rms_norm(x),  weight init ones_
    - unit_offset=True  (qwen3.5-style): out = (1 + w) * rms_norm(x), weight init zeros_

    The two are algebraically equivalent under w' = 1 + w but differ in parameter
    storage, init, and therefore checkpoint compatibility. Pick the style matching
    the checkpoint you intend to load.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6, unit_offset: bool = False) -> None:
        super().__init__()
        self.variance_epsilon = eps
        self.unit_offset = unit_offset
        init = torch.zeros(hidden_size) if unit_offset else torch.ones(hidden_size)
        self.weight = nn.Parameter(init)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        x_fp32 = hidden_states.to(torch.float32)
        variance = x_fp32.pow(2).mean(-1, keepdim=True)
        x_normed = x_fp32 * torch.rsqrt(variance + self.variance_epsilon)
        if self.unit_offset:
            # qwen3.5 style: compute (1 + w) * x in fp32, cast back at the end.
            output = x_normed * (1.0 + self.weight.float())
            return output.to(input_dtype)
        else:
            # qwen3 / llama style: cast x back then multiply by w (w retains its dtype).
            return self.weight * x_normed.to(input_dtype)

    def extra_repr(self) -> str:
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}, unit_offset={self.unit_offset}"


class RMSNormGated(nn.Module):
    """Gated RMSNorm used by GatedDeltaNet (port of Qwen3_5MoeRMSNormGated).

    Normalizes `hidden_states` then multiplies by SiLU(gate).
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
