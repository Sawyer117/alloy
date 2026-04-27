from __future__ import annotations

import torch
from torch import nn

from transformers.activations import ACT2FN

from ..registry import register_ffn


@register_ffn("qwen3_mlp")
class Qwen3MLP(nn.Module):
    """SwiGLU feed-forward block ported from qwen3 / qwen3.5.

    Parameter names (gate_proj / up_proj / down_proj) match HF checkpoints.
    The intermediate size is read from ``config.intermediate_size``.
    """

    def __init__(self, config, layer_idx: int | None = None) -> None:
        super().__init__()
        del layer_idx  # not needed for dense MLP
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
