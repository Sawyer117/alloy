from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch.nn as nn


@dataclass(frozen=True)
class MixerEntry:
    cls: type[nn.Module]
    attr_name: str  # name under which the module is stored on the DecoderLayer (matches HF state_dict)


MIXER_REGISTRY: dict[str, MixerEntry] = {}
FFN_REGISTRY: dict[str, type[nn.Module]] = {}


def register_mixer(name: str, attr_name: str) -> Callable[[type[nn.Module]], type[nn.Module]]:
    """
    Register a token-mixer (attention-like) module under a string key.

    `attr_name` determines the attribute on the DecoderLayer where the module is stored.
    For HF state_dict compatibility use:
      - "self_attn"   for full_attention / sliding_attention / gated GQA
      - "linear_attn" for linear_attention (e.g. GatedDeltaNet)
    """

    def _wrap(cls: type[nn.Module]) -> type[nn.Module]:
        if name in MIXER_REGISTRY:
            raise ValueError(f"Mixer '{name}' already registered")
        MIXER_REGISTRY[name] = MixerEntry(cls=cls, attr_name=attr_name)
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
