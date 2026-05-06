from .norm import (
    Qwen3RMSNorm,
    Qwen35RMSNorm,
    Qwen35RMSNormGated,
    # Backward-compat names: RMSNorm is a factory function, RMSNormGated
    # aliases Qwen35RMSNormGated. New code should use the source-coupled
    # names above.
    RMSNorm,
    RMSNormGated,
)
from .rotary import (
    RotaryEmbedding,
    Qwen3RotaryEmbedding,
    Qwen35RotaryEmbedding,
    apply_rotary_pos_emb,
    rotate_half,
)
from .attention_kernels import eager_attention_forward, repeat_kv

__all__ = [
    # norm
    "Qwen3RMSNorm",
    "Qwen35RMSNorm",
    "Qwen35RMSNormGated",
    "RMSNorm",
    "RMSNormGated",
    # rotary
    "RotaryEmbedding",
    "Qwen3RotaryEmbedding",
    "Qwen35RotaryEmbedding",
    "apply_rotary_pos_emb",
    "rotate_half",
    # attention kernels
    "eager_attention_forward",
    "repeat_kv",
]
