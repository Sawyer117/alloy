from .norm import (
    Qwen3RMSNorm,
    Qwen35RMSNorm,
    Qwen35RMSNormGated,
    DeepseekV4RMSNorm,
    DeepseekV4UnweightedRMSNorm,
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
    DeepseekV4RotaryEmbedding,
    apply_rotary_pos_emb,
    apply_rotary_pos_emb_interleaved,
    rotate_half,
    rotate_half_interleaved,
)
from .attention_kernels import eager_attention_forward, repeat_kv

__all__ = [
    # norm
    "Qwen3RMSNorm",
    "Qwen35RMSNorm",
    "Qwen35RMSNormGated",
    "DeepseekV4RMSNorm",
    "DeepseekV4UnweightedRMSNorm",
    "RMSNorm",
    "RMSNormGated",
    # rotary
    "RotaryEmbedding",
    "Qwen3RotaryEmbedding",
    "Qwen35RotaryEmbedding",
    "DeepseekV4RotaryEmbedding",
    "apply_rotary_pos_emb",
    "apply_rotary_pos_emb_interleaved",
    "rotate_half",
    "rotate_half_interleaved",
    # attention kernels
    "eager_attention_forward",
    "repeat_kv",
]
