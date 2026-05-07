from . import qwen3_attention  # noqa: F401  registers "qwen3_attention" and "qwen3_attention_sliding"
from . import qwen3_5_gdn  # noqa: F401  registers "qwen3_5_gdn"
# dsv4_attention import has TWO side effects: (1) registers
# "dsv4_sliding_attention" / "dsv4_hca_attention" / "dsv4_csa_attention" mixers,
# (2) auto-registers DeepseekV4HCACache / DeepseekV4CSACache into HF's
# LAYER_TYPE_CACHE_MAPPING via CacheLayerMixin.__init_subclass__.
from . import dsv4_attention  # noqa: F401
