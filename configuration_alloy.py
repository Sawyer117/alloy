from __future__ import annotations

from transformers.configuration_utils import PretrainedConfig


class AlloyConfig(PretrainedConfig):
    """Config for an Alloy transformer (mixed / hybrid token-mixer architectures).

    Ground truth for architecture:
      - ``layer_types[i]`` picks the token mixer at layer i (must match a registered mixer)
      - ``ffn_types[i]``   picks the feed-forward at layer i (must match a registered FFN)

    By default both are filled with their canonical defaults (``full_attention`` and ``mlp``)
    if omitted. Length must equal ``num_hidden_layers``.

    The config carries every hyperparameter needed by every registered module; each module
    only reads the fields it cares about. This matches HF's convention (one PretrainedConfig
    per model, read-what-you-need per submodule) and keeps ``config.json`` a single JSON blob.

    Field naming follows qwen3 / qwen3.5 so their checkpoints load without a state_dict rewrite.
    """

    model_type = "alloy"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        # embeddings / general
        vocab_size: int = 151936,
        hidden_size: int = 2048,
        num_hidden_layers: int = 4,
        max_position_embeddings: int = 32768,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-6,
        rms_norm_unit_offset: bool = False,
        use_cache: bool = True,
        tie_word_embeddings: bool = False,
        hidden_act: str = "silu",
        pad_token_id: int | None = None,
        bos_token_id: int | None = None,
        eos_token_id: int | list[int] | None = None,
        # per-layer architecture
        layer_types: list[str] | None = None,
        ffn_types: list[str] | None = None,
        # rotary
        rope_parameters: dict | None = None,
        # GQA
        num_attention_heads: int = 16,
        num_key_value_heads: int = 2,
        head_dim: int | None = None,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        attn_output_gate: bool = False,
        sliding_window: int | None = None,
        # GDN (linear attention)
        linear_num_key_heads: int = 16,
        linear_num_value_heads: int = 32,
        linear_key_head_dim: int = 128,
        linear_value_head_dim: int = 128,
        linear_conv_kernel_dim: int = 4,
        # dense FFN (MLP)
        intermediate_size: int = 8192,
        # MoE
        num_experts: int = 0,
        num_experts_per_tok: int = 0,
        moe_intermediate_size: int = 512,
        shared_expert_intermediate_size: int = 512,
        router_aux_loss_coef: float = 0.001,
        **kwargs,
    ) -> None:
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.rms_norm_unit_offset = rms_norm_unit_offset
        self.use_cache = use_cache
        self.hidden_act = hidden_act

        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim if head_dim is not None else (hidden_size // num_attention_heads)
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.attn_output_gate = attn_output_gate
        self.sliding_window = sliding_window

        self.linear_num_key_heads = linear_num_key_heads
        self.linear_num_value_heads = linear_num_value_heads
        self.linear_key_head_dim = linear_key_head_dim
        self.linear_value_head_dim = linear_value_head_dim
        self.linear_conv_kernel_dim = linear_conv_kernel_dim

        self.intermediate_size = intermediate_size

        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_intermediate_size = moe_intermediate_size
        self.shared_expert_intermediate_size = shared_expert_intermediate_size
        self.router_aux_loss_coef = router_aux_loss_coef

        if rope_parameters is None:
            rope_parameters = {"rope_type": "default", "rope_theta": 10000.0}
        self.rope_parameters = rope_parameters

        if layer_types is None:
            layer_types = ["full_attention"] * num_hidden_layers
        if ffn_types is None:
            ffn_types = ["mlp"] * num_hidden_layers

        if len(layer_types) != num_hidden_layers:
            raise ValueError(
                f"len(layer_types)={len(layer_types)} must equal num_hidden_layers={num_hidden_layers}"
            )
        if len(ffn_types) != num_hidden_layers:
            raise ValueError(
                f"len(ffn_types)={len(ffn_types)} must equal num_hidden_layers={num_hidden_layers}"
            )

        self.layer_types = list(layer_types)
        self.ffn_types = list(ffn_types)

        # Cross-field validation
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                f"num_attention_heads ({self.num_attention_heads}) must be divisible by "
                f"num_key_value_heads ({self.num_key_value_heads})"
            )
        if "moe" in self.ffn_types and self.num_experts <= 0:
            raise ValueError("ffn_types contains 'moe' but num_experts <= 0")

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


__all__ = ["AlloyConfig"]
