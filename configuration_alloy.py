from __future__ import annotations

import json
import re

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

    **Human-readable JSON output.**
    Hybrid-architecture configs are inherently field-heavy (one group of hyperparameters
    per token-mixer type, per FFN type, plus cross-cutting shape/norm/rotary settings).
    ``to_json_string`` reorders and groups fields by owning module and inserts separator
    keys so a reader of ``config.json`` can see at a glance which parameters belong to
    which subsystem. The separator keys (prefixed by ``_section_``) round-trip cleanly —
    ``__init__`` drops them on load.
    """

    model_type = "alloy"
    keys_to_ignore_at_inference = ["past_key_values"]

    # --------------------------------------------------------------------- #
    # Visual grouping for human-readable config.json
    # --------------------------------------------------------------------- #
    _SECTION_MARKER_PREFIX = "_section_"

    # Ordered (header, field_names) tuples. Anything not listed falls through
    # into an "Other" group at the bottom, so adding a new config field never
    # silently drops it from the JSON — it just lands in "Other" until the
    # group table is updated.
    _CONFIG_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
        (
            "Meta",
            (
                "model_type",
                "architectures",
                "torch_dtype",
                "dtype",
                "transformers_version",
                "tie_word_embeddings",
                "use_cache",
                "pad_token_id",
                "bos_token_id",
                "eos_token_id",
                "keys_to_ignore_at_inference",
            ),
        ),
        (
            "Global shape (cross-layer)",
            (
                "vocab_size",
                "hidden_size",
                "num_hidden_layers",
                "max_position_embeddings",
                "initializer_range",
                "hidden_act",
            ),
        ),
        (
            "Architecture mix (one entry per layer)",
            ("layer_types", "ffn_types"),
        ),
        (
            "Norm (RMSNorm, shared)",
            ("rms_norm_eps", "rms_norm_unit_offset"),
        ),
        (
            "Rotary (RoPE, shared across attention layers)",
            ("rope_parameters",),
        ),
        (
            "Attention --- full_attention / sliding_attention (GQAAttention)",
            (
                "num_attention_heads",
                "num_key_value_heads",
                "head_dim",
                "attention_bias",
                "attention_dropout",
                "attn_output_gate",
                "sliding_window",
            ),
        ),
        (
            "Linear attention --- linear_attention (GatedDeltaNet)",
            (
                "linear_num_key_heads",
                "linear_num_value_heads",
                "linear_key_head_dim",
                "linear_value_head_dim",
                "linear_conv_kernel_dim",
            ),
        ),
        (
            "Dense FFN --- mlp (SwiGLUMLP)",
            ("intermediate_size",),
        ),
        (
            "Sparse MoE --- moe (SparseMoEBlock)",
            (
                "num_experts",
                "num_experts_per_tok",
                "moe_intermediate_size",
                "shared_expert_intermediate_size",
                "router_aux_loss_coef",
            ),
        ),
    )

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
        # Drop visual section markers that round-trip through saved config.json
        for k in [key for key in kwargs if key.startswith(self._SECTION_MARKER_PREFIX)]:
            kwargs.pop(k)

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

    # ------------------------------------------------------------------ #
    # Human-readable JSON serialization
    # ------------------------------------------------------------------ #
    @classmethod
    def _slug(cls, header: str) -> str:
        return re.sub(r"[^a-z0-9]+", "_", header.lower()).strip("_")

    def to_json_string(self, use_diff: bool = True) -> str:
        """Serialize to JSON with fields grouped by owning module.

        Produces the same information content as the default
        ``PretrainedConfig.to_json_string`` but reorders keys into the module
        groups defined by ``_CONFIG_GROUPS`` and inserts ``_section_*`` marker
        keys between groups. The marker keys are dropped on load by
        ``__init__``, so ``save_pretrained`` / ``from_pretrained`` round-trips.
        """
        base_dict = json.loads(super().to_json_string(use_diff=use_diff))
        ordered: dict = {}
        seen: set[str] = set()
        for idx, (header, field_names) in enumerate(self._CONFIG_GROUPS):
            group = [(name, base_dict[name]) for name in field_names if name in base_dict]
            if not group:
                continue
            marker_key = f"{self._SECTION_MARKER_PREFIX}{idx:02d}_{self._slug(header)}"
            ordered[marker_key] = f"===== {header} ====="
            for name, value in group:
                ordered[name] = value
                seen.add(name)

        leftover = [(k, v) for k, v in base_dict.items() if k not in seen]
        if leftover:
            ordered[f"{self._SECTION_MARKER_PREFIX}99_other"] = "===== Other (ungrouped) ====="
            for k, v in leftover:
                ordered[k] = v

        return json.dumps(ordered, indent=2, sort_keys=False) + "\n"


__all__ = ["AlloyConfig"]
