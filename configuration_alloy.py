from __future__ import annotations

import json

from transformers.configuration_utils import PretrainedConfig


# Translation table from HuggingFace's canonical `layer_types` vocabulary
# (used inside qwen3 / qwen3.5 modeling code) to alloy's source-coupled
# registry keys. Lives here so both tests and examples can import it without
# reaching into private modules.
HF_LAYER_TYPE_TO_ALLOY: dict[str, str] = {
    "full_attention": "qwen3_attention",
    "sliding_attention": "qwen3_attention_sliding",
    "linear_attention": "qwen3_5_gdn",
}


def hf_layer_types_to_alloy(hf_layer_types) -> list[str]:
    """Translate an HF ``layer_types`` list into alloy's registry keys."""
    return [HF_LAYER_TYPE_TO_ALLOY.get(t, t) for t in hf_layer_types]


class AlloyConfig(PretrainedConfig):
    """Config for an Alloy transformer (mixed / hybrid token-mixer architectures).

    Ground truth for architecture:
      - ``layer_types[i]`` picks the token mixer at layer i (must match a registered mixer)
      - ``ffn_types[i]``   picks the feed-forward at layer i (must match a registered FFN)

    By default both are filled with their canonical defaults (``qwen3_attention`` and
    ``qwen3_mlp``) if omitted. Length must equal ``num_hidden_layers``.

    The config carries every hyperparameter needed by every registered module; each module
    only reads the fields it cares about. This matches HF's convention (one PretrainedConfig
    per model, read-what-you-need per submodule) and keeps ``config.json`` a single JSON blob.

    Field naming follows qwen3 / qwen3.5 so their checkpoints load without a state_dict rewrite.

    **Human-readable JSON output.**
    Hybrid-architecture configs are inherently field-heavy (one group of hyperparameters
    per token-mixer type, per FFN type, plus cross-cutting shape/norm/rotary settings).
    ``to_json_string`` reorders fields by owning module and inserts blank lines between
    groups so a reader of ``config.json`` sees visually separated blocks (attention,
    linear attention, MLP, MoE, ...) without any fake "section marker" keys cluttering
    the file. Blank lines inside a JSON object are valid whitespace, so round-trip
    through ``save_pretrained`` / ``from_pretrained`` is unchanged.

    **Runtime-only fields (not serialized).**
    Attributes whose name starts with a single underscore (e.g.
    ``_qwen3_5_gdn_implementation``) are treated as runtime fast-path
    selectors — they pick which registered implementation a module dispatches
    to (``"torch"`` / ``"triton"`` / ``"flash"`` / ``"npu_fused"`` / ...).
    The field name follows the mechanical rule
    ``_<module_key>_implementation`` so external fast-path packages can
    broadcast a backend across modules without a lookup table.

    These do not describe model architecture and are intentionally dropped by
    ``to_json_string`` so they never leak into the ``config.json`` published
    on the Hub. Set them post-construction:

    .. code-block:: python

        config = AlloyConfig(...)
        config._qwen3_5_gdn_implementation = "flash"   # picked up at module __init__
    """

    model_type = "alloy"
    keys_to_ignore_at_inference = ["past_key_values"]

    # --------------------------------------------------------------------- #
    # Visual grouping for human-readable config.json
    # --------------------------------------------------------------------- #
    # Ordered (field_names,) tuples. Anything not listed falls through into a
    # final "other" block at the bottom, so adding a new config field never
    # silently drops it from the JSON — it just lands in the last block until
    # the group table is updated.
    _CONFIG_GROUPS: tuple[tuple[str, ...], ...] = (
        # meta
        (
            "model_type", "architectures", "torch_dtype", "dtype", "transformers_version",
            "tie_word_embeddings", "use_cache",
            "pad_token_id", "bos_token_id", "eos_token_id", "keys_to_ignore_at_inference",
        ),
        # global shape
        (
            "vocab_size", "hidden_size", "num_hidden_layers",
            "max_position_embeddings", "initializer_range", "hidden_act",
        ),
        # architecture mix
        ("layer_types", "ffn_types"),
        # norm
        ("rms_norm_eps", "rms_norm_unit_offset"),
        # rotary
        ("rope_parameters",),
        # attention (Qwen3Attention: qwen3_attention / qwen3_attention_sliding)
        (
            "num_attention_heads", "num_key_value_heads", "head_dim",
            "attention_bias", "attention_dropout",
            "attn_output_gate", "sliding_window",
        ),
        # linear attention (Qwen35GatedDeltaNet -> qwen3_5_gdn)
        (
            "linear_num_key_heads", "linear_num_value_heads",
            "linear_key_head_dim", "linear_value_head_dim", "linear_conv_kernel_dim",
        ),
        # MLP (qwen3_mlp)
        ("intermediate_size",),
        # MoE (qwen3_5_moe)
        (
            "num_experts", "num_experts_per_tok",
            "moe_intermediate_size", "shared_expert_intermediate_size",
            "router_aux_loss_coef",
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
            layer_types = ["qwen3_attention"] * num_hidden_layers
        if ffn_types is None:
            ffn_types = ["qwen3_mlp"] * num_hidden_layers

        if len(layer_types) != num_hidden_layers:
            raise ValueError(
                f"len(layer_types)={len(layer_types)} must equal num_hidden_layers={num_hidden_layers}"
            )
        if len(ffn_types) != num_hidden_layers:
            raise ValueError(
                f"len(ffn_types)={len(ffn_types)} must equal num_hidden_layers={num_hidden_layers}"
            )

        # Cross-field validation that doesn't need layer_types / ffn_types
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                f"num_attention_heads ({self.num_attention_heads}) must be divisible by "
                f"num_key_value_heads ({self.num_key_value_heads})"
            )

        # PretrainedConfig runs a strict-dataclass validator on `layer_types`
        # against HF's canonical vocabulary (full_attention / sliding_attention /
        # ...). Alloy uses source-coupled keys (qwen3_attention / qwen3_5_gdn /
        # ...) that the HF validator rejects. We therefore defer the assignment
        # of `layer_types` / `ffn_types` until *after* the super init runs.
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

        self.layer_types = list(layer_types)
        self.ffn_types = list(ffn_types)

        if "qwen3_5_moe" in self.ffn_types and self.num_experts <= 0:
            raise ValueError("ffn_types contains 'qwen3_5_moe' but num_experts <= 0")

    # ------------------------------------------------------------------ #
    # Validator override: bypass HF's `validate_layer_type` strict check
    # ------------------------------------------------------------------ #
    def validate(self) -> None:
        """Run HF's PreTrainedConfig validators, but spoof ``layer_types`` past
        ``validate_layer_type``.

        HF's strict-dataclass on ``PreTrainedConfig`` ships a ``validate_layer_type``
        validator that requires every entry in ``self.layer_types`` to belong to
        a fixed canonical vocabulary (``full_attention``, ``sliding_attention``,
        ``linear_attention``, …). alloy uses source-coupled keys
        (``qwen3_attention``, ``qwen3_5_gdn``, …) that this validator rejects.

        We can't selectively disable a single strict-dataclass validator from a
        subclass, so we swap ``layer_types`` to a HF-canonical placeholder for
        the duration of the validator pass and restore afterwards. Every *other*
        validator HF contributes still runs against real values.

        Triggered by ``save_pretrained`` (which calls ``self.validate()``);
        ``__init__`` already avoids the issue by deferring the assignment of
        ``self.layer_types`` until after ``super().__init__()``.
        """
        saved = list(self.layer_types) if self.layer_types is not None else None
        try:
            if saved is not None:
                self.layer_types = ["full_attention"] * len(saved)
            super().validate()
        finally:
            if saved is not None:
                self.layer_types = saved

    # ------------------------------------------------------------------ #
    # Human-readable JSON serialization
    # ------------------------------------------------------------------ #
    def to_json_string(self, use_diff: bool = True) -> str:
        """Serialize to JSON with fields grouped by owning module.

        Reorders keys according to ``_CONFIG_GROUPS`` and post-processes the
        JSON text to insert a blank line between each group. Blank lines
        inside a JSON object are valid whitespace, so the output still parses
        and round-trips through ``save_pretrained`` / ``from_pretrained``
        unchanged.
        """
        base_dict = json.loads(super().to_json_string(use_diff=use_diff))

        # Drop runtime-only fields (leading-underscore attrs). These are
        # implementation hints (``_gdn_implementation`` etc.) that should
        # never travel with the model on disk — see class docstring.
        base_dict = {k: v for k, v in base_dict.items() if not k.startswith("_")}

        # Reorder: each group's fields, in order, then a final "leftover" block
        # for anything not covered by the group table.
        ordered: dict = {}
        first_of_group: list[str] = []
        seen: set[str] = set()
        for field_names in self._CONFIG_GROUPS:
            group = [(name, base_dict[name]) for name in field_names if name in base_dict]
            if not group:
                continue
            first_of_group.append(group[0][0])
            for name, value in group:
                ordered[name] = value
                seen.add(name)

        leftover = [(k, v) for k, v in base_dict.items() if k not in seen]
        if leftover:
            first_of_group.append(leftover[0][0])
            for k, v in leftover:
                ordered[k] = v

        raw = json.dumps(ordered, indent=2, sort_keys=False)

        # Insert a blank line before the first field of every group except the
        # very first. Match at the root indent level (2 spaces) so nested
        # object keys with the same name don't accidentally trigger.
        boundary_prefixes = tuple(f'  "{k}":' for k in first_of_group[1:])
        out_lines: list[str] = []
        for line in raw.split("\n"):
            if boundary_prefixes and line.startswith(boundary_prefixes):
                out_lines.append("")
            out_lines.append(line)
        return "\n".join(out_lines) + "\n"


__all__ = ["AlloyConfig", "HF_LAYER_TYPE_TO_ALLOY", "hf_layer_types_to_alloy"]
