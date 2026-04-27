from __future__ import annotations

from typing import Optional, Union

import torch
import torch.nn as nn

from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel

from .configuration_alloy import AlloyConfig
from .modules.attention.gdn import Qwen35GatedDeltaNet
from .modules.ffn.mlp import Qwen3MLP  # noqa: F401  used for typing / external imports
from .modules.ffn.moe import Qwen35SparseMoE, _Experts, _TopKRouter  # noqa: F401
from .modules.registry import get_ffn, get_mixer
from .modules.shared.norm import RMSNorm
from .modules.shared.rotary import RotaryEmbedding


def _update_linear_attn_mask(
    attention_mask: torch.Tensor | None,
    past_key_values: Cache | None,
) -> torch.Tensor | None:
    """Match qwen3_5_moe's ``_update_linear_attn_mask``: return None when either
    the cache has prior state (no need to zero padding again) or the mask is all-ones.
    Otherwise return the 2D mask unchanged for use inside Qwen35GatedDeltaNet.
    """
    has_prev = False
    if past_key_values is not None:
        fn = getattr(past_key_values, "has_previous_state", None)
        if callable(fn):
            try:
                has_prev = bool(fn())
            except TypeError:
                # DynamicCache exposes has_previous_state(layer_idx); treat len>0 as "prior"
                has_prev = past_key_values.get_seq_length() > 0
        else:
            has_prev = past_key_values.get_seq_length() > 0
    if has_prev:
        return None
    if attention_mask is not None and torch.all(attention_mask == 1):
        return None
    return attention_mask


# ---------------------------------------------------------------------------
# Decoder layer
# ---------------------------------------------------------------------------


class AlloyDecoderLayer(nn.Module):
    """A transformer block whose token mixer and FFN are chosen via registry.

    Sub-module attribute names match HF state_dict expectations:
      - mixer: ``self_attn`` (for full/sliding attention) or ``linear_attn`` (for linear attention)
      - FFN:   always ``self.mlp`` (covers both dense MLP and SparseMoE)
    """

    def __init__(self, config: AlloyConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.layer_type = config.layer_types[layer_idx]
        self.ffn_type = config.ffn_types[layer_idx]

        mixer_entry = get_mixer(self.layer_type)
        self._mixer_attr = mixer_entry.attr_name
        setattr(self, self._mixer_attr, mixer_entry.cls(config, layer_idx))

        ffn_cls = get_ffn(self.ffn_type)
        self.mlp = ffn_cls(config, layer_idx)

        self.input_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, unit_offset=config.rms_norm_unit_offset
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, unit_offset=config.rms_norm_unit_offset
        )

    @property
    def mixer(self) -> nn.Module:
        return getattr(self, self._mixer_attr)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        mixer_out = self.mixer(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            **kwargs,
        )
        if isinstance(mixer_out, tuple):
            mixer_out = mixer_out[0]
        hidden_states = residual + mixer_out

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        ffn_out = self.mlp(hidden_states)
        if isinstance(ffn_out, tuple):
            ffn_out = ffn_out[0]
        hidden_states = residual + ffn_out
        return hidden_states


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class AlloyPreTrainedModel(PreTrainedModel):
    config_class = AlloyConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["AlloyDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _is_stateful = True

    def _init_weights(self, module: nn.Module) -> None:
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.Conv1d):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, RMSNorm):
            if module.unit_offset:
                nn.init.zeros_(module.weight)
            else:
                nn.init.ones_(module.weight)
        elif isinstance(module, Qwen35GatedDeltaNet):
            nn.init.ones_(module.dt_bias)
            with torch.no_grad():
                module.A_log.copy_(torch.empty_like(module.A_log).uniform_(0, 16).log_())
        elif isinstance(module, _Experts):
            nn.init.normal_(module.gate_up_proj, mean=0.0, std=std)
            nn.init.normal_(module.down_proj, mean=0.0, std=std)
        elif isinstance(module, _TopKRouter):
            nn.init.normal_(module.weight, mean=0.0, std=std)


class AlloyModel(AlloyPreTrainedModel):
    def __init__(self, config: AlloyConfig) -> None:
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [AlloyDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, unit_offset=config.rms_norm_unit_offset
        )
        self.rotary_emb = RotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        self._mask_kinds = tuple(get_mixer(name).mask_kind for name in config.layer_types)
        self.has_causal_layers = "causal" in self._mask_kinds
        self.has_sliding_layers = "sliding" in self._mask_kinds
        self.has_linear_layers = "linear" in self._mask_kinds

        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.embed_tokens

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0

        if position_ids is None:
            position_ids = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            ).unsqueeze(0)

        # Build one mask per mask_kind once via HF utilities; layers index by mask_kind.
        mask_for_kind: dict[str, torch.Tensor | None] = {}
        if isinstance(attention_mask, dict):
            # Already prepared (e.g. by generate()); pass through.
            mask_for_kind = attention_mask
        else:
            mask_kwargs = {
                "config": self.config,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            if self.has_causal_layers:
                mask_for_kind["causal"] = create_causal_mask(**mask_kwargs)
            if self.has_sliding_layers:
                mask_for_kind["sliding"] = create_sliding_window_causal_mask(**mask_kwargs)
            if self.has_linear_layers:
                mask_for_kind["linear"] = _update_linear_attn_mask(attention_mask, past_key_values)

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for i, layer in enumerate(self.layers):
            layer_mask = mask_for_kind.get(self._mask_kinds[i])
            hidden_states = layer(
                hidden_states,
                attention_mask=layer_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                position_embeddings=position_embeddings,
                use_cache=use_cache,
            )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )


# ---------------------------------------------------------------------------
# Causal LM head
# ---------------------------------------------------------------------------


class AlloyForCausalLM(AlloyPreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config: AlloyConfig) -> None:
        super().__init__(config)
        self.model = AlloyModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.model.embed_tokens

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.model.embed_tokens = value

    def get_output_embeddings(self) -> nn.Module:
        return self.lm_head

    def set_output_embeddings(self, new_embeddings: nn.Module) -> None:
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )
        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss: Optional[torch.Tensor] = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=None,
            attentions=None,
        )


__all__ = [
    "AlloyPreTrainedModel",
    "AlloyModel",
    "AlloyForCausalLM",
]
