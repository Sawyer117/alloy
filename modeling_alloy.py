from __future__ import annotations

import inspect
from typing import Callable, Optional, Union

import torch
import torch.nn as nn

from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel

from .configuration_alloy import AlloyConfig
from .modules.registry import get_ffn, get_mixer
from .modules.shared.rotary import DeepseekV4RotaryEmbedding, RotaryEmbedding


def _make_rotary(config: AlloyConfig) -> nn.Module:
    """Pick the rotary class that matches ``config.rope_parameters`` shape.

    qwen3 / qwen3.5 ship a flat ``rope_parameters`` dict
    (``{"rope_type": "default", "rope_theta": ...}``); the parametric
    :class:`RotaryEmbedding` handles those.

    DeepSeek-V4 ships a dict-of-dicts
    (``{"main": {...}, "compress": {...}}``) carrying TWO sets of rope
    parameters (different ``rope_theta`` for the standard attention path
    vs the HCA/CSA compressors). It needs
    :class:`DeepseekV4RotaryEmbedding`, which maintains a separate
    ``inv_freq`` buffer per rope-type key and takes a ``layer_type``
    kwarg at forward.

    Auto-detected from shape so we don't tie the dispatch to family
    naming — any future model with multi-rope-type config gets the
    right class automatically.
    """
    rope_params = config.rope_parameters or {}
    is_dict_of_dicts = any(isinstance(v, dict) for v in rope_params.values())
    if is_dict_of_dicts:
        return DeepseekV4RotaryEmbedding(config)
    return RotaryEmbedding(config=config)


def _call_rotary(
    rotary_emb: nn.Module,
    hidden_states: torch.Tensor,
    position_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Call the model-level rotary, passing ``layer_type="main"`` when the
    rotary class needs it (DSV4 family).

    Detected via the ``layer_types`` instance attribute that
    :class:`DeepseekV4RotaryEmbedding` sets in __init__ and the
    parametric :class:`RotaryEmbedding` doesn't have.
    """
    if hasattr(rotary_emb, "layer_types"):
        return rotary_emb(hidden_states, position_ids, layer_type="main")
    return rotary_emb(hidden_states, position_ids)
# RMSNorm is the only non-stdlib module modeling_alloy *constructs directly*
# (input_layernorm / post_attention_layernorm / final norm). All registered
# mixer / FFN classes are looked up by string via get_mixer / get_ffn at
# AlloyDecoderLayer construction time and ship their own ``init_weights`` —
# this file never knows their names.
from .modules.shared.norm import DeepseekV4UnweightedRMSNorm, RMSNorm


def _accepted_param_names(fn: Callable) -> frozenset[str]:
    """Return the set of explicit parameter names accepted by ``fn``.

    Used to call HF's ``create_causal_mask`` / ``create_sliding_window_causal_mask``
    portably across transformers versions: older releases take ``inputs_embeds``
    + ``past_key_values``, newer ones drop those in favour of ``cache_position``
    + per-batch shape kwargs. We pass a superset and filter at call time.
    """
    return frozenset(inspect.signature(fn).parameters)


_CAUSAL_MASK_ACCEPTED = _accepted_param_names(create_causal_mask)
_SLIDING_MASK_ACCEPTED = _accepted_param_names(create_sliding_window_causal_mask)


def _call_mask_builder(fn: Callable, accepted: frozenset[str], **kwargs):
    """Call an HF mask builder with only the kwargs its installed version accepts."""
    return fn(**{k: v for k, v in kwargs.items() if k in accepted})


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
            except (TypeError, ValueError):
                # TypeError: signature variant — has_previous_state(layer_idx).
                # ValueError: HF raises this when called on a Cache with no
                #   LinearAttention layers (cache_utils.py:1057). Happens when
                #   the cache was built without seeing canonical layer_types
                #   on the config (e.g. legacy alloy config.json with
                #   source-coupled keys still installed externally; or
                #   externally-instantiated DynamicCache without our config).
                # Fall back to seq_length, which is semantically equivalent
                # for the "have we run any forward yet" question this site
                # is asking.
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
        input_ids: torch.LongTensor | None = None,
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
        # input_ids is forwarded uniformly; FFN classes that don't need it
        # (Qwen3MLP, Qwen35SparseMoE) absorb it via **kwargs. Hash-routed
        # MoE variants (DSV4 dsv4_hash_moe) read it to compute the
        # tid2eid expert lookup.
        ffn_out = self.mlp(hidden_states, input_ids=input_ids)
        if isinstance(ffn_out, tuple):
            ffn_out = ffn_out[0]
        hidden_states = residual + ffn_out
        return hidden_states


# ---------------------------------------------------------------------------
# MHC (Manifold-Constrained Hyper-Connections, DSV4 paper §2.2)
# ---------------------------------------------------------------------------
# Toggled by ``config.use_mhc=True``. AlloyModel switches from
# :class:`AlloyDecoderLayer` (single-stream) to :class:`AlloyMhcDecoderLayer`
# (hc_mult parallel residual streams, mixed via HyperConnection between
# every sublayer + collapsed at the end via HyperHead before the final
# norm). Mixer / FFN modules see the same single-stream input as in the
# non-MHC path — MHC purely changes how residuals flow.
#
# Ported from references/dsv4/modeling_deepseek_v4.py:810-906
# (DeepseekV4HyperConnection + DeepseekV4HyperHead).


class _HyperConnection(nn.Module):
    r"""mHC mapping at one sublayer site.

    Takes ``hidden_streams`` of shape ``[B, S, hc_mult, D]``, returns
    ``(post, comb, collapsed)``:

      * ``collapsed`` ``[B, S, D]`` — single-stream input for the sublayer
        (attn / mlp), built as a learned weighted sum across the stream
        axis using the ``pre`` weights.
      * ``post`` ``[B, S, hc_mult]`` — per-stream weights for placing
        the sublayer's single-stream output back into the multi-stream
        residual.
      * ``comb`` ``[B, S, hc_mult, hc_mult]`` — Sinkhorn-Knopp projected
        doubly-stochastic mixing matrix that combines the existing
        streams.

    The decoder layer uses TWO instances of this — ``attn_hc`` for the
    attention site, ``ffn_hc`` for the MLP site.
    """

    def __init__(self, config: AlloyConfig) -> None:
        super().__init__()
        self.hc_mult = config.hc_mult
        self.hc_sinkhorn_iters = config.hc_sinkhorn_iters
        self.hc_eps = config.hc_eps
        self.input_norm = DeepseekV4UnweightedRMSNorm(eps=config.rms_norm_eps)
        # ``mix`` outputs cover (pre, post, comb) for hc_mult streams:
        #   pre  : hc_mult logits
        #   post : hc_mult logits
        #   comb : hc_mult x hc_mult logits
        # Total = (2 + hc_mult) * hc_mult.
        mix = (2 + self.hc_mult) * self.hc_mult
        self.fn = nn.Parameter(torch.empty(mix, self.hc_mult * config.hidden_size))
        self.base = nn.Parameter(torch.empty(mix))
        # Per-output-class scale: index 0 = pre, 1 = post, 2 = comb.
        self.scale = nn.Parameter(torch.empty(3))

    def forward(
        self, hidden_streams: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        flat = self.input_norm(hidden_streams.flatten(start_dim=2).float())
        mix = torch.nn.functional.linear(flat, self.fn.float())
        pre_scale, post_scale, comb_scale = self.scale.unbind(0)
        hc = self.hc_mult
        pre = torch.sigmoid(mix[..., :hc] * pre_scale + self.base[:hc]) + self.hc_eps
        post = torch.sigmoid(mix[..., hc: 2 * hc] * post_scale + self.base[hc: 2 * hc]) + self.hc_eps
        comb = (
            torch.sigmoid(
                mix[..., 2 * hc:].view(*mix.shape[:-1], hc, hc) * comb_scale
                + self.base[2 * hc:].view(hc, hc)
            )
            + self.hc_eps
        )
        # Sinkhorn-Knopp projection onto the doubly-stochastic manifold:
        # alternate row + column normalisation for ``hc_sinkhorn_iters`` steps.
        for _ in range(self.hc_sinkhorn_iters):
            comb = comb / (comb.sum(dim=-1, keepdim=True) + self.hc_eps)
            comb = comb / (comb.sum(dim=-2, keepdim=True) + self.hc_eps)
        # Collapse hc_mult parallel streams into a single sequence ready
        # for the sublayer (attn / mlp).
        collapsed = (pre.unsqueeze(-1) * hidden_streams).sum(dim=2).to(hidden_streams.dtype)
        return post, comb, collapsed


class _HyperHead(nn.Module):
    """Final HC-stream collapse before the shared RMSNorm.

    Reduces ``[B, S, hc_mult, D]`` back to ``[B, S, D]`` so the rest of the
    head (final norm + lm_head) stays single-stream. Lighter than
    :class:`_HyperConnection` — only computes the ``pre`` collapse weights,
    not the ``post`` / ``comb`` placement.
    """

    def __init__(self, config: AlloyConfig) -> None:
        super().__init__()
        self.hc_mult = config.hc_mult
        self.input_norm = DeepseekV4UnweightedRMSNorm(eps=config.rms_norm_eps)
        self.eps = config.hc_eps
        self.hc_fn = nn.Parameter(torch.empty(self.hc_mult, self.hc_mult * config.hidden_size))
        self.hc_base = nn.Parameter(torch.empty(self.hc_mult))
        self.hc_scale = nn.Parameter(torch.empty(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        flat = self.input_norm(x.flatten(2).float())
        mixes = torch.nn.functional.linear(flat, self.hc_fn.float())
        pre = torch.sigmoid(mixes * self.hc_scale.float() + self.hc_base.float()) + self.eps
        return (pre.unsqueeze(-1) * x).sum(dim=2).to(x.dtype)


class AlloyMhcDecoderLayer(nn.Module):
    """Decoder block with multi-stream residual flow (MHC variant).

    Same mixer / FFN dispatch as :class:`AlloyDecoderLayer` (so any
    registered mixer + ffn pair works behind MHC), but the residual
    machinery is replaced:

      * Input/output shape is ``[B, S, hc_mult, D]`` (not ``[B, S, D]``).
      * Two :class:`_HyperConnection` instances (``attn_hc``, ``ffn_hc``)
        collapse the streams down for the sublayer + place its output
        back across the streams.
      * Sublayer (attn / mlp) sees a single ``[B, S, D]`` tensor — it
        doesn't know MHC exists. This is what lets us A/B test MHC on
        any alloy config (qwen3 / qwen3.5 / dsv4) just by flipping
        ``config.use_mhc``.
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

        self.attn_hc = _HyperConnection(config)
        self.ffn_hc = _HyperConnection(config)

    @property
    def mixer(self) -> nn.Module:
        return getattr(self, self._mixer_attr)

    def forward(
        self,
        hidden_streams: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        input_ids: torch.LongTensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        dtype = hidden_streams.dtype

        # Attention site.
        post, comb, collapsed = self.attn_hc(hidden_streams)
        mixer_out = self.mixer(
            hidden_states=self.input_layernorm(collapsed),
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            **kwargs,
        )
        if isinstance(mixer_out, tuple):
            mixer_out = mixer_out[0]
        # Reblend single-stream sublayer output back into multi-stream:
        #   post[..., None] * mixer_out[..., None, :] + comb @ hidden_streams
        hidden_streams = (
            post.to(dtype).unsqueeze(-1) * mixer_out.unsqueeze(-2)
            + torch.matmul(comb.to(dtype), hidden_streams)
        )

        # MLP site (same shape of mix as attention).
        post, comb, collapsed = self.ffn_hc(hidden_streams)
        mlp_out = self.mlp(self.post_attention_layernorm(collapsed), input_ids=input_ids)
        if isinstance(mlp_out, tuple):
            mlp_out = mlp_out[0]
        hidden_streams = (
            post.to(dtype).unsqueeze(-1) * mlp_out.unsqueeze(-2)
            + torch.matmul(comb.to(dtype), hidden_streams)
        )
        return hidden_streams


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

    @classmethod
    def _can_set_experts_implementation(cls) -> bool:
        """Authoritatively report that alloy supports HF's MoE expert dispatch.

        HF's stock heuristic in ``PreTrainedModel._can_set_experts_implementation``
        opens the modeling file of ``cls`` and grep's for the literal string
        ``"@use_experts_implementation"``. alloy applies that decorator on
        ``_Experts`` inside ``alloy/modules/ffn/qwen3_5_moe.py``, not in this file, so
        the heuristic returns False and HF then forces ``_experts_implementation``
        from the default ``"grouped_mm"`` down to ``"eager"`` for us — while the
        upstream HF reference (whose modeling file *does* contain that decorator
        string) keeps ``"grouped_mm"`` and dispatches to
        ``grouped_mm_experts_forward``. The two paths are mathematically
        equivalent but compose ops differently; in fp32 they disagree by 1 ulp
        per MoE layer, which compounds across depth.

        Override the heuristic so alloy picks the same backend HF picks (= same
        shared function in ``transformers.integrations.moe``) and we get
        byte-exact match on any backend HF can dispatch to.
        """
        return True

    def get_correct_experts_implementation(self, experts_implementation):
        """Accept any backend registered in ``ALL_EXPERTS_FUNCTIONS``.

        Some older transformers forks (notably the bytedance internal fork
        circa 5.2.0.dev0) hardcoded a whitelist of allowed
        ``_experts_implementation`` values inside
        ``PreTrainedModel.get_correct_experts_implementation`` —
        ``["eager", "grouped_mm", "batched_mm"]``. They rejected any other
        value at construction time, even when the caller had explicitly
        registered a new entry into ``ALL_EXPERTS_FUNCTIONS`` (e.g.
        ``hf_npu_binder`` registering ``"flash"``).

        Upstream transformers v5.7+ already reads
        ``ALL_EXPERTS_FUNCTIONS.keys()`` dynamically, so its parent
        implementation handles registered backends correctly *and* preserves
        useful behaviour like the ``"grouped_mm" → "eager"`` fallback when
        ``_grouped_mm_can_dispatch`` rejects the hardware. We must NOT
        bypass that.

        Strategy: try parent first (gets all 5.7+ goodness), and only
        rescue with our registry check if parent raises a hardcoded-whitelist
        error. Names truly absent from ``ALL_EXPERTS_FUNCTIONS`` propagate
        the original error, so genuine typos like ``"grupedmm"`` still fail
        loudly.
        """
        try:
            return super().get_correct_experts_implementation(experts_implementation)
        except (ValueError, KeyError):
            try:
                from transformers.integrations.moe import ALL_EXPERTS_FUNCTIONS
            except ImportError:
                raise
            if experts_implementation in ALL_EXPERTS_FUNCTIONS:
                return experts_implementation
            raise

    def _init_weights(self, module: nn.Module) -> None:
        """Initialise ``module``'s direct Parameters.

        Dispatch order:

        1. **Per-class ``_alloy_init_weights(self, init_std)`` hook**, if
           defined on the module. This is the extension point — any
           registered mixer / FFN / shared-primitive class ships its own
           init logic here, so adding a new module type never requires
           editing ``modeling_alloy.py``. The ``_alloy_`` prefix avoids
           colliding with HF's ``PreTrainedModel.init_weights()`` (the
           recursive driver, takes only ``self``).

        2. **PyTorch stdlib types** (``nn.Linear`` / ``nn.Embedding`` /
           ``nn.Conv1d``): initialised in alloy's standard form using
           ``config.initializer_range``. We can't put a hook on these (we
           don't own their classes) so they live here as a fallback.

        Note that this method is called recursively by HF's
        ``PreTrainedModel.init_weights`` for every module in the tree, so
        each per-class hook only needs to handle *its own* direct
        Parameters — bare ``nn.Parameter`` attributes that aren't part of
        a child ``nn.Module``. Children are visited separately.
        """
        std = self.config.initializer_range

        init_hook = getattr(module, "_alloy_init_weights", None)
        if callable(init_hook):
            init_hook(std)
            return

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


class AlloyModel(AlloyPreTrainedModel):
    def __init__(self, config: AlloyConfig) -> None:
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        # MHC toggles which DecoderLayer class to use. The single-stream
        # AlloyDecoderLayer keeps the standard residual flow; the MHC variant
        # owns hc_mult parallel residual streams plus per-site HyperConnection
        # mixing. Both share the same mixer/ffn registry, so a model can have
        # any layer_types/ffn_types combo on either path.
        layer_cls = AlloyMhcDecoderLayer if getattr(config, "use_mhc", False) else AlloyDecoderLayer
        self.layers = nn.ModuleList(
            [layer_cls(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, unit_offset=config.rms_norm_unit_offset
        )
        self.rotary_emb = _make_rotary(config)
        # MHC head: collapses multi-stream back to single-stream just before
        # the final norm. Only built when use_mhc=True so non-MHC models
        # don't carry the parameter overhead.
        self.hc_head: _HyperHead | None = (
            _HyperHead(config) if getattr(config, "use_mhc", False) else None
        )
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
            # DynamicCache reads ``config.layer_types`` (cache_utils.py:1242)
            # to dispatch each layer to its right cache class — Attention vs
            # LinearAttention. alloy stores ``layer_types`` in source-coupled
            # form (``qwen3_5_gdn``); HF only recognises the canonical names
            # (``linear_attention``). Hand the cache the canonical view via
            # ``get_text_config`` so GDN layers get LinearAttentionLayer slots
            # and ``has_previous_state`` queries don't crash later.
            past_key_values = DynamicCache(config=self.config.get_text_config(decoder=True))

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
            # HF's mask builders changed signature across transformers versions
            # (older take inputs_embeds + past_key_values; newer prefer
            # cache_position + per-batch shape kwargs). We assemble a superset
            # and let _call_mask_builder filter to whatever the installed
            # version actually accepts.
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )
            mask_kwargs = {
                "config": self.config,
                "inputs_embeds": inputs_embeds,
                "input_embeds": inputs_embeds,  # alt spelling used by some versions
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
                "cache_position": cache_position,
                "batch_size": inputs_embeds.shape[0],
                "kv_length": past_seen_tokens + inputs_embeds.shape[1],
                "kv_offset": past_seen_tokens,
            }
            if self.has_causal_layers:
                mask_for_kind["causal"] = _call_mask_builder(
                    create_causal_mask, _CAUSAL_MASK_ACCEPTED, **mask_kwargs
                )
            if self.has_sliding_layers:
                mask_for_kind["sliding"] = _call_mask_builder(
                    create_sliding_window_causal_mask, _SLIDING_MASK_ACCEPTED, **mask_kwargs
                )
            if self.has_linear_layers:
                mask_for_kind["linear"] = _update_linear_attn_mask(attention_mask, past_key_values)

        hidden_states = inputs_embeds
        position_embeddings = _call_rotary(self.rotary_emb, hidden_states, position_ids)

        # MHC mode: expand the embedding output into hc_mult parallel
        # residual streams. AlloyMhcDecoderLayer takes [B, S, hc_mult, D]
        # in/out, mixing the streams via HyperConnection at every sublayer
        # site. Non-MHC mode leaves hidden_states as [B, S, D].
        if self.hc_head is not None:
            hidden_states = (
                hidden_states.unsqueeze(2)
                .expand(-1, -1, self.config.hc_mult, -1)
                .contiguous()
            )

        for i, layer in enumerate(self.layers):
            layer_mask = mask_for_kind.get(self._mask_kinds[i])
            hidden_states = layer(
                hidden_states,
                attention_mask=layer_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                position_embeddings=position_embeddings,
                use_cache=use_cache,
                # input_ids is needed by hash-routed MoE FFNs (DSV4
                # dsv4_hash_moe). Other FFN variants ignore it via
                # **kwargs at the FFN forward.
                input_ids=input_ids,
            )

        # Collapse hc_mult streams back to [B, S, D] before the final norm.
        if self.hc_head is not None:
            hidden_states = self.hc_head(hidden_states)

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

        # Chunk-loss path (mindspeed-mm FSDP2 with enable_chunk_loss=true): the trainer
        # monkey-patches lm_head.forward to take (hidden_states, loss_func) and return the
        # loss directly, then sets enable_chunk_loss=True on the model. We must route to
        # the new lm_head signature instead of computing logits + loss separately —
        # otherwise the patched forward errors on the missing positional argument.
        if getattr(self, "enable_chunk_loss", False):
            logits = None
            loss = self.lm_head(hidden_states[:, slice_indices, :], self.loss_function)
        else:
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
