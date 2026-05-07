"""DeepSeek-V4 attention port — 3 mixer flavors + 2 cache layers + helpers.

Three layer types share one :class:`DeepseekV4Attention` class — `__init__`
builds the right compressor (or none) based on ``config.layer_types[layer_idx]``:

  * ``dsv4_sliding_attention`` — pure sliding-window attention, no compressor
  * ``dsv4_hca_attention``     — sliding KV + Heavily Compressed Attention
                                 (one compressed entry every 128 tokens,
                                 non-overlapping windows)
  * ``dsv4_csa_attention``     — sliding KV + Compressed Sparse Attention
                                 (overlapping 4-token windows + Lightning
                                 Indexer top-k gating)

Per the project convention, cache layer classes live in this file too —
:class:`DeepseekV4HCACache` and :class:`DeepseekV4CSACache` self-register
into HF's ``LAYER_TYPE_CACHE_MAPPING`` via the ``layer_type = "..."`` class
attribute (HF added this auto-registration in cache_utils.py:31-50, the
version that ships DSV4). DynamicCache picks them up automatically when
``config.get_text_config().layer_types`` carries the corresponding
canonical strings.

Source provenance: ported from ``references/dsv4/modeling_deepseek_v4.py``
lines 171-807. Class structure preserved 1:1 to keep upstream re-syncs
mechanical. Style adjustments: type hints, docstrings.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from transformers.cache_utils import Cache, DynamicSlidingWindowLayer
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from ..registry import register_mixer
from ..shared.attention_kernels import repeat_kv
from ..shared.norm import DeepseekV4RMSNorm, DeepseekV4UnweightedRMSNorm
from ..shared.rotary import (
    DeepseekV4RotaryEmbedding,
    apply_rotary_pos_emb_interleaved as apply_rotary_pos_emb,
)


# =========================================================================== #
# Cache layers (auto-registered via class attribute ``layer_type = "..."``)
# =========================================================================== #


class DeepseekV4HCACache(DynamicSlidingWindowLayer):
    r"""Cache layer for HCA blocks (paper §2.3.2).

    Holds the long-range compressor's buffer / running compressed entries /
    count on top of the sliding-window K=V branch. HCA uses *non-overlapping*
    windows, so there is no overlap state and no indexer.

    State is dict-keyed by entry name — HCA only uses ``"compressor"``;
    :class:`DeepseekV4CSACache` adds ``"indexer"`` to the same dicts so a
    single set of methods serves both:

    * ``compressed_kv[name]`` — the running list of compressed KV entries
      emitted so far (one every ``compress_rate`` source tokens; the
      long-range KVs the attention concatenates onto its sliding-window
      keys / values).
    * ``buffer_kv[name]`` / ``buffer_gate[name]`` — source tokens that
      arrived between two full windows; once the buffer hits
      ``compress_rate`` tokens the compressor closes a window, emits one
      entry, and drains the buffer.
    * ``entry_count[name]`` — number of compressed entries emitted so far,
      so ``entry_count[name] * compress_rate`` is the absolute position
      of the *next* window's first source token. Tracked separately from
      ``position_ids`` so prefill -> decode -> prefill stays consistent.

    HF auto-registers this class into ``LAYER_TYPE_CACHE_MAPPING`` via
    the ``layer_type = "heavily_compressed_attention"`` class attribute —
    DynamicCache then picks it up when ``config.get_text_config().layer_types``
    contains that canonical string (alloy translates from ``dsv4_hca_attention``
    via :func:`alloy.configuration_alloy.alloy_layer_types_to_hf`).
    """

    layer_type = "heavily_compressed_attention"

    def __init__(self, config) -> None:
        super().__init__(config)
        self.compress_rate = config.compress_rates["heavily_compressed_attention"]
        self.buffer_kv: dict[str, torch.Tensor | None] = {"compressor": None}
        self.buffer_gate: dict[str, torch.Tensor | None] = {"compressor": None}
        self.compressed_kv: dict[str, torch.Tensor | None] = {"compressor": None}
        self.entry_count: dict[str, int] = {"compressor": 0}

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, *args, **kwargs):
        """Sliding-window K=V update. DSV4 uses shared-KV MQA so ``keys`` and
        ``values`` point to the same storage on every layer.
        """
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)
            self.values = self.keys
        self.cumulative_length += key_states.shape[-2]
        full = torch.cat([self.keys, key_states], dim=-2)
        self.keys = full[:, :, -self.sliding_window + 1:, :]
        self.values = self.keys
        return full, full

    def store_compression_weights(
        self, name: str, kv: torch.Tensor, gate: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Concatenate new ``(kv, gate)`` for entry ``name`` with the buffer,
        peel off the longest window-aligned prefix as the chunk ready to
        compress, keep the leftover in the buffer for next call. Returns
        ``(chunk_kv, chunk_gate, first_window_position)``.
        """
        first_window_position = self.entry_count[name] * self.compress_rate
        buffered_kv, buffered_gate = self.buffer_kv[name], self.buffer_gate[name]
        if buffered_kv is not None and buffered_kv.shape[1]:
            kv = torch.cat([buffered_kv, kv], dim=1)
            gate = torch.cat([buffered_gate, gate], dim=1)
        usable = (kv.shape[1] // self.compress_rate) * self.compress_rate
        self.buffer_kv[name], self.buffer_gate[name] = kv[:, usable:], gate[:, usable:]
        return kv[:, :usable], gate[:, :usable], first_window_position

    def update_compressor_states(self, name: str, compressed: torch.Tensor) -> torch.Tensor:
        """Append freshly emitted compressed entries, bump entry_count, return
        running ``compressed_kv[name]``.
        """
        if self.compressed_kv[name] is None:
            self.compressed_kv[name] = compressed
        elif compressed.shape[1] > 0:
            self.compressed_kv[name] = torch.cat([self.compressed_kv[name], compressed], dim=1)
        self.entry_count[name] += compressed.shape[1]
        return self.compressed_kv[name]


class DeepseekV4CSACache(DeepseekV4HCACache):
    r"""Cache layer for CSA blocks (paper §2.3.1).

    Extends :class:`DeepseekV4HCACache` by adding ``"indexer"`` entry to the
    inherited dicts (the CSA block has both a main compressor and an
    indexer with their own running state), plus per-name *overlap state*
    for the two-series window scheme.

    The CSA ``kv_proj`` / ``gate_proj`` produce ``2 * head_dim`` features
    per token — two independent compressed series Ca and Cb stored in one
    tensor. Pooled entry ``w`` is the softmax-gated convex combination of
    window ``w-1``'s Ca slice with window ``w``'s Cb slice — effective
    width ``2 * compress_rate_csa``, stride ``compress_rate_csa``.

    Adjacent windows share state only through the previous window's Ca
    slice, so the only thing carried across forward boundaries is
    ``chunk[:, -1, :, :head_dim]`` of the last full window — that's what
    ``overlap_kv[name]`` / ``overlap_gate[name]`` persist.
    """

    layer_type = "compressed_sparse_attention"

    def __init__(self, config) -> None:
        super().__init__(config)
        self.compress_rate = config.compress_rates["compressed_sparse_attention"]
        self.buffer_kv["indexer"] = None
        self.buffer_gate["indexer"] = None
        self.compressed_kv["indexer"] = None
        self.entry_count["indexer"] = 0
        self.overlap_kv: dict[str, torch.Tensor | None] = {"compressor": None, "indexer": None}
        self.overlap_gate: dict[str, torch.Tensor | None] = {"compressor": None, "indexer": None}

    def update_overlap_state(
        self, name: str, chunk_kv: torch.Tensor, chunk_gate: torch.Tensor, head_dim: int
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Read the prior window's Ca slice (saved on the previous forward),
        persist the current call's last-window Ca slice for the next call.
        Only the ``:head_dim`` slice (Ca) is consumed downstream — Cb is
        already folded into the previous window's emitted entry. Returns
        ``(prior_kv, prior_gate)`` — both ``None`` on the very first call.
        """
        prior_kv, prior_gate = self.overlap_kv[name], self.overlap_gate[name]
        self.overlap_kv[name] = chunk_kv[:, -1, :, :head_dim].clone()
        self.overlap_gate[name] = chunk_gate[:, -1, :, :head_dim].clone()
        return prior_kv, prior_gate


# =========================================================================== #
# Helpers
# =========================================================================== #


class DeepseekV4GroupedLinear(nn.Linear):
    """Block-diagonal grouped linear used by the grouped output projection.

    The core attention's stacked output is ``num_attention_heads * head_dim``
    dim, which is very large (V4-Flash: 32768; V4-Pro: 65536). A direct
    projection to ``hidden_size`` would dominate per-token cost. The paper
    splits heads into ``g`` groups, projects each
    ``num_attention_heads * head_dim / g``-dim group independently to a
    ``d_g``-dim intermediate (with ``d_g < num_attention_heads * head_dim / g``),
    then mixes the resulting ``g * d_g`` vector to ``hidden_size`` through a
    follow-up linear (``self_attn.o_b_proj``). This module owns the per-group
    block (``self_attn.o_a_proj``).
    """

    def __init__(self, in_features_per_group: int, out_features: int, n_groups: int, bias: bool = False) -> None:
        super().__init__(in_features_per_group, out_features, bias=bias)
        self.n_groups = n_groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_shape = x.shape[:-2]
        hidden_dim = x.shape[-1]
        w = self.weight.view(self.n_groups, -1, hidden_dim).transpose(1, 2)
        x = x.reshape(-1, self.n_groups, hidden_dim).transpose(0, 1)
        y = torch.bmm(x, w).transpose(0, 1)
        return y.reshape(*input_shape, self.n_groups, -1)


# =========================================================================== #
# Compressors + Indexer
# =========================================================================== #


class DeepseekV4HCACompressor(nn.Module):
    """Heavily Compressed Attention compressor (paper §2.3.2, eqs. 20-23).

    Compresses every ``compress_rate_hca`` (m'=128) source tokens into a
    single compressed KV entry: ``C^Comp_i = Σ_{j∈window} softmax(Z_j + B)_j ⊙ C_j``.
    RoPE on the trailing ``rope_head_dim`` slice is applied at the absolute
    window position (``i * compress_rate_hca + first_window_position``) so
    cross-call concatenation stays causality-correct.

    Returns the running list of *all* compressed entries emitted so far
    (shape ``[B, 1, T, head_dim]``), so the attention can attend over the
    full long-range history.

    Stateless mode (``past_key_values is None``): compress every complete
    window from ``hidden_states`` and discard the remainder.
    """

    rope_layer_type = "compress"

    def __init__(self, config) -> None:
        super().__init__()
        self.compress_rate = config.compress_rates["heavily_compressed_attention"]
        self.head_dim = config.head_dim
        self.kv_proj = nn.Linear(config.hidden_size, self.head_dim, bias=False)
        self.gate_proj = nn.Linear(config.hidden_size, self.head_dim, bias=False)
        self.position_bias = nn.Parameter(torch.empty(self.compress_rate, self.head_dim))
        self.kv_norm = DeepseekV4RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.rotary_emb = DeepseekV4RotaryEmbedding(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        q_residual: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values: Cache | None,
        layer_idx: int,
    ) -> torch.Tensor:
        batch, _, _ = hidden_states.shape
        cache_layer: DeepseekV4HCACache = (
            past_key_values.layers[layer_idx] if past_key_values is not None else None
        )
        kv = self.kv_proj(hidden_states)
        gate = self.gate_proj(hidden_states)
        if cache_layer is None:
            usable = (kv.shape[1] // self.compress_rate) * self.compress_rate
            chunk_kv, chunk_gate, first_window_position = kv[:, :usable], gate[:, :usable], 0
        else:
            chunk_kv, chunk_gate, first_window_position = cache_layer.store_compression_weights(
                "compressor", kv, gate
            )

        if chunk_kv.shape[1] > 0:
            n_windows = chunk_kv.shape[1] // self.compress_rate
            chunk_kv = chunk_kv.view(batch, n_windows, self.compress_rate, -1)
            chunk_gate = chunk_gate.view(batch, n_windows, self.compress_rate, -1) + self.position_bias.to(
                chunk_gate.dtype
            )
            compressed = self.kv_norm(
                (chunk_kv * chunk_gate.softmax(dim=2, dtype=torch.float32).to(chunk_kv.dtype)).sum(dim=2)
            )
            positions = torch.arange(n_windows, device=compressed.device)
            positions = (positions * self.compress_rate + first_window_position).unsqueeze(0).expand(batch, -1)
            cos, sin = self.rotary_emb(compressed, position_ids=positions, layer_type=self.rope_layer_type)
            compressed = apply_rotary_pos_emb(compressed.unsqueeze(1), cos, sin).squeeze(1)
        else:
            compressed = chunk_kv.new_zeros((batch, 0, self.head_dim))

        if cache_layer is not None:
            compressed = cache_layer.update_compressor_states("compressor", compressed)
        return compressed.unsqueeze(1)


class DeepseekV4Indexer(nn.Module):
    r"""Lightning Indexer for CSA (paper §2.3.1, eqs. 13-17).

    Picks the top-k compressed KV blocks per query, with ``k = config.index_topk``.
    Each query attends only to those k of the ``seq_len / compress_rate_csa``
    compressed entries — reduction factor ``(seq_len / compress_rate_csa) /
    index_topk`` over full attention against the entire compressed sequence.

    Runs its own scaled-down compressor at ``index_head_dim`` over the same
    windows as the outer CSA compressor, then scores queries against the
    compressed keys with ``∑_h w_{t,h} · ReLU(q_{t,h} · K^IComp_s)`` and keeps
    the top ``index_topk`` indices.

    Has its own rotary because it applies RoPE to two sets of tensors:
    compressed keys at deterministic positions, queries at the model's
    current ``position_ids``. Both use the same ``"compress"`` rope_theta
    so query/key inner products are translation-invariant.
    """

    rope_layer_type = "compress"

    def __init__(self, config) -> None:
        super().__init__()
        self.compress_rate = config.compress_rates["compressed_sparse_attention"]
        self.num_heads = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.index_topk = config.index_topk
        self.softmax_scale = self.head_dim ** -0.5
        self.weights_scaling = self.num_heads ** -0.5
        self.kv_proj = nn.Linear(config.hidden_size, 2 * self.head_dim, bias=False)
        self.gate_proj = nn.Linear(config.hidden_size, 2 * self.head_dim, bias=False)
        self.position_bias = nn.Parameter(torch.empty(self.compress_rate, 2 * self.head_dim))
        self.kv_norm = DeepseekV4RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.q_b_proj = nn.Linear(config.q_lora_rank, self.num_heads * self.head_dim, bias=False)
        self.weights_proj = nn.Linear(config.hidden_size, self.num_heads, bias=False)
        self.rotary_emb = DeepseekV4RotaryEmbedding(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        q_residual: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values: Cache | None,
        layer_idx: int,
    ) -> torch.LongTensor:
        batch, seq_len, _ = hidden_states.shape
        cache_layer: DeepseekV4CSACache = (
            past_key_values.layers[layer_idx] if past_key_values is not None else None
        )
        kv = self.kv_proj(hidden_states)
        gate = self.gate_proj(hidden_states)

        if cache_layer is None:
            usable = (kv.shape[1] // self.compress_rate) * self.compress_rate
            chunk_kv, chunk_gate, first_window_position = kv[:, :usable], gate[:, :usable], 0
        else:
            chunk_kv, chunk_gate, first_window_position = cache_layer.store_compression_weights(
                "indexer", kv, gate
            )

        if chunk_kv.shape[1] > 0:
            n_windows = chunk_kv.shape[1] // self.compress_rate
            ratio = self.compress_rate
            chunk_kv = chunk_kv.view(batch, n_windows, ratio, -1)
            chunk_gate = chunk_gate.view(batch, n_windows, ratio, -1) + self.position_bias.to(chunk_gate.dtype)

            # Same Ca / Cb overlap layout as the outer CSA compressor, at index_head_dim.
            new_kv = chunk_kv.new_zeros((batch, n_windows, 2 * ratio, self.head_dim))
            new_gate = chunk_gate.new_full((batch, n_windows, 2 * ratio, self.head_dim), float("-inf"))
            new_kv[:, :, ratio:] = chunk_kv[..., self.head_dim:]
            new_gate[:, :, ratio:] = chunk_gate[..., self.head_dim:]
            if n_windows > 1:
                new_kv[:, 1:, :ratio] = chunk_kv[:, :-1, :, :self.head_dim]
                new_gate[:, 1:, :ratio] = chunk_gate[:, :-1, :, :self.head_dim]
            if cache_layer is not None:
                prior_kv, prior_gate = cache_layer.update_overlap_state(
                    "indexer", chunk_kv, chunk_gate, self.head_dim
                )
                if prior_kv is not None:
                    new_kv[:, 0, :ratio] = prior_kv.to(new_kv.dtype)
                    new_gate[:, 0, :ratio] = prior_gate.to(new_gate.dtype)

            compressed = self.kv_norm(
                (new_kv * new_gate.softmax(dim=2, dtype=torch.float32).to(new_kv.dtype)).sum(dim=2)
            )
            positions = torch.arange(n_windows, device=compressed.device)
            positions = positions * self.compress_rate + first_window_position
            positions = positions.unsqueeze(0).expand(batch, -1)
            cos, sin = self.rotary_emb(compressed, position_ids=positions, layer_type=self.rope_layer_type)
            compressed = apply_rotary_pos_emb(compressed.unsqueeze(1), cos, sin).squeeze(1)
        else:
            compressed = chunk_kv.new_zeros((batch, 0, self.head_dim))

        compressed_kv = (
            compressed if cache_layer is None
            else cache_layer.update_compressor_states("indexer", compressed)
        )

        cos_q, sin_q = self.rotary_emb(hidden_states, position_ids=position_ids, layer_type=self.rope_layer_type)
        q = self.q_b_proj(q_residual).view(batch, seq_len, -1, self.head_dim).transpose(1, 2)
        q = apply_rotary_pos_emb(q, cos_q, sin_q).transpose(1, 2)

        # ReLU(q·k^T) * weights, then top-k.
        scores = torch.matmul(q.float(), compressed_kv.transpose(-1, -2).float().unsqueeze(1))
        scores = F.relu(scores) * self.softmax_scale
        weights = self.weights_proj(hidden_states).float() * self.weights_scaling
        index_scores = (scores * weights.unsqueeze(-1)).sum(dim=2)
        topk = min(self.index_topk, compressed_kv.shape[1])
        return index_scores.topk(topk, dim=-1).indices


class DeepseekV4CSACompressor(nn.Module):
    """Compressed Sparse Attention compressor (paper §2.3.1, eqs. 9-17).

    Compresses every ``compress_rate_csa`` (m=4) source tokens with overlapping
    windows (effective width ``2 * compress_rate_csa``, stride
    ``compress_rate_csa``) and runs a Lightning Indexer to gather the top
    ``index_topk`` entries per query before they reach core attention.

    ``kv_proj`` / ``gate_proj`` / ``position_bias`` project to ``2 * head_dim``:
    each token contributes two independent compressed series Ca and Cb stored
    in one tensor (``Ca = [..., :head_dim]`` is its contribution to the *next*
    window's compressed entry; ``Cb = [..., head_dim:]`` to the *current*).
    Compressed entry w is the softmax-gated convex combination of window
    ``w-1``'s Ca slice with window w's Cb slice over ``2 * compress_rate``
    slots. For ``w = 0`` we need the previous window's Ca from the prior
    forward call — held in ``cache_layer.overlap_kv``.
    """

    rope_layer_type = "compress"

    def __init__(self, config) -> None:
        super().__init__()
        self.compress_rate = config.compress_rates["compressed_sparse_attention"]
        self.head_dim = config.head_dim
        self.kv_proj = nn.Linear(config.hidden_size, 2 * self.head_dim, bias=False)
        self.gate_proj = nn.Linear(config.hidden_size, 2 * self.head_dim, bias=False)
        self.position_bias = nn.Parameter(torch.empty(self.compress_rate, 2 * self.head_dim))
        self.kv_norm = DeepseekV4RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.rotary_emb = DeepseekV4RotaryEmbedding(config)
        self.indexer = DeepseekV4Indexer(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        q_residual: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values: Cache | None,
        layer_idx: int,
    ) -> torch.Tensor:
        batch, seq_len, _ = hidden_states.shape
        cache_layer: DeepseekV4CSACache = (
            past_key_values.layers[layer_idx] if past_key_values is not None else None
        )
        kv = self.kv_proj(hidden_states)
        gate = self.gate_proj(hidden_states)

        if cache_layer is None:
            usable = (kv.shape[1] // self.compress_rate) * self.compress_rate
            chunk_kv, chunk_gate, first_window_position = kv[:, :usable], gate[:, :usable], 0
        else:
            chunk_kv, chunk_gate, first_window_position = cache_layer.store_compression_weights(
                "compressor", kv, gate
            )

        if chunk_kv.shape[1] > 0:
            n_windows = chunk_kv.shape[1] // self.compress_rate
            ratio = self.compress_rate
            chunk_kv = chunk_kv.view(batch, n_windows, ratio, -1)
            chunk_gate = chunk_gate.view(batch, n_windows, ratio, -1) + self.position_bias.to(chunk_gate.dtype)

            # Lay out the two series in [B, n_win, 2*ratio, head_dim]: Cb in the
            # second half (current window), Ca of the previous window in the
            # first half. Window 0's first half stays zero-kv / -inf-gate
            # (softmax weight 0) on the very first forward; on later calls
            # the cache fills it from the saved Ca slice.
            new_kv = chunk_kv.new_zeros((batch, n_windows, 2 * ratio, self.head_dim))
            new_gate = chunk_gate.new_full((batch, n_windows, 2 * ratio, self.head_dim), float("-inf"))
            new_kv[:, :, ratio:] = chunk_kv[..., self.head_dim:]
            new_gate[:, :, ratio:] = chunk_gate[..., self.head_dim:]
            if n_windows > 1:
                new_kv[:, 1:, :ratio] = chunk_kv[:, :-1, :, :self.head_dim]
                new_gate[:, 1:, :ratio] = chunk_gate[:, :-1, :, :self.head_dim]
            if cache_layer is not None:
                prior_kv, prior_gate = cache_layer.update_overlap_state(
                    "compressor", chunk_kv, chunk_gate, self.head_dim
                )
                if prior_kv is not None:
                    new_kv[:, 0, :ratio] = prior_kv.to(new_kv.dtype)
                    new_gate[:, 0, :ratio] = prior_gate.to(new_gate.dtype)

            # Softmax in fp32 for stability (bf16/fp16 logits can collapse pairs
            # that only differ slightly, especially with large window widths).
            compressed = self.kv_norm(
                (new_kv * new_gate.softmax(dim=2, dtype=torch.float32).to(new_kv.dtype)).sum(dim=2)
            )
            positions = torch.arange(n_windows, device=compressed.device)
            positions = positions * self.compress_rate + first_window_position
            positions = positions.unsqueeze(0).expand(batch, -1)
            cos, sin = self.rotary_emb(compressed, position_ids=positions, layer_type=self.rope_layer_type)
            compressed = apply_rotary_pos_emb(compressed.unsqueeze(1), cos, sin).squeeze(1)
        else:
            compressed = chunk_kv.new_zeros((batch, 0, self.head_dim))

        if cache_layer is not None:
            compressed = cache_layer.update_compressor_states("compressor", compressed)
        compressed_kv = compressed.unsqueeze(1)

        # Lightning Indexer: gather top-`index_topk` compressed entries per query.
        topk = self.indexer(hidden_states, q_residual, position_ids, past_key_values, layer_idx)
        expanded = compressed_kv.unsqueeze(2).expand(-1, -1, seq_len, -1, -1)
        idx = topk.unsqueeze(1).unsqueeze(-1).expand(-1, 1, -1, -1, self.head_dim)
        return torch.gather(expanded, 3, idx).reshape(batch, 1, -1, self.head_dim)


# =========================================================================== #
# Eager attention with sinks (DSV4-specific fallback)
# =========================================================================== #


def _eager_attention_with_sinks(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float | int = 0.0,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Eager attention with per-head learnable sinks (gpt-oss style).

    DSV4 attention has a learnable ``sinks`` parameter per head. The
    softmax sees the regular logits *plus* a synthetic sink logit per
    head; the sink is dropped from the output but its presence "absorbs"
    a configurable fraction of attention mass, which the paper argues
    stabilizes very long-context behavior. Used as the fallback when
    ``config._attn_implementation`` falls through ``ALL_ATTENTION_FUNCTIONS``.
    """
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    sinks = module.sinks.reshape(1, -1, 1, 1).expand(query.shape[0], -1, query.shape[-2], -1)
    combined_logits = torch.cat([attn_weights, sinks], dim=-1)
    # Subtract per-row max for bf16/fp16 overflow safety (slight numerical
    # change vs the pure-softmax form, but matches HF's reference).
    combined_logits = combined_logits - combined_logits.max(dim=-1, keepdim=True).values
    probs = F.softmax(combined_logits, dim=-1, dtype=combined_logits.dtype)
    scores = probs[..., :-1]  # drop the sink column
    attn_weights = F.dropout(scores, p=dropout, training=module.training).to(value_states.dtype)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


_COMPRESSOR_CLASSES: dict[str, type | None] = {
    # Keyed by the alloy source-coupled mixer name (matches config.layer_types).
    "dsv4_sliding_attention": None,
    "dsv4_csa_attention": DeepseekV4CSACompressor,
    "dsv4_hca_attention": DeepseekV4HCACompressor,
}


# =========================================================================== #
# Main attention class — registered under 3 names
# =========================================================================== #


@register_mixer("dsv4_sliding_attention", attr_name="self_attn", mask_kind="sliding")
@register_mixer("dsv4_hca_attention", attr_name="self_attn", mask_kind="sliding")
@register_mixer("dsv4_csa_attention", attr_name="self_attn", mask_kind="sliding")
class DeepseekV4Attention(nn.Module):
    r"""DeepSeek-V4 attention block.

    Differences from classic attention:

    * **Shared-KV MQA**: ``num_key_value_heads = 1``; ``kv_proj`` projects
      directly to that single KV head and the same tensor is read as both
      key and value.
    * **Partial RoPE** on the first ``rope_head_dim`` of each head; head
      layout is ``[nope | rope]``. RoPE is applied with conjugate rotation
      to the attention output's rope slice so each KV entry's contribution
      stays a function of *relative* distance to the query.
    * **Per-head learnable sinks** (gpt-oss style) absorb a fraction of
      attention mass for long-context stability.
    * **Grouped low-rank output projection**: ``o_a_proj`` (block-diagonal
      grouped linear) + ``o_b_proj`` (mixing linear) replace the standard
      single output projection.
    * **Three cache variants** routed by ``config.layer_types[layer_idx]``:
      pure sliding (no compressor), sliding + CSA, sliding + HCA.

    Registered under three alloy mixer names (``dsv4_sliding_attention`` /
    ``dsv4_hca_attention`` / ``dsv4_csa_attention``). The name decides which
    compressor (or none) is constructed at ``__init__`` time.
    """

    def __init__(self, config, layer_idx: int) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        # alloy-side name; e.g. "dsv4_hca_attention". The original HF DSV4
        # uses canonical names ("heavily_compressed_attention"); both work
        # in our _COMPRESSOR_CLASSES table after the alloy mapping.
        self.layer_type = config.layer_types[layer_idx]
        self.num_heads = config.num_attention_heads
        self.num_key_value_groups = config.num_attention_heads  # single KV head, broadcast to all
        self.head_dim = config.head_dim
        self.sliding_window = config.sliding_window
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.scaling = self.head_dim ** -0.5

        self.q_a_proj = nn.Linear(config.hidden_size, config.q_lora_rank, bias=False)
        self.q_a_norm = DeepseekV4RMSNorm(config.q_lora_rank, eps=config.rms_norm_eps)
        self.q_b_proj = nn.Linear(config.q_lora_rank, self.num_heads * self.head_dim, bias=False)
        self.q_b_norm = DeepseekV4UnweightedRMSNorm(eps=config.rms_norm_eps)
        self.kv_proj = nn.Linear(config.hidden_size, self.head_dim, bias=False)
        self.kv_norm = DeepseekV4RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.o_a_proj = DeepseekV4GroupedLinear(
            self.num_heads * self.head_dim // config.o_groups,
            config.o_groups * config.o_lora_rank,
            config.o_groups,
        )
        self.o_b_proj = nn.Linear(config.o_groups * config.o_lora_rank, config.hidden_size, bias=False)
        self.sinks = nn.Parameter(torch.empty(self.num_heads))

        compressor_cls = _COMPRESSOR_CLASSES.get(self.layer_type)
        self.compressor = compressor_cls(config) if compressor_cls is not None else None

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        if position_embeddings is None:
            raise ValueError("DeepseekV4Attention requires position_embeddings (cos, sin) tuple.")
        cos, sin = position_embeddings

        q_residual = self.q_a_norm(self.q_a_proj(hidden_states))
        q = self.q_b_proj(q_residual).view(*hidden_shape).transpose(1, 2)
        q = self.q_b_norm(q)
        q = apply_rotary_pos_emb(q, cos, sin)

        kv = self.kv_norm(self.kv_proj(hidden_states)).view(*hidden_shape).transpose(1, 2)
        kv = apply_rotary_pos_emb(kv, cos, sin)

        if past_key_values is not None:  # sliding where K==V
            kv = past_key_values.update(kv, kv, self.layer_idx)[0]

        if self.compressor is not None:  # CSA or HCA
            compressed_kv = self.compressor(
                hidden_states, q_residual, position_ids, past_key_values, self.layer_idx
            )
            kv = torch.cat([kv, compressed_kv], dim=2)

        # The compressor concatenates extra entries onto the KV axis after the
        # sliding-window cache update, so a tensor attention_mask (built for
        # the pre-concat KV length) needs right-padding to cover them with 0
        # (= "always attend"). Skip when attention_mask is a non-tensor
        # (e.g. flex-attention BlockMask).
        if isinstance(attention_mask, torch.Tensor) and kv.shape[2] > attention_mask.shape[-1]:
            attention_mask = F.pad(
                attention_mask, (0, kv.shape[2] - attention_mask.shape[-1]), value=0.0
            )

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, _eager_attention_with_sinks
        )
        attn_output, attn_weights = attention_interface(
            self,
            q,
            kv,
            kv,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            s_aux=self.sinks,
            **kwargs,
        )

        # K=V in V4, so V picked up RoPE on its trailing rope slice. Apply
        # the conjugate rotation (-sin) at the query position to undo it on
        # the rope slice of the output before the grouped output projection
        # mixes heads. The transpose pair is just layout: apply_rotary expects
        # [B, S, H, D] (its unsqueeze_dim=1 adds a head-broadcast dim to
        # cos/sin); attention gave us [B, H, S, D].
        attn_output = apply_rotary_pos_emb(attn_output.transpose(1, 2), cos, -sin).transpose(1, 2)

        grouped = attn_output.reshape(*input_shape, self.config.o_groups, -1)
        grouped = self.o_a_proj(grouped).flatten(2)
        output = self.o_b_proj(grouped)
        return output, attn_weights


__all__ = [
    "DeepseekV4HCACache",
    "DeepseekV4CSACache",
    "DeepseekV4GroupedLinear",
    "DeepseekV4HCACompressor",
    "DeepseekV4Indexer",
    "DeepseekV4CSACompressor",
    "DeepseekV4Attention",
]
