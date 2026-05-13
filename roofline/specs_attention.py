"""DSV4 attention roofline spec — one class for all 3 layer types.

Models DSV4 attention as standard SDPA with an *effective KV length per query*
that varies by layer flavor and depends on (query_len + kv_cache_len) — the
full context length, not just the new tokens being processed this forward:

  * ``dsv4_sliding_attention``  — KV_len = min(Q+P, sliding_window)
  * ``dsv4_hca_attention``      — KV_len = min(Q+P, W) + ceil((Q+P) / hca_rate)
                                  (sliding + heavy compressed entries)
  * ``dsv4_csa_attention``      — KV_len = min(Q+P, W) + index_topk
                                  (sliding + Lightning-Indexer-gated entries)

Where Q = ``query_len`` (new tokens this forward), P = ``kv_cache_len``
(tokens already in cache from previous forwards). For prefill from cold
P=0, so KV_len reduces to f(Q). For decode Q=1, P=full_context, so KV_len
is dominated by P. Mini-prefill is a chunk with Q=chunk_len and P=prefix.

DSV4 is MQA (``num_kv_heads == 1``): K and V are a single shared head broadcast
to all attention heads, halving KV-side activation traffic; under fusion the
shared K/V tensor is also read once for both the QK and AV stages.

Components included (matmul-dominated, 95%+ of total compute):

  * Q LoRA projection (``q_a_proj`` + ``q_b_proj``) — Q new tokens
  * KV projection (``kv_proj``, MQA single head) — Q new tokens
  * Compressor projections (HCA/CSA only) — Q new tokens
  * Main SDPA at effective KV_len — Q queries × full effective KV
  * Output: grouped ``o_a_proj`` (block-diagonal) + ``o_b_proj`` — Q new tokens
  * Lightning Indexer (CSA only) — Q queries × n_compressed (full context's
    compressed entries, including those cached from previous forwards)

Components skipped (sub-1% combined for typical sizes):

  * RMSNorm, RoPE, attention-mask handling
  * Softmax FLOPs inside SDPA
  * Compressor's softmax-over-windows + position-bias add
  * Per-head sinks compute
  * Distinct write of new KV entries to cache (small, tracked via
    effective_kv_len since the read path dominates total HBM traffic)

Bytes model: optimal-fusion FlashAttention assumptions — input X read once,
output written once, all weight buffers read once each. The KV tensor IS
materialized to HBM (too big for SRAM at any realistic context length) at
effective KV_len, contributing ``B · 1 · KV_len · head_dim · es`` traffic
(single MQA head, K==V shared).
"""
from __future__ import annotations

import math

import torch

from .specs import RooflineSpec, dtype_size, register_spec


def _effective_kv_len(query_len: int, layer_type: str, config, *, kv_cache_len: int = 0) -> int:
    """KV length each query attends to, by layer type.

    Computed over the full context (``query_len + kv_cache_len``); the cache
    contributes most of the KV in mini-prefill / decode regimes.
    """
    total_seq = query_len + kv_cache_len
    sw = min(total_seq, config.sliding_window)
    if layer_type == "dsv4_sliding_attention":
        return sw
    if layer_type == "dsv4_hca_attention":
        hca_rate = config.compress_rates["heavily_compressed_attention"]
        return sw + math.ceil(total_seq / hca_rate)
    if layer_type == "dsv4_csa_attention":
        return sw + config.index_topk
    raise KeyError(f"unknown DSV4 layer type: {layer_type!r}")


def _projection_flops(n_query_tokens: int, config) -> int:
    """Q/KV/O matmul flops common to all 3 layer types — operate on Q new tokens.

    DSV4 is MQA — kv_proj output dim is just ``head_dim``, not
    ``num_kv_heads * head_dim``. Output is grouped: ``o_a_proj`` is block-
    diagonal so its FLOPs equal a (n_heads*head_dim → o_lora_rank) projection
    even though the weight is partitioned into ``o_groups`` blocks.
    """
    hidden = config.hidden_size
    qlr = config.q_lora_rank
    nh = config.num_attention_heads
    hd = config.head_dim
    og = config.o_groups
    olr = config.o_lora_rank
    return (
        2 * n_query_tokens * hidden * qlr        # q_a_proj
        + 2 * n_query_tokens * qlr * (nh * hd)   # q_b_proj
        + 2 * n_query_tokens * hidden * hd       # kv_proj (MQA, single head)
        + 2 * n_query_tokens * (nh * hd) * olr   # o_a_proj (grouped, equiv flops)
        + 2 * n_query_tokens * (og * olr) * hidden  # o_b_proj
    )


def _projection_weights(config, es: int) -> int:
    """Weight bytes for Q/KV/O projections + per-head sinks."""
    hidden = config.hidden_size
    qlr = config.q_lora_rank
    nh = config.num_attention_heads
    hd = config.head_dim
    og = config.o_groups
    olr = config.o_lora_rank
    return (
        hidden * qlr * es
        + qlr * nh * hd * es
        + hidden * hd * es
        + nh * hd * olr * es
        + og * olr * hidden * es
        + nh * es
    )


def _hca_compressor_flops(query_len: int, config) -> int:
    """HCA compressor: kv_proj + gate_proj on Q new tokens [hidden -> head_dim].

    Cached compressed entries from previous forwards are NOT re-projected;
    only the fresh ``query_len`` tokens flow through these linears each call.
    """
    hd = config.head_dim
    return 2 * 2 * query_len * config.hidden_size * hd  # ×2 for kv + gate


def _hca_compressor_weights(config, es: int) -> int:
    hd = config.head_dim
    rate = config.compress_rates["heavily_compressed_attention"]
    return (
        2 * config.hidden_size * hd * es
        + rate * hd * es
        + hd * es
    )


def _csa_compressor_flops(query_len: int, config) -> int:
    """CSA compressor: kv_proj + gate_proj on Q new tokens [hidden -> 2*head_dim]."""
    hd = config.head_dim
    return 2 * 2 * query_len * config.hidden_size * (2 * hd)


def _csa_compressor_weights(config, es: int) -> int:
    hd = config.head_dim
    rate = config.compress_rates["compressed_sparse_attention"]
    return (
        2 * config.hidden_size * (2 * hd) * es
        + rate * (2 * hd) * es
        + hd * es
    )


def _indexer_flops(batch: int, query_len: int, config, *, kv_cache_len: int = 0) -> int:
    """Lightning Indexer FLOPs (CSA only).

    The score matmul is ``query_len × n_compressed × index_head_dim`` per head,
    where ``n_compressed = ceil((query_len + kv_cache_len) / csa_rate)`` covers
    BOTH freshly emitted and previously cached compressed entries.

    At long-context decode (Q=1, large P), this term grows as O(P/csa_rate)
    per query — much smaller per-query than at prefill, but still the
    dominant indexer cost since the projection terms scale with Q only.
    The indexer's projections (kv_proj/gate_proj/q_b_proj/weights_proj)
    operate on Q new tokens — cached compressed entries are not re-projected.
    """
    hidden = config.hidden_size
    qlr = config.q_lora_rank
    nih = config.index_n_heads
    ihd = config.index_head_dim
    csa_rate = config.compress_rates["compressed_sparse_attention"]
    total_seq = query_len + kv_cache_len
    n_compressed = math.ceil(total_seq / csa_rate)
    n_query_tokens = batch * query_len
    return (
        2 * n_query_tokens * hidden * (2 * ihd)
        + 2 * n_query_tokens * hidden * (2 * ihd)
        + 2 * n_query_tokens * qlr * (nih * ihd)
        + 2 * n_query_tokens * hidden * nih
        + 2 * batch * query_len * nih * ihd * n_compressed
    )


def _indexer_weights(config, es: int) -> int:
    hidden = config.hidden_size
    qlr = config.q_lora_rank
    nih = config.index_n_heads
    ihd = config.index_head_dim
    csa_rate = config.compress_rates["compressed_sparse_attention"]
    return (
        hidden * (2 * ihd) * es
        + hidden * (2 * ihd) * es
        + qlr * nih * ihd * es
        + hidden * nih * es
        + csa_rate * (2 * ihd) * es
        + ihd * es
    )


# --------------------------------------------------------------------------- #
# DSV4AttentionSpec — one spec instance per layer-type name.
# --------------------------------------------------------------------------- #


class DSV4AttentionSpec(RooflineSpec):
    """DSV4 attention spec; instantiate one per layer type and register under
    the matching name.

    ``flops`` / ``bytes`` accept ``kv_cache_len`` via ``**kwargs`` for
    mini-prefill / decode modes. Default ``kv_cache_len=0`` is the cold
    prefill path.

    Variables: B = batch, Q = query_len, P = kv_cache_len, T = Q + P (total
    seq), Nq = B * Q (new query tokens), H = hidden_size, qlr = q_lora_rank,
    nh = num_attention_heads, hd = head_dim, og = o_groups, olr = o_lora_rank,
    SW = min(T, sliding_window), nih = index_n_heads, ihd = index_head_dim,
    rate_h = compress_rates["heavily_compressed_attention"],
    rate_c = compress_rates["compressed_sparse_attention"],
    es = dtype_size(dtype). DSV4 is MQA so kv_len here is the SINGLE shared
    head; cached + new entries share the same KV buffer.

    ## effective KV length per layer type
        dsv4_sliding_attention : kv_len = SW
        dsv4_hca_attention     : kv_len = SW + ceil(T / rate_h)
        dsv4_csa_attention     : kv_len = SW + index_topk

    ## FLOPs (Q/KV/O projections + main SDPA + per-layer-type extras)
        2 * Nq * H * qlr                # q_a_proj (down)
      + 2 * Nq * qlr * (nh * hd)        # q_b_proj (up)
      + 2 * Nq * H * hd                 # kv_proj (MQA single head)
      + 2 * Nq * (nh * hd) * olr        # o_a_proj (grouped, equiv flops)
      + 2 * Nq * (og * olr) * H         # o_b_proj
      + 4 * B * nh * Q * kv_len * hd    # main SDPA (Q × kv_len, multi-head)
      [HCA extra]
        + 4 * Q * H * hd                # compressor: kv_proj + gate_proj
      [CSA extra]
        + 4 * Q * H * (2 * hd)          # compressor: kv_proj + gate_proj (2*hd)
        + 4 * Nq * H * (2 * ihd)        # indexer: kv_proj + gate_proj
        + 2 * Nq * qlr * (nih * ihd)    # indexer: q_b_proj
        + 2 * Nq * H * nih              # indexer: weights_proj
        + 2 * B * Q * nih * ihd * ceil(T / rate_c)   # indexer: score matmul

    ## Bytes (FlashAttention-style fusion; single MQA KV head materialized)
        H*qlr*es + qlr*nh*hd*es + H*hd*es + nh*hd*olr*es + og*olr*H*es + nh*es
            # Q/KV/O projection weights + per-head sinks [nh]
      + Nq * H * es                     # activation read (new tokens)
      + Nq * H * es                     # activation write (new tokens)
      + B * kv_len * hd * es            # KV buffer (single shared head)
      [HCA extra weights]
        + 2 * H * hd * es + rate_h * hd * es + hd * es
      [CSA extra weights]
        + 2 * H * (2*hd) * es + rate_c * (2*hd) * es + hd * es   # compressor
        + 2 * H * (2*ihd) * es + qlr * nih * ihd * es + H * nih * es
            + rate_c * (2*ihd) * es + ihd * es                   # indexer
    """

    def __init__(self, layer_type: str) -> None:
        if layer_type not in {
            "dsv4_sliding_attention",
            "dsv4_hca_attention",
            "dsv4_csa_attention",
        }:
            raise ValueError(f"unknown DSV4 layer type: {layer_type!r}")
        self.layer_type = layer_type

    def _check_in_shape(self, in_shape: tuple[int, ...]) -> tuple[int, int]:
        if len(in_shape) != 3:
            raise ValueError(
                f"DSV4AttentionSpec expects 3D in_shape (batch, query_len, hidden); "
                f"got {in_shape}."
            )
        return in_shape[0], in_shape[1]

    def flops(self, in_shape: tuple[int, ...], config, **kwargs) -> int:
        kv_cache_len = int(kwargs.get("kv_cache_len", 0))
        batch, query_len = self._check_in_shape(in_shape)
        n_query_tokens = batch * query_len
        nh = config.num_attention_heads
        hd = config.head_dim
        kv_len = _effective_kv_len(query_len, self.layer_type, config, kv_cache_len=kv_cache_len)

        # Q/KV/O projections + grouped output — operate on Q new tokens
        flops = _projection_flops(n_query_tokens, config)

        # Main SDPA: Q queries × full effective KV (cache + new)
        flops += 4 * batch * nh * query_len * kv_len * hd

        if self.layer_type == "dsv4_hca_attention":
            flops += _hca_compressor_flops(query_len, config)
        elif self.layer_type == "dsv4_csa_attention":
            flops += _csa_compressor_flops(query_len, config)
            flops += _indexer_flops(batch, query_len, config, kv_cache_len=kv_cache_len)

        return flops

    def bytes(self, in_shape: tuple[int, ...], config, dtype: torch.dtype, **kwargs) -> int:
        kv_cache_len = int(kwargs.get("kv_cache_len", 0))
        batch, query_len = self._check_in_shape(in_shape)
        n_query_tokens = batch * query_len
        hidden = config.hidden_size
        hd = config.head_dim
        es = dtype_size(dtype)
        kv_len = _effective_kv_len(query_len, self.layer_type, config, kv_cache_len=kv_cache_len)

        weights = _projection_weights(config, es)
        if self.layer_type == "dsv4_hca_attention":
            weights += _hca_compressor_weights(config, es)
        elif self.layer_type == "dsv4_csa_attention":
            weights += _csa_compressor_weights(config, es)
            weights += _indexer_weights(config, es)

        # Activation bytes under FlashAttention-style fusion:
        #   * Input X read once (Q new tokens)
        #   * Output written once (Q new tokens)
        #   * KV tensor materialized in HBM at full effective length (single
        #     MQA head, K==V shared) — covers cached + new entries; new-entry
        #     write is folded in (small relative to the read).
        act_in = n_query_tokens * hidden * es
        act_out = n_query_tokens * hidden * es
        kv_act = batch * 1 * kv_len * hd * es

        return weights + act_in + act_out + kv_act


# --------------------------------------------------------------------------- #
# Registration
# --------------------------------------------------------------------------- #


register_spec("dsv4_sliding_attention", DSV4AttentionSpec("dsv4_sliding_attention"))
register_spec("dsv4_hca_attention", DSV4AttentionSpec("dsv4_hca_attention"))
register_spec("dsv4_csa_attention", DSV4AttentionSpec("dsv4_csa_attention"))


# =========================================================================== #
# Qwen3 standard MHA / GQA attention spec
# =========================================================================== #


def _qwen3_effective_kv_len(query_len: int, layer_type: str, config, *, kv_cache_len: int = 0) -> int:
    """KV length each query attends to.

    Full causal: attends to entire prefix + queries (total_seq).
    Sliding: capped at ``config.sliding_window``.
    """
    total_seq = query_len + kv_cache_len
    if layer_type == "qwen3_attention_sliding":
        return min(total_seq, config.sliding_window)
    return total_seq


class Qwen3AttentionSpec(RooflineSpec):
    r"""Spec for Qwen3 / Qwen3.5 standard MHA + GQA attention.

    One class registered under two names:

      * ``qwen3_attention``         — full causal, KV_len = total_seq
      * ``qwen3_attention_sliding`` — sliding window, KV_len capped at
                                       ``config.sliding_window``

    GQA: ``num_key_value_heads`` may be < ``num_attention_heads``. Each
    KV head is broadcast to ``num_heads / num_kv_heads`` query heads
    inside SDPA. Reduces KV-side HBM bandwidth by the GQA factor (this is
    the headline GQA advantage for long-context decode).

    Optional features:

      * ``attn_output_gate=True`` (qwen3.5 style) — q_proj produces 2x
        output; second half is sigmoid-gated against attention output
        before o_proj. Adds ~num_heads*head_dim weight + small flops.
      * ``attention_bias=True`` — bias terms on q/k/v/o linears.

    Skipped in v1 (sub-1% combined): RMSNorm on q/k, RoPE.

    Variables: B = batch, Q = query_len, P = kv_cache_len, Nq = B * Q,
    H = hidden_size, nh = num_attention_heads, nkv = num_key_value_heads,
    hd = head_dim, qd = nh*hd, kvd = nkv*hd, g = 2 if attn_output_gate
    else 1, es = dtype_size(dtype). kv_len = min(Q+P, sliding_window) for
    ``qwen3_attention_sliding``, else Q+P.

    ## FLOPs
        2 * Nq * H * (qd * g)           # q_proj (× g if attn_output_gate)
      + 2 * Nq * H * kvd                # k_proj
      + 2 * Nq * H * kvd                # v_proj
      + 2 * Nq * qd * H                 # o_proj
      + bias_flops                      # Nq * (qd*g + 2*kvd + H) if attention_bias
      + 4 * B * nh * Q * kv_len * hd    # SDPA QK^T + AV (Q has nh heads, KV broadcast)
      + 4 * Nq * qd                     # output gate sigmoid + mul (only if g=2)

    ## Bytes (FlashAttention-style fusion; K and V are separate tensors)
        H * (qd*g) * es                 # q_proj weights
      + H * kvd * es * 2                # k_proj + v_proj weights
      + qd * H * es                     # o_proj weights
      + 2 * hd * es                     # q_norm + k_norm gain vectors
      + bias_bytes                      # (qd*g + 2*kvd + H) * es if attention_bias
      + Nq * H * es                     # activation read
      + Nq * H * es                     # activation write
      + B * 2 * nkv * kv_len * hd * es  # KV cache (K and V both materialized, GQA = nkv heads)
    """

    def __init__(self, layer_type: str) -> None:
        if layer_type not in {"qwen3_attention", "qwen3_attention_sliding"}:
            raise ValueError(f"unknown qwen3 layer type: {layer_type!r}")
        self.layer_type = layer_type

    def _check_in_shape(self, in_shape: tuple[int, ...]) -> tuple[int, int]:
        if len(in_shape) != 3:
            raise ValueError(
                f"Qwen3AttentionSpec expects 3D in_shape (batch, query_len, hidden); "
                f"got {in_shape}."
            )
        return in_shape[0], in_shape[1]

    def _read_config(self, config) -> dict:
        hidden = config.hidden_size
        num_heads = config.num_attention_heads
        num_kv_heads = config.num_key_value_heads
        head_dim = getattr(config, "head_dim", None) or (hidden // num_heads)
        return {
            "hidden": hidden,
            "num_heads": num_heads,
            "num_kv_heads": num_kv_heads,
            "head_dim": head_dim,
            "q_dim": num_heads * head_dim,
            "kv_dim": num_kv_heads * head_dim,
            "gate_mul": 2 if bool(getattr(config, "attn_output_gate", False)) else 1,
            "bias": bool(getattr(config, "attention_bias", False)),
        }

    def flops(self, in_shape: tuple[int, ...], config, **kwargs) -> int:
        kv_cache_len = int(kwargs.get("kv_cache_len", 0))
        batch, query_len = self._check_in_shape(in_shape)
        d = self._read_config(config)
        n_query_tokens = batch * query_len
        kv_len = _qwen3_effective_kv_len(query_len, self.layer_type, config, kv_cache_len=kv_cache_len)

        # Linear projections
        flops = (
            2 * n_query_tokens * d["hidden"] * (d["q_dim"] * d["gate_mul"])  # q_proj (× gate_mul if gated)
            + 2 * n_query_tokens * d["hidden"] * d["kv_dim"]                  # k_proj
            + 2 * n_query_tokens * d["hidden"] * d["kv_dim"]                  # v_proj
            + 2 * n_query_tokens * d["q_dim"] * d["hidden"]                   # o_proj
        )
        if d["bias"]:
            flops += n_query_tokens * (
                d["q_dim"] * d["gate_mul"] + 2 * d["kv_dim"] + d["hidden"]
            )

        # SDPA: QK^T + AV at effective KV_len. Q has full num_heads;
        # K/V are GQA-grouped but broadcast for the matmul, so flop count
        # uses num_heads on both passes.
        flops += 4 * batch * d["num_heads"] * query_len * kv_len * d["head_dim"]

        # Output gate (qwen3.5 style): sigmoid + elementwise mul on [N, q_dim]
        # ~3 flops/elem for sigmoid + 1 for mul ≈ 4·N·q_dim
        if d["gate_mul"] == 2:
            flops += 4 * n_query_tokens * d["q_dim"]

        return flops

    def bytes(self, in_shape: tuple[int, ...], config, dtype: torch.dtype, **kwargs) -> int:
        kv_cache_len = int(kwargs.get("kv_cache_len", 0))
        batch, query_len = self._check_in_shape(in_shape)
        d = self._read_config(config)
        es = dtype_size(dtype)
        n_query_tokens = batch * query_len
        kv_len = _qwen3_effective_kv_len(query_len, self.layer_type, config, kv_cache_len=kv_cache_len)

        # Weight bytes
        weights = (
            d["hidden"] * (d["q_dim"] * d["gate_mul"]) * es   # q_proj
            + d["hidden"] * d["kv_dim"] * es                   # k_proj
            + d["hidden"] * d["kv_dim"] * es                   # v_proj
            + d["q_dim"] * d["hidden"] * es                    # o_proj
            + 2 * d["head_dim"] * es                           # q_norm + k_norm gains
        )
        if d["bias"]:
            weights += (d["q_dim"] * d["gate_mul"] + 2 * d["kv_dim"] + d["hidden"]) * es

        # Activations under fusion: input read, output written
        act_in = n_query_tokens * d["hidden"] * es
        act_out = n_query_tokens * d["hidden"] * es

        # KV materialization in HBM. K and V are SEPARATE tensors (no V==K
        # sharing as in DSV4), so 2× the per-tensor traffic. GQA reduces by
        # using num_kv_heads instead of num_heads.
        kv_act = batch * 2 * d["num_kv_heads"] * kv_len * d["head_dim"] * es

        return weights + act_in + act_out + kv_act


register_spec("qwen3_attention", Qwen3AttentionSpec("qwen3_attention"))
register_spec("qwen3_attention_sliding", Qwen3AttentionSpec("qwen3_attention_sliding"))


__all__ = [
    "DSV4AttentionSpec",
    "Qwen3AttentionSpec",
]
