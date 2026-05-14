"""Roofline spec for DFlash draft-model attention.

DFlash drafts (z-lab/dflash repo) use a modified Qwen3 attention block:

  * **K and V are computed on ``cat([target_hidden, noise_block])``**,
    where ``target_hidden`` is a length-``ctx_len`` context tensor
    derived from the target model's anchor-layer hidden states. So K/V
    projection runs on ``ctx_len + q_len`` tokens (not just q_len).
  * **Q is computed on the noise block only**, so Q projection runs on
    ``q_len`` tokens.
  * **Attention is non-causal** — every query in the noise block sees
    every K (past + ctx + noise). Flop count is
    ``4 * B * H * q_len * (past + ctx + q_len) * head_dim``; this
    matches the Qwen3AttentionSpec convention which already counts the
    full non-causal area (causal kernels save no flops in practice).
  * **MLP and O projection run on q_len only** — ``target_hidden``
    never enters the post-attention path.

See ``dflash_understanding.md`` for the per-round data flow.

Workload kwargs (passed through ``**kwargs`` to ``flops`` / ``bytes``):

  * ``kv_cache_len`` (int): existing draft KV cache length (= accepted
    prefix tokens accumulated from prior rounds; 0 at round 1).
  * ``ctx_len``     (int): length of the target_hidden context this
    round (= acc_len + 1 at steady state; = prompt_len at round 1).
"""
from __future__ import annotations

import math
from typing import Optional

import torch

from .specs import RooflineSpec, dtype_size, register_spec


# --------------------------------------------------------------------------- #
# Spec
# --------------------------------------------------------------------------- #


class DFlashAttentionSpec(RooflineSpec):
    r"""Spec for one DFlash draft decoder layer's attention.

    Variables: B = batch, Q = q_len (= block_size, the noise block size),
    P = kv_cache_len (accumulated accepted prefix in the draft cache),
    C = ctx_len (length of target_hidden this round), H = hidden_size,
    nh = num_attention_heads, nkv = num_key_value_heads, hd = head_dim,
    es = dtype_size(dtype). Total non-causal KV length at attention
    time = P + C + Q.

    ## FLOPs
        2 * B * Q * H * (nh * hd)            # q_proj (Q on noise only)
      + 2 * B * (C + Q) * H * (nkv * hd)     # k_proj on cat[target_hidden, noise]
      + 2 * B * (C + Q) * H * (nkv * hd)     # v_proj
      + 2 * B * Q * (nh * hd) * H            # o_proj (Q on noise only)
      + 4 * B * nh * Q * (P + C + Q) * hd    # non-causal SDPA QK^T + AV

    ## Bytes (fused, K and V are separate tensors)
        H * (nh * hd) * es                   # q_proj weights
      + 2 * H * (nkv * hd) * es              # k_proj + v_proj weights
      + (nh * hd) * H * es                   # o_proj weights
      + 2 * hd * es                          # q_norm + k_norm gain vectors
      + B * Q * H * es                       # noise activation read (Q only)
      + B * C * H * es                       # target_hidden activation read
      + B * Q * H * es                       # output write (Q only)
      + B * 2 * nkv * (P + C + Q) * hd * es  # KV materialized non-causally at
                                              # full length (past + ctx + noise)

    Note: the draft cache stores K/V at length ``P + C`` after the round's
    crop (the noise block is dropped, the target_hidden K/V is retained
    as the new accepted prefix). This spec models the IN-FLIGHT KV
    traffic during the forward, which is the relevant byte cost.
    """

    def __init__(self) -> None:
        pass

    def _check_in_shape(self, in_shape: tuple[int, ...]) -> tuple[int, int]:
        if len(in_shape) != 3:
            raise ValueError(
                f"DFlashAttentionSpec expects 3D in_shape (batch, q_len, hidden); "
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
            "bias": bool(getattr(config, "attention_bias", False)),
        }

    def flops(self, in_shape: tuple[int, ...], config, **kwargs) -> int:
        kv_cache_len = int(kwargs.get("kv_cache_len", 0))
        ctx_len = int(kwargs.get("ctx_len", 0))
        batch, q_len = self._check_in_shape(in_shape)
        d = self._read_config(config)
        n_q = batch * q_len
        n_kv_in = batch * (ctx_len + q_len)
        total_kv = kv_cache_len + ctx_len + q_len

        flops = (
            2 * n_q * d["hidden"] * d["q_dim"]           # q_proj
            + 2 * n_kv_in * d["hidden"] * d["kv_dim"]    # k_proj on cat tokens
            + 2 * n_kv_in * d["hidden"] * d["kv_dim"]    # v_proj
            + 2 * n_q * d["q_dim"] * d["hidden"]         # o_proj
        )
        if d["bias"]:
            flops += (
                n_q * (d["q_dim"] + d["hidden"])
                + n_kv_in * (2 * d["kv_dim"])
            )
        # Non-causal SDPA over (past + ctx + noise)
        flops += 4 * batch * d["num_heads"] * q_len * total_kv * d["head_dim"]
        return flops

    def bytes(self, in_shape: tuple[int, ...], config, dtype: torch.dtype, **kwargs) -> int:
        kv_cache_len = int(kwargs.get("kv_cache_len", 0))
        ctx_len = int(kwargs.get("ctx_len", 0))
        batch, q_len = self._check_in_shape(in_shape)
        d = self._read_config(config)
        es = dtype_size(dtype)
        total_kv = kv_cache_len + ctx_len + q_len

        # Projection weights (same as plain qwen3, no gate)
        weights = (
            d["hidden"] * d["q_dim"] * es
            + 2 * d["hidden"] * d["kv_dim"] * es
            + d["q_dim"] * d["hidden"] * es
            + 2 * d["head_dim"] * es   # q_norm + k_norm gain
        )
        if d["bias"]:
            weights += (d["q_dim"] + 2 * d["kv_dim"] + d["hidden"]) * es

        # Activations
        noise_in = batch * q_len * d["hidden"] * es
        ctx_in = batch * ctx_len * d["hidden"] * es
        act_out = batch * q_len * d["hidden"] * es
        # KV in-flight: K and V both materialized at full non-causal length
        kv_act = batch * 2 * d["num_kv_heads"] * total_kv * d["head_dim"] * es

        return weights + noise_in + ctx_in + act_out + kv_act


# Register so it's discoverable via get_spec("dflash_attention") and usable
# from layer_types in custom configs.
register_spec("dflash_attention", DFlashAttentionSpec())


# --------------------------------------------------------------------------- #
# DSV4-style DFlash draft attention spec
# --------------------------------------------------------------------------- #


class DSV4DFlashAttentionSpec(RooflineSpec):
    r"""DFlash draft attention with DSV4-style projections + HCA/CSA/sliding
    effective-KV truncation.

    Combines:
      * **DFlash semantics** (matches ``DFlashAttentionSpec``):
          - Q projected on the noise block (Q tokens) only.
          - K/V projected on ``cat([target_hidden, noise])`` = (C + Q) tokens.
          - Non-causal SDPA: every noise query sees every available K.
          - O projected on noise (Q tokens) only.
      * **DSV4 projection structure** (matches ``DSV4AttentionSpec``):
          - Q-LoRA (``q_a_proj`` ``q_b_proj``) via ``q_lora_rank``.
          - MQA single KV head (``num_key_value_heads=1``, large ``head_dim``).
          - Grouped O-LoRA (``o_a_proj`` block-diagonal + ``o_b_proj``)
            via ``o_groups`` ``o_lora_rank``.
      * **HCA/CSA/sliding effective KV truncation** over total seq
        ``T = past + ctx + Q``:
          - ``dsv4_sliding_attention``: kv_eff = min(T, sliding_window)
          - ``dsv4_hca_attention``    : kv_eff = SW + ceil(T / rate_h)
          - ``dsv4_csa_attention``    : kv_eff = SW + index_topk
      * **HCA compressor / CSA compressor + indexer** charged per layer type.

    The SDPA term uses ``kv_eff`` (truncated) for both FLOPs and KV materialized
    bytes — same FlashAttention-style fusion assumption as the rest of alloy.

    Variables: B = batch, Q = q_len (noise block), C = ctx_len (target_hidden
    context length), P = kv_cache_len (accumulated accepted prefix in the
    draft cache), T = P + C + Q, H = hidden_size, qlr = q_lora_rank,
    nh = num_attention_heads, hd = head_dim, og = o_groups, olr = o_lora_rank,
    SW = min(T, sliding_window), nih = index_n_heads, ihd = index_head_dim,
    rate_h = compress_rates["heavily_compressed_attention"],
    rate_c = compress_rates["compressed_sparse_attention"],
    es = dtype_size(dtype).

    ## FLOPs
        2 * B * Q * H * qlr                  # q_a_proj (down)         [Q noise]
      + 2 * B * Q * qlr * (nh * hd)          # q_b_proj (up)           [Q noise]
      + 2 * B * (C+Q) * H * hd               # kv_proj (MQA single)    [ctx+noise]
      + 2 * B * Q * (nh * hd) * olr          # o_a_proj (grouped)      [Q noise]
      + 2 * B * Q * (og * olr) * H           # o_b_proj                [Q noise]
      + 4 * B * nh * Q * kv_eff * hd         # SDPA (Q × kv_eff truncated)
      [HCA extra]
        + 4 * B * (C+Q) * H * hd             # compressor: kv_proj + gate_proj
                                                on the newly projected (C+Q) tokens
      [CSA extra]
        + 4 * B * (C+Q) * H * (2*hd)         # compressor (2*hd output)
        + 4 * B * Q * H * (2*ihd)            # indexer: kv_proj + gate_proj on Q
        + 2 * B * Q * qlr * (nih*ihd)        # indexer: q_b_proj on Q
        + 2 * B * Q * H * nih                # indexer: weights_proj on Q
        + 2 * B * Q * nih * ihd * ceil(T/rate_c)   # indexer: score matmul

    ## Bytes (FA-style fusion; single MQA head materialized at kv_eff)
        H*qlr*es + qlr*nh*hd*es + H*hd*es + nh*hd*olr*es + og*olr*H*es + nh*es
            # Q/KV/O projection weights + per-head sinks
      + B*Q*H*es                             # noise activation read
      + B*C*H*es                             # ctx (target_hidden) activation read
      + B*Q*H*es                             # output write (noise only)
      + B*1*kv_eff*hd*es                     # KV buffer at effective length
      [HCA extra weights]
        + 2*H*hd*es + rate_h*hd*es + hd*es
      [CSA extra weights]
        + 2*H*(2*hd)*es + rate_c*(2*hd)*es + hd*es        # compressor
        + 2*H*(2*ihd)*es + qlr*nih*ihd*es + H*nih*es
            + rate_c*(2*ihd)*es + ihd*es                  # indexer
    """

    _LAYER_TYPES = {
        "dsv4_sliding_attention",
        "dsv4_hca_attention",
        "dsv4_csa_attention",
    }

    def __init__(self, layer_type: str) -> None:
        if layer_type not in self._LAYER_TYPES:
            raise ValueError(
                f"unknown DSV4 layer type for DFlash draft: {layer_type!r}; "
                f"expected one of {sorted(self._LAYER_TYPES)}"
            )
        self.layer_type = layer_type

    def _check_in_shape(self, in_shape: tuple[int, ...]) -> tuple[int, int]:
        if len(in_shape) != 3:
            raise ValueError(
                f"DSV4DFlashAttentionSpec expects 3D in_shape (batch, q_len, hidden); "
                f"got {in_shape}."
            )
        return in_shape[0], in_shape[1]

    def _kv_eff(self, q_len: int, ctx_len: int, kv_cache_len: int, config) -> int:
        total = q_len + ctx_len + kv_cache_len
        sw = min(total, config.sliding_window)
        if self.layer_type == "dsv4_sliding_attention":
            return sw
        if self.layer_type == "dsv4_hca_attention":
            rate_h = config.compress_rates["heavily_compressed_attention"]
            return sw + math.ceil(total / rate_h)
        # csa
        return sw + config.index_topk

    def flops(self, in_shape: tuple[int, ...], config, **kwargs) -> int:
        kv_cache_len = int(kwargs.get("kv_cache_len", 0))
        ctx_len = int(kwargs.get("ctx_len", 0))
        batch, q_len = self._check_in_shape(in_shape)

        hidden = config.hidden_size
        qlr = config.q_lora_rank
        nh = config.num_attention_heads
        hd = config.head_dim
        og = config.o_groups
        olr = config.o_lora_rank

        n_q = batch * q_len
        n_kv_in = batch * (ctx_len + q_len)
        total = q_len + ctx_len + kv_cache_len
        kv_eff = self._kv_eff(q_len, ctx_len, kv_cache_len, config)

        # Q-LoRA on Q noise + KV proj (MQA single head) on ctx+Q + O-LoRA on Q
        flops = (
            2 * n_q * hidden * qlr                # q_a_proj
            + 2 * n_q * qlr * (nh * hd)           # q_b_proj
            + 2 * n_kv_in * hidden * hd           # kv_proj on (ctx+Q)
            + 2 * n_q * (nh * hd) * olr           # o_a_proj
            + 2 * n_q * (og * olr) * hidden       # o_b_proj
        )
        # Non-causal SDPA truncated to kv_eff (HCA/CSA/sliding rule)
        flops += 4 * batch * nh * q_len * kv_eff * hd

        if self.layer_type == "dsv4_hca_attention":
            # HCA compressor: kv_proj + gate_proj on (ctx+Q) tokens
            flops += 4 * (ctx_len + q_len) * hidden * hd
        elif self.layer_type == "dsv4_csa_attention":
            # CSA compressor: kv_proj + gate_proj on (ctx+Q) tokens, 2*hd out
            flops += 4 * (ctx_len + q_len) * hidden * (2 * hd)
            # Indexer
            nih = config.index_n_heads
            ihd = config.index_head_dim
            rate_c = config.compress_rates["compressed_sparse_attention"]
            flops += 4 * n_q * hidden * (2 * ihd)    # indexer kv+gate on Q
            flops += 2 * n_q * qlr * (nih * ihd)     # indexer q_b_proj on Q
            flops += 2 * n_q * hidden * nih          # indexer weights_proj on Q
            flops += 2 * batch * q_len * nih * ihd * math.ceil(total / rate_c)
        return flops

    def bytes(self, in_shape: tuple[int, ...], config, dtype: torch.dtype, **kwargs) -> int:
        kv_cache_len = int(kwargs.get("kv_cache_len", 0))
        ctx_len = int(kwargs.get("ctx_len", 0))
        batch, q_len = self._check_in_shape(in_shape)
        es = dtype_size(dtype)

        hidden = config.hidden_size
        qlr = config.q_lora_rank
        nh = config.num_attention_heads
        hd = config.head_dim
        og = config.o_groups
        olr = config.o_lora_rank
        kv_eff = self._kv_eff(q_len, ctx_len, kv_cache_len, config)

        # Q/KV/O projection weights + per-head sinks
        weights = (
            hidden * qlr * es              # q_a_proj
            + qlr * nh * hd * es           # q_b_proj
            + hidden * hd * es             # kv_proj (MQA single head)
            + nh * hd * olr * es           # o_a_proj
            + og * olr * hidden * es       # o_b_proj
            + nh * es                      # per-head sinks
        )
        if self.layer_type == "dsv4_hca_attention":
            weights += (
                2 * hidden * hd * es
                + config.compress_rates["heavily_compressed_attention"] * hd * es
                + hd * es
            )
        elif self.layer_type == "dsv4_csa_attention":
            rate_c = config.compress_rates["compressed_sparse_attention"]
            nih = config.index_n_heads
            ihd = config.index_head_dim
            # compressor
            weights += (
                2 * hidden * (2 * hd) * es
                + rate_c * (2 * hd) * es
                + hd * es
            )
            # indexer
            weights += (
                2 * hidden * (2 * ihd) * es
                + qlr * nih * ihd * es
                + hidden * nih * es
                + rate_c * (2 * ihd) * es
                + ihd * es
            )

        # Activations (DFlash convention: noise + ctx read, noise write)
        noise_in = batch * q_len * hidden * es
        ctx_in = batch * ctx_len * hidden * es
        act_out = batch * q_len * hidden * es
        # KV buffer materialized at effective length (single MQA head)
        kv_act = batch * 1 * kv_eff * hd * es

        return weights + noise_in + ctx_in + act_out + kv_act


# Register so it's discoverable via get_spec(...) and usable from layer_types.
register_spec("dsv4_sliding_dflash_attention", DSV4DFlashAttentionSpec("dsv4_sliding_attention"))
register_spec("dsv4_hca_dflash_attention", DSV4DFlashAttentionSpec("dsv4_hca_attention"))
register_spec("dsv4_csa_dflash_attention", DSV4DFlashAttentionSpec("dsv4_csa_attention"))


__all__ = ["DFlashAttentionSpec", "DSV4DFlashAttentionSpec"]
