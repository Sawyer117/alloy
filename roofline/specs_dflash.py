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


__all__ = ["DFlashAttentionSpec"]
