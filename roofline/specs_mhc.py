"""Roofline specs for the MHC (Manifold-constrained Hyper-Connections) machinery.

MHC is a multi-stream residual flow scheme introduced by DeepSeek-V4
(paper §2.2) and implemented in alloy by ``AlloyMhcDecoderLayer``. When
``config.use_mhc=True`` each decoder layer carries ``hc_mult`` parallel
residual streams instead of one; two ``_HyperConnection`` modules at the
mixer and FFN sublayer sites collapse the streams down for the
sublayer and place its output back across the streams, and a single
``_HyperHead`` at the model tail reduces back to one stream before the
final RMSNorm + lm_head.

The mixer / FFN modules themselves see a single-stream input and don't
know MHC exists — that's the design point that makes MHC ablatable
across any alloy mixer-FFN combination. So **adding MHC to the
roofline is additive**: same per-layer mixer / FFN cost, plus the HC
machinery this file accounts for.

Costs are non-trivial at V4-Pro scale: with ``hc_mult=4`` and
``hc_sinkhorn_iters=20``, each ``_HyperConnection`` does

  * RMSNorm on a ``[N, hc*H]`` tensor (the flattened hc_mult streams)
  * one big ``[N, hc*H] @ [(2+hc)*hc, hc*H]`` linear (the ``fn`` mix
    matrix that emits ``pre``/``post``/``comb`` logits in one shot)
  * 20 Sinkhorn-Knopp iterations on a ``[N, hc, hc]`` mixing matrix
  * a weighted-sum collapse over the stream axis

The ``fn`` weight alone is ``hc_mult^2 * (2 + hc_mult) * hidden_size``
parameters per HyperConnection — for V4-Pro hc=4: 24 * 4 * 7168 ≈ 688K
params × 2 per layer × 61 layers ≈ 84M params total. Not huge as a
fraction of the 671B-param model, but the per-token bytes traffic
adds up (re-read every forward).

Specs here are NOT registered into the global ``_REGISTRY`` because
they aren't dispatched via ``config.layer_types`` / ``config.ffn_types``
names; they're inserted by ``analyze.py`` directly when
``config.use_mhc=True``.
"""
from __future__ import annotations

import torch

from .specs import RooflineSpec, dtype_size


_INPUT_NORM_FLOPS_PER_ELEM = 4   # match RMSNormSpec's per-element approximation
_SIGMOID_FLOPS_PER_ELEM = 4      # exp + add + div + scalar-mul-bias (per-elem)


class MhcHyperConnectionSpec(RooflineSpec):
    """One ``_HyperConnection`` site (pre-mixer or pre-ffn).

    Variables: B = batch, S = query_len, N = B * S, hc = hc_mult,
    H = hidden_size, K = hc_sinkhorn_iters, es = dtype_size(dtype).

    The MHC code path runs the norm + ``fn`` linear in fp32 (see
    ``input_norm(x.flatten(...).float())`` in
    ``modeling_alloy.py:_HyperConnection.forward``). Bytes pessimistically
    assume the fp32 promotion materialises an intermediate ``[N, hc*H]``
    tensor in fp32 (4 bytes/elem). FLOPs are dtype-agnostic.

    ## FLOPs
        4 * N * hc * H                  # input_norm (RMSNorm on flattened [N, hc*H])
      + 2 * N * (hc*H) * ((2+hc)*hc)    # ``fn`` linear: emits pre/post/comb logits at once
      + 4 * N * (2+hc) * hc             # sigmoid + scale + bias on the mix output
      + 4 * K * N * hc * hc             # Sinkhorn-Knopp: K iters * 2 (row + col) * sum-div per [N, hc, hc]
      + 2 * N * hc * H                  # collapse: elementwise mul + reduce-sum over the stream axis

    ## Bytes (optimal fusion, fp32-promoted intermediates)
        (2 + hc) * hc * (hc * H) * es   # ``fn`` weight: shape [(2+hc)*hc, hc*H]
      +     (2 + hc) * hc * es          # ``base`` bias vector
      +     3 * es                      # 3 scalar scales (pre/post/comb)
      +     N * hc * H * es             # hidden_streams read (multi-stream input)
      +     N * H * es                  # collapsed output write
      + 2 * N * hc * H * 4              # fp32 promotion: norm read+write @ 4 B/elem
    """

    def flops(self, in_shape: tuple[int, ...], config, **kwargs) -> int:
        n_tokens = 1
        for d in in_shape[:-1]:
            n_tokens *= d
        hidden = in_shape[-1]
        hc = config.hc_mult
        sinkhorn_iters = getattr(config, "hc_sinkhorn_iters", 4)

        # 1. RMSNorm on [N, hc*H]
        norm_flops = _INPUT_NORM_FLOPS_PER_ELEM * n_tokens * hc * hidden
        # 2. ``fn`` linear: [N, hc*H] @ [hc*H, (2+hc)*hc]
        fn_linear_flops = 2 * n_tokens * (hc * hidden) * ((2 + hc) * hc)
        # 3. sigmoid + scale + bias on the (2+hc)*hc mix output
        sigmoid_flops = _SIGMOID_FLOPS_PER_ELEM * n_tokens * (2 + hc) * hc
        # 4. Sinkhorn-Knopp: K iters, each 2 norms (row + col), each ~hc*hc sum-divides
        sinkhorn_flops = 4 * sinkhorn_iters * n_tokens * hc * hc
        # 5. Collapse: (pre[:, :, :, None] * streams).sum(dim=2) = mul + reduce-sum
        collapse_flops = 2 * n_tokens * hc * hidden

        return (
            norm_flops + fn_linear_flops + sigmoid_flops
            + sinkhorn_flops + collapse_flops
        )

    def bytes(self, in_shape: tuple[int, ...], config, dtype: torch.dtype, **kwargs) -> int:
        n_tokens = 1
        for d in in_shape[:-1]:
            n_tokens *= d
        hidden = in_shape[-1]
        hc = config.hc_mult
        es = dtype_size(dtype)
        FP32 = 4

        # ``fn`` weight [(2+hc)*hc, hc*H] — the dominant weight term
        fn_w = (2 + hc) * hc * (hc * hidden) * es
        # ``base`` (per-output bias) + 3 scalar scales
        small_params = ((2 + hc) * hc + 3) * es
        # Activations: input multi-stream read, single-stream output write
        act_in = n_tokens * hc * hidden * es
        act_out = n_tokens * hidden * es
        # fp32 promotion overhead: norm reads + writes a fp32 intermediate
        promote_io = 2 * n_tokens * hc * hidden * FP32

        return fn_w + small_params + act_in + act_out + promote_io


class MhcHyperHeadSpec(RooflineSpec):
    """The final ``_HyperHead`` collapse: reduces ``[B, S, hc, H]`` -> ``[B, S, H]``.

    Single-shot replacement for the very last residual; happens once per
    forward (NOT per layer).

    Variables: B = batch, S = query_len, N = B * S, hc = hc_mult,
    H = hidden_size, es = dtype_size(dtype). Same fp32-promotion
    convention as MhcHyperConnectionSpec.

    ## FLOPs
        4 * N * hc * H                  # input_norm
      + 2 * N * (hc*H) * hc             # hc_fn linear: emits ``pre`` weights only
      + 4 * N * hc                      # sigmoid + scale + bias
      + 2 * N * hc * H                  # collapse: elementwise mul + reduce-sum

    ## Bytes
        hc * (hc * H) * es              # hc_fn weight: [hc, hc*H]
      + hc * es                         # hc_base bias
      +     es                          # 1 scalar scale
      + N * hc * H * es                 # multi-stream input read
      + N * H * es                      # single-stream output write
      + 2 * N * hc * H * 4              # fp32 norm promotion overhead
    """

    def flops(self, in_shape: tuple[int, ...], config, **kwargs) -> int:
        n_tokens = 1
        for d in in_shape[:-1]:
            n_tokens *= d
        hidden = in_shape[-1]
        hc = config.hc_mult
        norm_flops = _INPUT_NORM_FLOPS_PER_ELEM * n_tokens * hc * hidden
        hc_fn_flops = 2 * n_tokens * (hc * hidden) * hc
        sigmoid_flops = _SIGMOID_FLOPS_PER_ELEM * n_tokens * hc
        collapse_flops = 2 * n_tokens * hc * hidden
        return norm_flops + hc_fn_flops + sigmoid_flops + collapse_flops

    def bytes(self, in_shape: tuple[int, ...], config, dtype: torch.dtype, **kwargs) -> int:
        n_tokens = 1
        for d in in_shape[:-1]:
            n_tokens *= d
        hidden = in_shape[-1]
        hc = config.hc_mult
        es = dtype_size(dtype)
        FP32 = 4
        hc_fn_w = hc * (hc * hidden) * es
        small_params = (hc + 1) * es
        act_in = n_tokens * hc * hidden * es
        act_out = n_tokens * hidden * es
        promote_io = 2 * n_tokens * hc * hidden * FP32
        return hc_fn_w + small_params + act_in + act_out + promote_io


__all__ = ["MhcHyperConnectionSpec", "MhcHyperHeadSpec"]
