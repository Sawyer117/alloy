"""CSA implementation divergence - severity diagnostic.

Compares two standalone PyTorch implementations of DeepSeek-V4 CSA attention
to quantify how different they are. Both implementations take the same
inputs (Q, compressed_kv, topk_idxs, attn_sink, scale) and produce an
attention output; the question is by how much the outputs disagree.

  Path A - gather-then-dense  (HF transformers / alloy current path)
    1. gather: for each query t, materialize kv[topk_idxs[t]] into a flat
       [B, 1, T*K, D] tensor (each query's K entries laid out in a band)
    2. dense attention over all T*K entries (no per-query masking)
    3. add sink column, softmax, drop sink, @ gathered V

  Path B - per-query sparse  (MindSpeed-LLM torch fallback + Triton kernel)
    1. full Q · compressed_kv^T over all n_compressed positions
    2. apply mask: -inf except at each query's topk_idxs[t] positions
    3. add sink column, softmax, drop sink, @ compressed V

Companion doc: ``model_gym/csa_implementation_divergence.md``.

Run::

    python alloy/debug/csa_divergence_check.py
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Path A - HF / alloy: gather upstream, dense attention
# --------------------------------------------------------------------------- #


def path_a_hf_alloy(
    Q: torch.Tensor,            # [B, H, T, D]
    compressed_kv: torch.Tensor,# [B, 1, n_compressed, D]
    topk_idxs: torch.Tensor,    # [B, T, K]
    attn_sink: torch.Tensor,    # [H]
    scale: float,
) -> torch.Tensor:
    """HF transformers / alloy CSA path: gather then dense attention.

    Mirrors :class:`DeepseekV4CSACompressor.forward` + the right-pad-with-0
    attention_mask handling in :class:`DeepseekV4Attention.forward`.
    """
    B, H, T, D = Q.shape
    K = topk_idxs.shape[-1]

    # 1. Gather (alloy line 499-503 of dsv4_attention.py)
    expanded = compressed_kv.unsqueeze(2).expand(-1, -1, T, -1, -1)
    idx = topk_idxs.unsqueeze(1).unsqueeze(-1).expand(-1, 1, -1, -1, D)
    gathered_kv = torch.gather(expanded, 3, idx).reshape(B, 1, -1, D)
    # shape: [B, 1, T*K, D]; layout: queries' K entries in contiguous bands

    # 2. Dense attention (the right-pad-with-0 mask means no per-query gating)
    attn_weights = torch.matmul(Q, gathered_kv.transpose(-1, -2)) * scale
    # shape: [B, H, T, T*K]

    # 3. Sink column + row-max stabilization + softmax + drop sink + @ V
    sinks = attn_sink.reshape(1, -1, 1, 1).expand(B, -1, T, -1)
    combined_logits = torch.cat([attn_weights, sinks], dim=-1)
    combined_logits = combined_logits - combined_logits.max(dim=-1, keepdim=True).values
    probs = F.softmax(combined_logits, dim=-1, dtype=combined_logits.dtype)
    scores = probs[..., :-1]
    return torch.matmul(scores, gathered_kv)


# --------------------------------------------------------------------------- #
# Path B - MindSpeed-LLM: per-query sparse via topk-mask
# --------------------------------------------------------------------------- #


def path_a_pr_fixed(
    Q: torch.Tensor,
    compressed_kv: torch.Tensor,
    topk_idxs: torch.Tensor,
    attn_sink: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """HF PR #45879 fix: gather then dense attention with block-diagonal bias.

    Adds the per-query block-diagonal mask the PR introduces — query t can
    only attend to positions [t*K : (t+1)*K] of the gathered KV. Math now
    matches :func:`path_b_mindspeed`; compute does not (still O(T*T*K)
    matmul versus MindSpeed's O(T*n_compressed)).
    """
    B, H, T, D = Q.shape
    K = topk_idxs.shape[-1]

    # 1. Same gather as the pre-fix path
    expanded = compressed_kv.unsqueeze(2).expand(-1, -1, T, -1, -1)
    idx = topk_idxs.unsqueeze(1).unsqueeze(-1).expand(-1, 1, -1, -1, D)
    gathered_kv = torch.gather(expanded, 3, idx).reshape(B, 1, -1, D)  # [B, 1, T*K, D]

    # 2. Block-diagonal bias: 0 on the diagonal band, -inf elsewhere
    block_bias = torch.full(
        (B, 1, T, T, K), float("-inf"),
        device=Q.device, dtype=Q.dtype,
    )
    arange_t = torch.arange(T, device=Q.device)
    block_bias[:, 0, arange_t, arange_t, :] = 0.0
    block_bias = block_bias.view(B, 1, T, T * K)

    # 3. Dense attention masked by block-diagonal bias
    attn_weights = torch.matmul(Q, gathered_kv.transpose(-1, -2)) * scale
    attn_weights = attn_weights + block_bias

    # 4. Sink column + row-max stabilization + softmax + drop sink + @ V
    sinks = attn_sink.reshape(1, -1, 1, 1).expand(B, -1, T, -1)
    combined_logits = torch.cat([attn_weights, sinks], dim=-1)
    combined_logits = combined_logits - combined_logits.max(dim=-1, keepdim=True).values
    probs = F.softmax(combined_logits, dim=-1, dtype=combined_logits.dtype)
    scores = probs[..., :-1]
    return torch.matmul(scores, gathered_kv)


def path_b_mindspeed(
    Q: torch.Tensor,
    compressed_kv: torch.Tensor,
    topk_idxs: torch.Tensor,
    attn_sink: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """MindSpeed-LLM CSA path (``sparse_flash_attn`` torch fallback).

    Mirrors lines 34-82 of
    ``mindspeed_llm/tasks/models/transformer/deepseek4/g2_attention_kernel.py``.
    """
    B, H, T, D = Q.shape
    n_compressed = compressed_kv.shape[2]

    # 1. Full QK^T
    attn_weights = torch.matmul(Q, compressed_kv.transpose(-1, -2)) * scale
    # shape: [B, H, T, n_compressed]

    # 2. Per-query topk mask
    neg = torch.finfo(attn_weights.dtype).min
    index_mask = torch.full(
        (B, 1, T, n_compressed + 1),  # +1 column reserved for sink (kept at 0)
        neg, dtype=attn_weights.dtype, device=attn_weights.device,
    )
    index_mask.scatter_(-1, topk_idxs.unsqueeze(1), 0)
    attn_weights = attn_weights + index_mask[..., :-1]

    # 3. Sink column + row-max stabilization + softmax + drop sink + @ V
    sinks = attn_sink.reshape(1, -1, 1, 1).expand(B, -1, T, -1)
    combined_logits = torch.cat([attn_weights, sinks], dim=-1)
    combined_logits = combined_logits - combined_logits.max(dim=-1, keepdim=True).values
    probs = F.softmax(combined_logits, dim=-1, dtype=combined_logits.dtype)
    scores = probs[..., :-1]
    return torch.matmul(scores, compressed_kv)


# --------------------------------------------------------------------------- #
# Comparison harness
# --------------------------------------------------------------------------- #


def diff_stats(out_a: torch.Tensor, out_b: torch.Tensor) -> dict:
    diff = (out_a - out_b).abs()
    ref_max = out_a.abs().max().item()
    cos_sim = F.cosine_similarity(
        out_a.flatten(-2), out_b.flatten(-2), dim=-1,
    )
    return {
        "max_abs": diff.max().item(),
        "mean_abs": diff.mean().item(),
        "ref_max_abs": ref_max,
        "rel_max": (diff.max().item() / ref_max) if ref_max > 0 else float("inf"),
        "cos_sim_min": cos_sim.min().item(),
        "cos_sim_mean": cos_sim.mean().item(),
    }


def make_random_topk(B: int, T: int, K: int, n_compressed: int, generator) -> torch.Tensor:
    """Each query picks K distinct indices from [0, n_compressed)."""
    out = torch.empty(B, T, K, dtype=torch.long)
    for b in range(B):
        for t in range(T):
            out[b, t] = torch.randperm(n_compressed, generator=generator)[:K]
    return out


def run_case(
    name: str,
    *,
    B: int, T: int, H: int, D: int, n_compressed: int, K: int,
    seeds: list[int],
    dtype: torch.dtype = torch.float32,
    topk_builder=None,
) -> None:
    print(f"\n=== {name} ===")
    print(f"   B={B}  T={T}  H={H}  D={D}  n_compressed={n_compressed}  K={K}  dtype={dtype}")
    print(f"   {'seed':>4}  {'max_abs':>10}  {'rel_max':>8}  {'cos_sim_min':>12}  {'cos_sim_mean':>13}")

    for seed in seeds:
        gen = torch.Generator().manual_seed(seed)
        Q = torch.randn(B, H, T, D, dtype=dtype, generator=gen)
        compressed_kv = torch.randn(B, 1, n_compressed, D, dtype=dtype, generator=gen)
        if topk_builder is None:
            topk_idxs = make_random_topk(B, T, K, n_compressed, gen)
        else:
            topk_idxs = topk_builder(B, T, K, n_compressed, gen)
        attn_sink = torch.zeros(H, dtype=dtype)
        scale = D ** -0.5

        out_a = path_a_hf_alloy(Q, compressed_kv, topk_idxs, attn_sink, scale)
        out_b = path_b_mindspeed(Q, compressed_kv, topk_idxs, attn_sink, scale)
        s = diff_stats(out_a, out_b)
        print(f"   {seed:>4}  {s['max_abs']:>10.4e}  {s['rel_max']*100:>7.2f}%  "
              f"{s['cos_sim_min']:>12.6f}  {s['cos_sim_mean']:>13.6f}")


def run_equivalence_pr_vs_mindspeed(
    *,
    B: int, T: int, H: int, D: int, n_compressed: int, K: int,
    seeds: list[int], dtype: torch.dtype = torch.float32,
) -> None:
    """PR #45879 fix vs MindSpeed-LLM should be byte-exact in fp32.

    Both compute the same math (per-query sparse attention with sinks);
    they differ only in how much compute they spend getting there.
    """
    print(f"\n=== PR #45879 fix  vs  MindSpeed-LLM (numerical equivalence) ===")
    print(f"   B={B}  T={T}  H={H}  D={D}  n_compressed={n_compressed}  K={K}  dtype={dtype}")
    print(f"   {'seed':>4}  {'max_abs':>10}  {'rel_max':>8}  {'cos_sim_min':>12}")
    for seed in seeds:
        gen = torch.Generator().manual_seed(seed)
        Q = torch.randn(B, H, T, D, dtype=dtype, generator=gen)
        compressed_kv = torch.randn(B, 1, n_compressed, D, dtype=dtype, generator=gen)
        topk_idxs = make_random_topk(B, T, K, n_compressed, gen)
        attn_sink = torch.zeros(H, dtype=dtype)
        scale = D ** -0.5

        out_pr = path_a_pr_fixed(Q, compressed_kv, topk_idxs, attn_sink, scale)
        out_ms = path_b_mindspeed(Q, compressed_kv, topk_idxs, attn_sink, scale)
        s = diff_stats(out_pr, out_ms)
        print(f"   {seed:>4}  {s['max_abs']:>10.4e}  {s['rel_max']*100:>7.2f}%  "
              f"{s['cos_sim_min']:>12.6f}")


def benchmark_speed(
    *, B: int, T: int, H: int, D: int, n_compressed: int, K: int,
    n_warmup: int = 5, n_iter: int = 20, dtype: torch.dtype = torch.float32,
) -> None:
    """Compute-only speed comparison on CPU (fp32, no GPU).

    The matmul sizes in QK^T tell the story:
      PR fix : Q @ gathered_kv^T -> [B, H, T, T*K]   compute O(B*H*T*T*K*D)
      MindSpd: Q @ compressed_kv^T -> [B, H, T, n_comp]  compute O(B*H*T*n_comp*D)

    Ratio = (T*K) / n_compressed.  Realistic V4-Pro scale gives ~thousands.
    """
    import time

    gen = torch.Generator().manual_seed(0)
    Q = torch.randn(B, H, T, D, dtype=dtype, generator=gen)
    compressed_kv = torch.randn(B, 1, n_compressed, D, dtype=dtype, generator=gen)
    topk_idxs = make_random_topk(B, T, K, n_compressed, gen)
    attn_sink = torch.zeros(H, dtype=dtype)
    scale = D ** -0.5

    print(f"\n=== Speed benchmark (CPU, fp32) ===")
    print(f"   B={B}  T={T}  H={H}  D={D}  n_compressed={n_compressed}  K={K}")
    print(f"   matmul ratio (PR fix / MindSpeed) = T*K / n_compressed = "
          f"{T*K}/{n_compressed} = {(T*K)/n_compressed:.1f}x")

    # Warmup
    for _ in range(n_warmup):
        path_a_pr_fixed(Q, compressed_kv, topk_idxs, attn_sink, scale)
        path_b_mindspeed(Q, compressed_kv, topk_idxs, attn_sink, scale)

    # Measure PR fix
    t0 = time.perf_counter()
    for _ in range(n_iter):
        path_a_pr_fixed(Q, compressed_kv, topk_idxs, attn_sink, scale)
    t_pr = (time.perf_counter() - t0) / n_iter

    # Measure MindSpeed
    t0 = time.perf_counter()
    for _ in range(n_iter):
        path_b_mindspeed(Q, compressed_kv, topk_idxs, attn_sink, scale)
    t_ms = (time.perf_counter() - t0) / n_iter

    speedup = t_pr / t_ms if t_ms > 0 else float("inf")
    print(f"   PR fix     : {t_pr*1000:>8.3f} ms/call  ({n_iter} iters avg)")
    print(f"   MindSpeed  : {t_ms*1000:>8.3f} ms/call  ({n_iter} iters avg)")
    print(f"   speedup    : {speedup:>8.2f}x in favor of MindSpeed")


def main() -> int:
    # ---------- Sanity checks: paths should AGREE in degenerate cases --- #

    # Same topk across all queries -> gather-flatten == compressed_kv (rearranged),
    # and both paths see the same K entries per query. Should agree.
    def same_topk(B, T, K, n_compressed, gen):
        common = torch.randperm(n_compressed, generator=gen)[:K]
        return common.unsqueeze(0).unsqueeze(0).expand(B, T, K).contiguous()

    run_case("Sanity: all queries share same topk (paths should agree)",
             B=2, T=8, H=4, D=16, n_compressed=64, K=4,
             seeds=[0, 1, 2], topk_builder=same_topk)

    # K == n_compressed: no actual sparsity. Both paths attend to everything.
    def full_topk(B, T, K, n_compressed, gen):
        return torch.arange(n_compressed).unsqueeze(0).unsqueeze(0).expand(B, T, K).contiguous()

    run_case("Sanity: K = n_compressed (no sparsity, paths should agree)",
             B=2, T=8, H=4, D=16, n_compressed=32, K=32,
             seeds=[0, 1, 2], topk_builder=full_topk)

    # ---------- Real test: random distinct topk per query (worst case) --- #

    run_case("Random topk per query - small (worst case, paths should disagree)",
             B=2, T=8, H=4, D=16, n_compressed=64, K=4,
             seeds=[0, 1, 2])

    run_case("Random topk per query - V4-Pro-ish scaled (T=64)",
             B=1, T=64, H=8, D=64, n_compressed=512, K=32,
             seeds=[0, 1, 2])

    run_case("Random topk per query - long context (T=256, K=64)",
             B=1, T=256, H=8, D=64, n_compressed=1024, K=64,
             seeds=[0, 1, 2])

    # ---------- PR #45879 fix vs MindSpeed: numerical equivalence ------- #
    run_equivalence_pr_vs_mindspeed(B=2, T=8, H=4, D=16, n_compressed=64, K=4,
                                    seeds=[0, 1, 2])
    run_equivalence_pr_vs_mindspeed(B=1, T=64, H=8, D=64, n_compressed=512, K=32,
                                    seeds=[0, 1, 2])
    run_equivalence_pr_vs_mindspeed(B=1, T=256, H=8, D=64, n_compressed=1024, K=64,
                                    seeds=[0, 1, 2])

    # ---------- Speed benchmark ----------------------------------------- #
    # Small V4-ish: PR fix ratio = T*K/n_compressed = 64*32/512 = 4x
    benchmark_speed(B=1, T=64, H=8, D=64, n_compressed=512, K=32)
    # Longer V4-ish: ratio = 256*64/1024 = 16x
    benchmark_speed(B=1, T=256, H=8, D=64, n_compressed=1024, K=64)
    # Bigger still — keep within CPU budget but show realistic scale
    benchmark_speed(B=1, T=512, H=8, D=64, n_compressed=2048, K=128, n_iter=10)

    print()
    print("INTERPRETATION:")
    print("  - Sanity scenarios (same topk / K = n_compressed) should give max_abs ~ 0 (< 1e-6)")
    print("    and cos_sim ~ 1.0. If not, one of the paths has a bug.")
    print("  - Random-topk scenarios reveal the architectural divergence:")
    print("    * cos_sim << 1.0  -> outputs point in different directions (different math)")
    print("    * rel_max ~ tens of %  -> magnitude-significant divergence,")
    print("      load-checkpoint-and-SFT would diverge")
    print("    * cos_sim ~ 1.0 but rel_max significant -> directions agree, magnitudes off")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
