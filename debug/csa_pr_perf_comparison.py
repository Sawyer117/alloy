"""Final perf-equivalence comparison: PR #45879 vs. perf follow-up patch.

Reproduces both DeepSeek-V4 CSA per-query-mask implementations inline so a
reviewer can run a single script to verify the equivalence + speedup claims.

  * arthur:  gather-then-block-bias path from PR #45879
             ``return gathered, block_bias`` -> attention runs over [B,1,S*k,D] KV
  * ours:    scatter-bias path (perf follow-up)
             ``return compressed_kv, scatter_bias`` -> attention runs over [B,1,T,D] KV

Each path computes the full per-query CSA attention with per-head sinks. We
compare the final attention output (the math both paths produce).

Run::

    python alloy/debug/csa_pr_perf_comparison.py
"""
from __future__ import annotations

import time

import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Arthur PR #45879: gather + block-diagonal bias + dense attention
# --------------------------------------------------------------------------- #


def arthur_attention_output(
    Q: torch.Tensor,
    compressed_kv: torch.Tensor,
    topk_idxs: torch.Tensor,
    attn_sink: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    B, H, T_q, D = Q.shape
    K = topk_idxs.shape[-1]

    # 1. Gather (exactly as in PR #45879)
    valid = topk_idxs >= 0
    safe_topk = topk_idxs.clamp(min=0)
    expanded = compressed_kv.unsqueeze(2).expand(-1, -1, T_q, -1, -1)
    idx = safe_topk.unsqueeze(1).unsqueeze(-1).expand(-1, 1, -1, -1, D)
    gathered = torch.gather(expanded, 3, idx).reshape(B, 1, -1, D)  # [B, 1, S*k, D]

    # 2. Block-diagonal bias
    block_bias = gathered.new_full((B, 1, T_q, T_q, K), float("-inf"))
    allowed = torch.where(valid, gathered.new_zeros(()), gathered.new_full((), float("-inf")))
    arange_s = torch.arange(T_q, device=gathered.device)
    block_bias[:, 0, arange_s, arange_s, :] = allowed
    block_bias = block_bias.view(B, 1, T_q, T_q * K)

    # 3. Dense attention + sink
    attn_weights = torch.matmul(Q, gathered.transpose(-1, -2)) * scale
    attn_weights = attn_weights + block_bias
    sinks = attn_sink.reshape(1, -1, 1, 1).expand(B, -1, T_q, -1)
    combined_logits = torch.cat([attn_weights, sinks], dim=-1)
    combined_logits = combined_logits - combined_logits.max(dim=-1, keepdim=True).values
    probs = F.softmax(combined_logits, dim=-1, dtype=combined_logits.dtype)
    scores = probs[..., :-1]
    return torch.matmul(scores, gathered)


# --------------------------------------------------------------------------- #
# Our perf follow-up: scatter bias on compressed axis, no gather
# --------------------------------------------------------------------------- #


def ours_attention_output(
    Q: torch.Tensor,
    compressed_kv: torch.Tensor,
    topk_idxs: torch.Tensor,
    attn_sink: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    B, H, T_q, D = Q.shape
    T = compressed_kv.shape[2]

    # 1. Scatter bias: route invalid (-1) sentinels to throwaway column `T`
    valid = topk_idxs >= 0
    safe_topk = torch.where(valid, topk_idxs, torch.full_like(topk_idxs, T))
    compressed_bias = compressed_kv.new_full((B, 1, T_q, T + 1), float("-inf"))
    compressed_bias.scatter_(-1, safe_topk.unsqueeze(1), 0.0)
    compressed_bias = compressed_bias[..., :T]  # drop sentinel column

    # 2. Dense attention + sink (over [T] compressed entries, not [S*k])
    attn_weights = torch.matmul(Q, compressed_kv.transpose(-1, -2)) * scale
    attn_weights = attn_weights + compressed_bias
    sinks = attn_sink.reshape(1, -1, 1, 1).expand(B, -1, T_q, -1)
    combined_logits = torch.cat([attn_weights, sinks], dim=-1)
    combined_logits = combined_logits - combined_logits.max(dim=-1, keepdim=True).values
    probs = F.softmax(combined_logits, dim=-1, dtype=combined_logits.dtype)
    scores = probs[..., :-1]
    return torch.matmul(scores, compressed_kv)


# --------------------------------------------------------------------------- #
# Test harness
# --------------------------------------------------------------------------- #


def make_inputs(B, T_q, H, D, T_compressed, K, dtype, seed):
    gen = torch.Generator().manual_seed(seed)
    Q = torch.randn(B, H, T_q, D, dtype=dtype, generator=gen)
    compressed_kv = torch.randn(B, 1, T_compressed, D, dtype=dtype, generator=gen)
    topk_idxs = torch.empty(B, T_q, K, dtype=torch.long)
    for b in range(B):
        for t in range(T_q):
            topk_idxs[b, t] = torch.randperm(T_compressed, generator=gen)[:K]
    attn_sink = torch.zeros(H, dtype=dtype)
    return Q, compressed_kv, topk_idxs, attn_sink, D ** -0.5


CONFIGS = [
    ("small",        dict(B=2, T_q=8,   H=4, D=16, T_compressed=64,   K=4)),
    ("V4-Pro-ish",   dict(B=1, T_q=64,  H=8, D=64, T_compressed=512,  K=32)),
    ("long ctx",     dict(B=1, T_q=256, H=8, D=64, T_compressed=1024, K=64)),
    ("larger",       dict(B=1, T_q=512, H=8, D=64, T_compressed=2048, K=128)),
]


def numerical_compare(name, cfg, seeds, dtype):
    print(f"\n--- {name} | {dtype} ---")
    print(f"   shape: B={cfg['B']} T_q={cfg['T_q']} H={cfg['H']} D={cfg['D']} "
          f"T_compressed={cfg['T_compressed']} K={cfg['K']}")
    print(f"   {'seed':>4}  {'max_abs':>11}  {'rel_max':>8}  {'cos_sim':>10}")
    for seed in seeds:
        Q, kv, topk, sink, scale = make_inputs(**cfg, dtype=dtype, seed=seed)
        out_a = arthur_attention_output(Q, kv, topk, sink, scale)
        out_o = ours_attention_output(Q, kv, topk, sink, scale)
        diff = (out_a.float() - out_o.float()).abs()
        ref_max = out_a.abs().float().max().item()
        rel = (diff.max().item() / ref_max * 100) if ref_max > 0 else 0.0
        cos_sim = F.cosine_similarity(
            out_a.float().flatten(-2), out_o.float().flatten(-2), dim=-1,
        ).mean().item()
        print(f"   {seed:>4}  {diff.max().item():>11.4e}  {rel:>7.4f}%  {cos_sim:>10.6f}")


def speed_benchmark(cfg, dtype, n_warmup, n_iter):
    Q, kv, topk, sink, scale = make_inputs(**cfg, dtype=dtype, seed=0)
    for _ in range(n_warmup):
        arthur_attention_output(Q, kv, topk, sink, scale)
        ours_attention_output(Q, kv, topk, sink, scale)
    t0 = time.perf_counter()
    for _ in range(n_iter):
        arthur_attention_output(Q, kv, topk, sink, scale)
    t_a = (time.perf_counter() - t0) / n_iter
    t0 = time.perf_counter()
    for _ in range(n_iter):
        ours_attention_output(Q, kv, topk, sink, scale)
    t_o = (time.perf_counter() - t0) / n_iter
    return t_a, t_o


def main():
    # 1. Numerical equivalence in fp32 + bf16
    print("=" * 70)
    print("NUMERICAL EQUIVALENCE: PR #45879  vs.  perf follow-up")
    print("=" * 70)
    for dtype in [torch.float32, torch.bfloat16]:
        for name, c in CONFIGS[:3]:  # skip 'larger' for the numerical pass (slow)
            numerical_compare(name, c, seeds=[0, 1, 2], dtype=dtype)

    # 2. Speed benchmark
    print()
    print("=" * 70)
    print("SPEED BENCHMARK: CPU, single thread")
    print("=" * 70)
    for dtype in [torch.float32, torch.bfloat16]:
        print(f"\ndtype = {dtype}")
        print(f"  {'config':<12}  {'matmul ratio':>13}  "
              f"{'#45879 ms':>10}  {'ours ms':>10}  {'speedup':>9}")
        print(f"  {'-'*12}  {'-'*13}  {'-'*10}  {'-'*10}  {'-'*9}")
        for name, c in CONFIGS:
            n_iter = 10 if c["T_q"] >= 512 else 20
            ratio = (c["T_q"] * c["K"]) / c["T_compressed"]
            t_a, t_o = speed_benchmark(c, dtype=dtype, n_warmup=5, n_iter=n_iter)
            speedup = t_a / t_o
            print(f"  {name:<12}  {ratio:>11.1f}x   "
                  f"{t_a*1000:>9.2f}   {t_o*1000:>9.2f}   {speedup:>7.2f}x")


if __name__ == "__main__":
    main()
