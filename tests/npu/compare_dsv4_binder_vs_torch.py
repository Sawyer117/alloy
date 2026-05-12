"""Binder ON vs OFF for DeepSeek-V4 CSA — precision + speed on alloy.

DSV4-specific sibling of ``compare_binder_vs_torch.py``. Builds a small
alloy DSV4 model with at least one ``dsv4_csa_attention`` layer (the
only DSV4 attention flavour the binder currently fast-paths), then
forwards the same input twice:

  1. **OFF (baseline)** — default dispatch. ``dsv4_csa.attention``
     resolves to alloy's ``_torch_csa_attention`` (eager attention +
     scatter-bias mask). This is the byte-exact reference path on this
     hardware.

  2. **ON (binder triton)** — ``activate(model, prefer="triton")``
     flips ``_dsv4_csa_implementation`` to ``"triton"`` so the CSA
     attention call dispatches to
     ``hf_npu_binder.deepseek_v4.sparse_flash_attention.triton`` (BHSD
     adapter over the vendored MindSpeed SFA kernel). HCA / sliding
     / MoE layers continue using alloy's torch fallback — only CSA
     swaps backends.

The diff between the two logit outputs is the **CSA fast-path drift**;
in bf16 we expect it to sit in the accumulation-order noise floor
(~1e-3 to 1e-2 max_abs). Same hardware, same random weights, same
input — the only thing that changes is the CSA attention impl, so any
larger drift is a wrapper bug.

Shape constraint: the binder's triton adapter only supports total
topk widths in ``CONFIG_MAP`` ``{128, 160, 640}``. ``sliding_window +
index_topk`` must land on one of these. Defaults (128 + 32 = 160) do.

Skipped cleanly if ``torch_npu`` or ``hf_npu_binder`` is not installed.

Usage::

    python -m alloy.tests.npu.compare_dsv4_binder_vs_torch \\
        --num-layers 4 --dtype bf16 --n-repeat 5

    # Also try the ascendc path (requires CANN with aclnnSparseAttnSharedkv):
    python -m alloy.tests.npu.compare_dsv4_binder_vs_torch --prefer ascendc
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

_IMPORT_ERR: str | None = None
try:
    import torch_npu  # noqa: F401
    from torch_npu.contrib import transfer_to_npu  # noqa: F401  patches torch.cuda -> npu
except ImportError as _e:
    _IMPORT_ERR = f"torch_npu not available ({_e}); this script must run on NPU hardware."

if _IMPORT_ERR is None:
    try:
        import hf_npu_binder  # noqa: F401
    except ImportError as _e:
        _IMPORT_ERR = f"hf_npu_binder not installed ({_e}); pip install hf-npu-binder."

if _IMPORT_ERR is None:
    from alloy import AlloyConfig, AlloyForCausalLM
    import alloy.integrations.hf_npu_binder as binder
    from alloy.tests._compare_utils import diff_logits, pick_device
else:
    AlloyConfig = AlloyForCausalLM = binder = None  # type: ignore[assignment]
    diff_logits = pick_device = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Config builder — small DSV4 with at least one CSA layer
# ---------------------------------------------------------------------------
def _layer_pattern(num_layers: int) -> list[str]:
    """Mirror DSV4's representative ``[hca, hca, csa, sliding]`` repeat so
    every attention flavour shows up, and the CSA layer is the one that
    swaps backends under ``activate(prefer="triton")``.
    """
    base = ["dsv4_hca_attention", "dsv4_hca_attention",
            "dsv4_csa_attention", "dsv4_sliding_attention"]
    out: list[str] = []
    while len(out) < num_layers:
        out.extend(base)
    return out[:num_layers]


def _ffn_pattern(num_layers: int) -> list[str]:
    """DSV4 convention: hash routing on the first few layers, topk after."""
    base = ["dsv4_hash_moe", "dsv4_moe", "dsv4_moe", "dsv4_moe"]
    out: list[str] = []
    while len(out) < num_layers:
        out.extend(base)
    return out[:num_layers]


def _build_config(args) -> AlloyConfig:
    head_dim = args.hidden_size // args.num_attention_heads
    return AlloyConfig(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_layers,
        num_attention_heads=args.num_attention_heads,
        num_key_value_heads=1,            # DSV4 is single-KV-head MQA
        head_dim=head_dim,
        intermediate_size=args.hidden_size,
        max_position_embeddings=max(args.seq_len, 512),
        sliding_window=args.sliding_window,
        rms_norm_eps=1e-6,
        attention_bias=False,
        attention_dropout=0.0,
        # MHC (mixture of head clusters)
        hc_mult=2,
        hc_sinkhorn_iters=2,
        hc_eps=1e-6,
        use_mhc=True,
        # MoE
        n_routed_experts=args.num_experts,
        num_experts_per_tok=args.num_experts_per_tok,
        scoring_func="sqrtsoftplus",
        routed_scaling_factor=1.5,
        swiglu_limit=10.0,
        mlp_bias=False,
        # DSV4 attention
        q_lora_rank=args.hidden_size // 2,
        o_groups=2,
        o_lora_rank=args.hidden_size // 2,
        # Lightning Indexer (CSA only). index_topk + sliding_window must
        # land on a CONFIG_MAP width for the binder triton path.
        index_n_heads=args.num_attention_heads,
        index_head_dim=head_dim,
        index_topk=args.index_topk,
        compress_rates={
            "compressed_sparse_attention": 4,
            "heavily_compressed_attention": 8,
        },
        rope_parameters={
            "main":     {"rope_type": "default", "rope_theta": 10000.0,  "partial_rotary_factor": 0.25},
            "compress": {"rope_type": "default", "rope_theta": 100000.0, "partial_rotary_factor": 0.25},
        },
        layer_types=_layer_pattern(args.num_layers),
        ffn_types=_ffn_pattern(args.num_layers),
        attn_implementation="eager",
    )


# ---------------------------------------------------------------------------
# Forward + timing
# ---------------------------------------------------------------------------
def _measure(
    model: AlloyForCausalLM,
    input_ids: torch.Tensor,
    *,
    n_warmup: int,
    n_repeat: int,
    device: torch.device,
) -> tuple[torch.Tensor, float, float]:
    """Returns (logits_of_last_call, t_first_seconds, t_avg_seconds_over_n_repeat)."""
    model.eval()
    sync = torch.npu.synchronize if device.type == "npu" else (lambda: None)

    with torch.no_grad():
        sync()
        t0 = time.perf_counter()
        out = model(input_ids=input_ids, use_cache=False)
        sync()
        t_first = time.perf_counter() - t0

        for _ in range(n_warmup):
            out = model(input_ids=input_ids, use_cache=False)
        sync()

        t0 = time.perf_counter()
        for _ in range(n_repeat):
            out = model(input_ids=input_ids, use_cache=False)
        sync()
        t_avg = (time.perf_counter() - t0) / n_repeat

    return out.logits, t_first, t_avg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _dtype_from_str(s: str) -> torch.dtype:
    return {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[s]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--num-attention-heads", type=int, default=8)
    parser.add_argument("--num-experts", type=int, default=4)
    parser.add_argument("--num-experts-per-tok", type=int, default=2)
    parser.add_argument("--vocab-size", type=int, default=1024)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--sliding-window", type=int, default=128,
        help="Sliding-window size W. The binder triton path requires "
             "W + index_topk to be in CONFIG_MAP {128, 160, 640}.",
    )
    parser.add_argument(
        "--index-topk", type=int, default=32,
        help="Lightning Indexer top-k value. Default 32 makes total = 128+32=160.",
    )
    parser.add_argument("--dtype", default="bf16", choices=["fp32", "bf16", "fp16"])
    parser.add_argument(
        "--prefer", default="triton", choices=["triton", "ascendc", "torch"],
        help="binder backend to compare baseline against. 'triton' is the "
             "MindSpeed kernel via BHSD adapter; 'ascendc' is CANN's "
             "aclnnSparseAttnSharedkv (requires the op in libopapi.so).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-warmup", type=int, default=2)
    parser.add_argument("--n-repeat", type=int, default=5)
    args = parser.parse_args()

    if _IMPORT_ERR is not None:
        print(f"SKIP - {_IMPORT_ERR}")
        return 0

    device = pick_device()
    dtype = _dtype_from_str(args.dtype)
    print(f"device={device}  dtype={dtype}  layers={args.num_layers}  hidden={args.hidden_size}  "
          f"seq_len={args.seq_len}  sliding_window={args.sliding_window}  index_topk={args.index_topk}")

    cfg = _build_config(args)
    csa_layers = sum(1 for lt in cfg.layer_types if lt == "dsv4_csa_attention")
    if csa_layers == 0:
        print(f"SKIP - no dsv4_csa_attention layers in the chosen pattern "
              f"(layer_types={cfg.layer_types}); nothing for binder to fast-path.")
        return 0
    print(f"CSA layers in this run: {csa_layers}/{args.num_layers}")

    torch.manual_seed(args.seed)
    # Generate on CPU then move — some CANN releases route a direct
    # ``torch.randint(..., device='npu')`` through ``aclnnInplaceRandom``
    # whose dynamic kernel config fails to parse on older toolkits.
    input_ids = torch.randint(0, cfg.vocab_size, (args.batch_size, args.seq_len)).to(device)

    # =========================================================================
    # Phase 1: BASELINE — alloy default ('torch' dispatch)
    # =========================================================================
    print("\n" + "=" * 70)
    print("[baseline] alloy default (CSA -> _torch_csa_attention)")
    torch.manual_seed(args.seed)
    baseline_model = AlloyForCausalLM(cfg).to(device=device, dtype=dtype)
    baseline_logits, t_first_off, t_avg_off = _measure(
        baseline_model, input_ids,
        n_warmup=args.n_warmup, n_repeat=args.n_repeat, device=device,
    )
    print(f"[baseline] forward - first {t_first_off*1000:.2f} ms  /  avg {t_avg_off*1000:.2f} ms over {args.n_repeat}")

    state_dict = {k: v.detach().clone() for k, v in baseline_model.state_dict().items()}
    baseline_logits_cpu = baseline_logits.detach().to("cpu").to(torch.float32)
    del baseline_model
    if device.type == "npu":
        torch.npu.empty_cache()

    # =========================================================================
    # Phase 2: BINDER — activate(prefer="triton") before constructing the model
    # =========================================================================
    print("\n" + "=" * 70)
    print(f"[binder]   activating prefer={args.prefer!r}")
    cfg_b = AlloyConfig(**cfg.to_dict())
    fake = type("Model", (), {"config": cfg_b})()
    # Use the mapping form so we only touch the dsv4_csa surface. The
    # string-broadcast form would also flip _qwen3_5_gdn_implementation
    # and _experts_implementation (neither relevant for this DSV4 model;
    # the experts flip in particular is the one that has caused HF's
    # _check_and_adjust_experts_implementation to reject the value).
    chosen = binder.activate(fake, prefer={"dsv4_csa": args.prefer})
    print(f"[binder]   activate() set: {chosen}")

    torch.manual_seed(args.seed)
    binder_model = AlloyForCausalLM(cfg_b).to(device=device, dtype=dtype)
    missing, unexpected = binder_model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"[binder]   state_dict load: {len(missing)} missing, {len(unexpected)} unexpected")
        print(f"           missing[:3]={missing[:3]}  unexpected[:3]={unexpected[:3]}")

    binder_logits, t_first_on, t_avg_on = _measure(
        binder_model, input_ids,
        n_warmup=args.n_warmup, n_repeat=args.n_repeat, device=device,
    )
    print(f"[binder]   forward - first {t_first_on*1000:.2f} ms  /  avg {t_avg_on*1000:.2f} ms over {args.n_repeat}")

    binder_logits_cpu = binder_logits.detach().to("cpu").to(torch.float32)
    del binder_model
    if device.type == "npu":
        torch.npu.empty_cache()

    # =========================================================================
    # Compare
    # =========================================================================
    print("\n" + "=" * 70)
    print("=== Precision (binder vs torch baseline, on the SAME hardware) ===")
    diffs = diff_logits(baseline_logits_cpu, binder_logits_cpu)
    for k, v in diffs.items():
        print(f"  {k:14s} {v:.6e}")

    print("\n=== Speed (avg over n-repeat) ===")
    print(f"  baseline (torch CSA):       {t_avg_off*1000:.2f} ms")
    print(f"  binder ({args.prefer}):     {t_avg_on*1000:.2f} ms")
    if t_avg_on > 0:
        speedup = t_avg_off / t_avg_on
        print(f"  speedup:                    {speedup:.2f}x")

    print("\n=== First-call latency (cold-start) ===")
    print(f"  baseline:  {t_first_off*1000:.2f} ms")
    print(f"  binder:    {t_first_on*1000:.2f} ms")

    return 0


if __name__ == "__main__":
    sys.exit(main())
