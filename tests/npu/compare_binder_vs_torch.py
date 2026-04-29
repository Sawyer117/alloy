"""Binder ON vs OFF — speed + precision comparison on alloy.

Build a small alloy GDN+MoE model, run forward in two modes back-to-back:

  1. **OFF (baseline)** — every dispatch surface hits its ``"torch"`` default
     (alloy's eager port of HF's reference). No NPU fused kernel involved
     in the math, but the ops still execute on NPU because that's the
     device the tensors live on. Slow but the numerical reference for
     this hardware.

  2. **ON (binder)**  — ``alloy.integrations.hf_npu_binder.activate(model,
     prefer="flash")`` flips ``_qwen3_5_gdn_implementation`` and
     ``_experts_implementation`` to the binder backends.
     ``Qwen35GatedDeltaNet`` resolves its sub-functions at ``__init__``,
     so we have to build a fresh model after activate(); we then load the
     baseline ``state_dict`` so weights are byte-identical.

Both models execute the same input on the same device. The diff between
their outputs is the **fused-kernel drift** (lower-precision arithmetic
order); the wall-clock ratio is the **fused-kernel speedup**.

Skipped cleanly if ``torch_npu`` or ``hf_npu_binder`` is not installed —
the script is safe to invoke on a CPU dev box (it will print SKIP and
exit 0).

Usage::

    python -m alloy.tests.npu.compare_binder_vs_torch \\
        --num-layers 4 --hidden-size 512 --seq-len 256 \\
        --dtype bf16 --prefer flash --n-repeat 5
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Module-top imports use try/except so ``--help`` works on a CPU dev box.
# ``main()`` checks ``_IMPORT_ERR`` and prints SKIP when we can't run.
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

_IMPORT_ERR: str | None = None
try:
    import torch_npu  # noqa: F401
    from torch_npu.contrib import transfer_to_npu  # noqa: F401  patches torch.cuda → npu
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
# Config builder — small qwen3.5-MoE-like model exercising GDN + MoE
# ---------------------------------------------------------------------------
def _layer_pattern(num_layers: int) -> list[str]:
    """Mirror qwen3.5's canonical ``[gdn, gdn, gdn, full_attention]`` repeat."""
    base = ["qwen3_5_gdn", "qwen3_5_gdn", "qwen3_5_gdn", "qwen3_attention"]
    out: list[str] = []
    while len(out) < num_layers:
        out.extend(base)
    return out[:num_layers]


def _build_config(args) -> AlloyConfig:
    return AlloyConfig(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_layers,
        num_attention_heads=args.num_attention_heads,
        num_key_value_heads=args.num_key_value_heads,
        head_dim=args.hidden_size // args.num_attention_heads,
        intermediate_size=args.hidden_size * 2,
        max_position_embeddings=max(args.seq_len, 512),
        hidden_act="silu",
        rms_norm_eps=1e-6,
        rms_norm_unit_offset=True,    # qwen3.5 style
        attn_output_gate=True,         # qwen3.5 style
        layer_types=_layer_pattern(args.num_layers),
        ffn_types=["qwen3_5_moe"] * args.num_layers,
        # GDN
        linear_num_key_heads=args.num_key_value_heads,
        linear_num_value_heads=args.num_attention_heads,
        linear_key_head_dim=args.hidden_size // args.num_attention_heads,
        linear_value_head_dim=args.hidden_size // args.num_attention_heads,
        linear_conv_kernel_dim=4,
        # MoE
        num_experts=args.num_experts,
        num_experts_per_tok=args.num_experts_per_tok,
        moe_intermediate_size=args.hidden_size,
        shared_expert_intermediate_size=args.hidden_size,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10000.0,
            "partial_rotary_factor": 0.25,
        },
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
        # First call (includes any lazy class build / kernel autotune).
        sync()
        t0 = time.perf_counter()
        out = model(input_ids=input_ids, use_cache=False)
        sync()
        t_first = time.perf_counter() - t0

        # Warmup
        for _ in range(n_warmup):
            out = model(input_ids=input_ids, use_cache=False)
        sync()

        # Time the steady-state
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
    parser.add_argument("--num-attention-heads", type=int, default=4)
    parser.add_argument("--num-key-value-heads", type=int, default=2)
    parser.add_argument("--num-experts", type=int, default=4)
    parser.add_argument("--num-experts-per-tok", type=int, default=2)
    parser.add_argument("--vocab-size", type=int, default=1024)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--dtype", default="bf16", choices=["fp32", "bf16", "fp16"])
    parser.add_argument("--prefer", default="flash", choices=["flash", "triton", "torch"],
                        help="binder backend to compare baseline against")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-warmup", type=int, default=2)
    parser.add_argument("--n-repeat", type=int, default=5)
    args = parser.parse_args()

    if _IMPORT_ERR is not None:
        print(f"SKIP — {_IMPORT_ERR}")
        return 0

    device = pick_device()
    dtype = _dtype_from_str(args.dtype)
    print(f"device={device}  dtype={dtype}  layers={args.num_layers}  hidden={args.hidden_size}  seq_len={args.seq_len}")

    cfg = _build_config(args)

    torch.manual_seed(args.seed)
    input_ids = torch.randint(0, cfg.vocab_size, (args.batch_size, args.seq_len), device=device)

    # =========================================================================
    # Phase 1: BASELINE — torch defaults (no binder activation)
    # =========================================================================
    print("\n" + "=" * 70)
    print(f"[baseline] alloy default ('torch' dispatch) — building {sum(1 for _ in cfg.layer_types)} layers...")
    torch.manual_seed(args.seed)
    baseline_model = AlloyForCausalLM(cfg).to(device=device, dtype=dtype)
    baseline_logits, t_first_off, t_avg_off = _measure(
        baseline_model, input_ids,
        n_warmup=args.n_warmup, n_repeat=args.n_repeat, device=device,
    )
    print(f"[baseline] forward — first {t_first_off*1000:.2f} ms  /  avg {t_avg_off*1000:.2f} ms over {args.n_repeat}")

    # Save weights to load into the binder model so the ONLY difference is dispatch.
    state_dict = {k: v.detach().clone() for k, v in baseline_model.state_dict().items()}
    baseline_logits_cpu = baseline_logits.detach().to("cpu").to(torch.float32)
    del baseline_model
    if device.type == "npu":
        torch.npu.empty_cache()

    # =========================================================================
    # Phase 2: BINDER — activate(prefer=...) before constructing the model
    # =========================================================================
    print("\n" + "=" * 70)
    print(f"[binder]   activating prefer={args.prefer!r} (sets _<module>_implementation on config)...")
    cfg_b = AlloyConfig(**cfg.to_dict())
    fake = type("Model", (), {"config": cfg_b})()
    chosen = binder.activate(fake, prefer=args.prefer)
    print(f"[binder]   activate() set: {chosen}")

    # Re-seed identically — but weights come from state_dict load anyway, so
    # this is only here to keep the comparison maximally clean.
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
    print(f"[binder]   forward — first {t_first_on*1000:.2f} ms  /  avg {t_avg_on*1000:.2f} ms over {args.n_repeat}")

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
    print(f"  baseline (torch dispatch):  {t_avg_off*1000:.2f} ms")
    print(f"  binder ({args.prefer}):           {t_avg_on*1000:.2f} ms")
    if t_avg_on > 0:
        speedup = t_avg_off / t_avg_on
        print(f"  speedup:                    {speedup:.2f}x")

    print("\n=== First-call latency (cold-start, includes lazy autograd.Function build) ===")
    print(f"  baseline:  {t_first_off*1000:.2f} ms")
    print(f"  binder:    {t_first_on*1000:.2f} ms")

    return 0


if __name__ == "__main__":
    sys.exit(main())
