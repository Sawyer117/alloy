"""Simulated training step — alloy (with optional binder) vs HF reference.

Builds a small qwen3.5-MoE model on both sides with random init, copies
HF's weights into alloy so both have byte-identical parameters, runs one
forward + manual CE-with-shift loss + backward on each, and diffs:

  - forward logits
  - loss scalar
  - backward grads on every parameter (matched by name)

Pass ``--prefer flash`` (or ``triton``) to activate hf-npu-binder before
constructing the alloy model — the resulting grad diff vs HF tells you
whether the fast paths are training-correct, not just inference-correct.

The script is hardware-agnostic: pure-torch (default ``--prefer torch``)
runs on CPU/CUDA/NPU. ``--prefer flash`` / ``triton`` need triton or
torch_npu and skip cleanly otherwise.

Notes
-----
* ``output_router_logits=False`` is forced on both forwards so the MoE
  aux load-balancing loss does not fire — we want the diff to reflect
  pure CE-on-logits + pure backward, no second loss term.
* Manual CE-with-shift is computed identically on both sides instead of
  passing ``labels=`` to HF, so HF's internal loss helper drift can't
  contribute to the diff.
* Weights are HF-first: build HF, then ``alloy_model.load_state_dict(hf_model.state_dict())``.
  Param names must match qwen3.5 canonical (alloy uses HF naming) — the
  script reports any missing / unexpected keys verbosely so divergence
  is loud.

Usage::

    python -m alloy.tests.npu.compare_training_step_vs_hf
    python -m alloy.tests.npu.compare_training_step_vs_hf --prefer flash
    python -m alloy.tests.npu.compare_training_step_vs_hf --num-layers 4 --seq-len 64
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

# Module-top imports use try/except so ``--help`` works on a CPU box.
# The actual work happens after argparse; we check ``_BINDER_ERR`` /
# ``_NPU_ERR`` then.
_NPU_ERR: Optional[str] = None
try:
    import torch_npu  # noqa: F401
    from torch_npu.contrib import transfer_to_npu  # noqa: F401
except ImportError as _e:
    _NPU_ERR = f"torch_npu not available ({_e})"

_BINDER_ERR: Optional[str] = None
try:
    import hf_npu_binder  # noqa: F401
    import alloy.integrations.hf_npu_binder as _binder
except ImportError as _e:
    _BINDER_ERR = f"hf_npu_binder not installed ({_e})"
    _binder = None  # type: ignore[assignment]

from alloy import AlloyForCausalLM
from alloy.tests._compare_utils import (
    alloy_config_from_qwen3_5_text,
    diff_logits,
    pick_device,
)


# ---------------------------------------------------------------------------
# Build the HF reference with the canonical qwen3.5 3:1 mixer pattern
# ---------------------------------------------------------------------------
def _build_hf_text_config(args):
    from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig

    layer_types: list[str] = []
    while len(layer_types) < args.num_layers:
        layer_types.extend(["linear_attention"] * 3 + ["full_attention"])
    layer_types = layer_types[:args.num_layers]

    return Qwen3_5MoeTextConfig(
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
        attention_dropout=0.0,
        layer_types=layer_types,
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
        decoder_sparse_step=1,
        norm_topk_prob=True,
        output_router_logits=False,         # disable aux load-balancing loss
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0,
                         "partial_rotary_factor": 0.25},
        tie_word_embeddings=False,
    )


# ---------------------------------------------------------------------------
# Same loss formula on both sides, no HF labels= shortcut
# ---------------------------------------------------------------------------
def _shifted_ce_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Standard causal-LM CE: predict token[t+1] from logits[t]."""
    shift_logits = logits[..., :-1, :].contiguous().to(torch.float32)
    shift_labels = labels[..., 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
    )


# ---------------------------------------------------------------------------
# Per-layer forward hooks — pinpoint where drift starts
# ---------------------------------------------------------------------------
def _capture_hook(store: dict[str, torch.Tensor], key: str):
    """Forward hook that snapshots module output to CPU fp32."""
    def hook(module, args, output):
        # HF DecoderLayer returns tuple; alloy returns Tensor.
        t = output[0] if isinstance(output, tuple) else output
        store[key] = t.detach().to("cpu", torch.float32)
    return hook


def _attach_layer_hooks(model: torch.nn.Module) -> tuple[dict[str, torch.Tensor], list]:
    """Hook embed → every decoder layer output → final norm. Returns (store, handles)."""
    store: dict[str, torch.Tensor] = {}
    handles = [
        model.model.embed_tokens.register_forward_hook(_capture_hook(store, "embed")),
    ]
    for i, layer in enumerate(model.model.layers):
        handles.append(layer.register_forward_hook(_capture_hook(store, f"layer_{i}")))
    handles.append(model.model.norm.register_forward_hook(_capture_hook(store, "final_norm")))
    return store, handles


def _diff(ref: torch.Tensor, ours: torch.Tensor) -> dict[str, float]:
    ref = ref.to(torch.float32)
    ours = ours.to(torch.float32)
    if ref.shape != ours.shape:
        return {
            "shape_mismatch": True,
            "ref_shape": tuple(ref.shape),
            "our_shape": tuple(ours.shape),
        }
    diff = (ref - ours).abs()
    ref_max = ref.abs().max().clamp_min(1e-12)
    return {
        "max_abs": diff.max().item(),
        "mean_abs": diff.mean().item(),
        "max_ref_abs": ref.abs().max().item(),
        "relative_max": (diff.max() / ref_max).item(),
    }


# ---------------------------------------------------------------------------
# Forward + backward + grad collection
# ---------------------------------------------------------------------------
def _train_step(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Run one forward + backward step with per-layer forward hooks attached.

    Returns (logits_cpu_fp32, loss_cpu_fp32, {param_name: grad_cpu_fp32}, {capture_key: tensor_cpu_fp32}).
    """
    captures, handles = _attach_layer_hooks(model)
    try:
        model.train()
        model.zero_grad(set_to_none=True)

        out = model(input_ids=input_ids, output_router_logits=False, use_cache=False)
        logits = out.logits if hasattr(out, "logits") else out[0]
        loss = _shifted_ce_loss(logits, labels)
        loss.backward()

        grads: dict[str, torch.Tensor] = {}
        for name, p in model.named_parameters():
            if p.grad is not None:
                grads[name] = p.grad.detach().to("cpu", torch.float32).clone()
    finally:
        for h in handles:
            h.remove()

    captures["logits"] = logits.detach().to("cpu", torch.float32)
    return (
        logits.detach().to("cpu", torch.float32),
        loss.detach().to("cpu", torch.float32),
        grads,
        captures,
    )


# ---------------------------------------------------------------------------
# Wall-clock timing — forward only and full train step (forward + backward)
# ---------------------------------------------------------------------------
def _time_train_step(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    *,
    n_warmup: int,
    n_repeat: int,
    device: torch.device,
) -> dict[str, float]:
    """Returns per-iteration ms for forward-only and full train-step."""
    sync = torch.npu.synchronize if device.type == "npu" else (
        torch.cuda.synchronize if device.type == "cuda" else (lambda: None)
    )
    model.train()

    # Warmup runs full train step so caches / autotune kick in.
    for _ in range(n_warmup):
        model.zero_grad(set_to_none=True)
        out = model(input_ids=input_ids, output_router_logits=False, use_cache=False)
        logits = out.logits if hasattr(out, "logits") else out[0]
        loss = _shifted_ce_loss(logits, labels)
        loss.backward()

    # Forward-only timing (no_grad to skip autograd graph recording).
    sync()
    t0 = time.perf_counter()
    for _ in range(n_repeat):
        with torch.no_grad():
            model(input_ids=input_ids, output_router_logits=False, use_cache=False)
    sync()
    forward_ms = (time.perf_counter() - t0) / n_repeat * 1000.0

    # Full train-step timing.
    sync()
    t0 = time.perf_counter()
    for _ in range(n_repeat):
        model.zero_grad(set_to_none=True)
        out = model(input_ids=input_ids, output_router_logits=False, use_cache=False)
        logits = out.logits if hasattr(out, "logits") else out[0]
        loss = _shifted_ce_loss(logits, labels)
        loss.backward()
    sync()
    train_step_ms = (time.perf_counter() - t0) / n_repeat * 1000.0

    return {
        "forward_ms": forward_ms,
        "train_step_ms": train_step_ms,
        "backward_ms": max(train_step_ms - forward_ms, 0.0),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _dtype_from_str(s: str) -> torch.dtype:
    return {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[s]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--num-layers", type=int, default=4,
                        help="Try multiples of 4 to keep the canonical 3:1 pattern symmetric.")
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--num-attention-heads", type=int, default=4)
    parser.add_argument("--num-key-value-heads", type=int, default=2)
    parser.add_argument("--num-experts", type=int, default=4)
    parser.add_argument("--num-experts-per-tok", type=int, default=2)
    parser.add_argument("--vocab-size", type=int, default=512)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--dtype", default="fp32", choices=["fp32", "bf16", "fp16"],
                        help="fp32 for tightest numerical comparison; bf16/fp16 for realistic NPU run.")
    parser.add_argument("--prefer", default="torch", choices=["torch", "flash", "triton"],
                        help="binder backend to bind alloy with. 'torch' = no activation. "
                             "When != 'torch', binder is also activated on HF so its experts "
                             "go through ALL_EXPERTS_FUNCTIONS['<prefer>'] (= binder's kernel). "
                             "GDN on HF stays eager — there is no dispatch hook in HF's "
                             "Qwen3_5MoeGatedDeltaNet — so GDN-layer drift is expected here.")
    parser.add_argument("--attn-impl", default="eager", choices=["eager", "sdpa"],
                        help="attn_implementation for BOTH alloy and HF. Default 'eager' is "
                             "the byte-exact baseline. 'sdpa' on NPU dispatches to "
                             "npu_fused_attention (= flash attention 2 on Ascend), and on "
                             "CUDA dispatches to torch's scaled_dot_product_attention — both "
                             "sides hit the same kernel either way.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top-grad-diffs", type=int, default=10,
                        help="Print this many params with the worst grad diffs.")
    parser.add_argument("--n-warmup", type=int, default=1,
                        help="Train-step warmup iterations before timing.")
    parser.add_argument("--n-repeat", type=int, default=3,
                        help="Train-step iterations averaged for timing.")
    parser.add_argument("--no-timing", action="store_true",
                        help="Skip the timing pass (precision-only run).")
    parser.add_argument("--device", default=None,
                        help="Force device (cpu / cuda / npu). Default: auto-pick best.")
    args = parser.parse_args()

    # Skip checks AFTER argparse so --help works on CPU box.
    if args.prefer in ("flash", "triton"):
        if _BINDER_ERR is not None:
            print(f"SKIP — --prefer {args.prefer!r} requires hf_npu_binder: {_BINDER_ERR}")
            return 0
        try:
            import triton  # noqa: F401
        except ImportError:
            print(f"SKIP — --prefer {args.prefer!r} needs triton; "
                  f"this is a CPU dev box. Run on NPU/CUDA.")
            return 0

    device = pick_device(args.device)
    dtype = _dtype_from_str(args.dtype)
    print(f"device={device}  dtype={dtype}  prefer={args.prefer}")

    # =========================================================================
    # Phase 1: HF reference — build, forward, backward
    # =========================================================================
    from transformers.models.qwen3_5_moe import Qwen3_5MoeForCausalLM

    hf_cfg = _build_hf_text_config(args)
    print(f"\n[1/3] Building HF Qwen3_5MoeForCausalLM "
          f"(layers={hf_cfg.num_hidden_layers} hidden={hf_cfg.hidden_size} "
          f"experts={hf_cfg.num_experts}/{hf_cfg.num_experts_per_tok})")
    print(f"      layer_types: {hf_cfg.layer_types}")

    # Lock both sides to the same attn backend. Default 'eager' for byte-exact
    # baseline; 'sdpa' on NPU lands on npu_fused_attention (flash attn 2).
    hf_cfg._attn_implementation = args.attn_impl

    torch.manual_seed(args.seed)
    hf_model = Qwen3_5MoeForCausalLM(hf_cfg).to(device=device, dtype=dtype)

    # If binder is active, also flip HF's _experts_implementation so its
    # MoE experts dispatch through binder's flash kernel via HF's own
    # ALL_EXPERTS_FUNCTIONS table. (HF's @use_experts_implementation
    # decorator reads this field at every forward, so setting it after
    # construction works.) GDN on HF has no dispatch hook — stays eager.
    if args.prefer != "torch":
        chosen_hf = _binder.activate(hf_model, prefer=args.prefer)
        print(f"      activated binder on HF too: {chosen_hf}  "
              f"(GDN on HF cannot be redirected — drift expected on linear_attn layers)")

    # Same input + labels for both sides. Use input_ids as labels so the
    # CE-with-shift sees a well-defined target.
    torch.manual_seed(args.seed + 1)
    input_ids = torch.randint(0, hf_cfg.vocab_size, (args.batch_size, args.seq_len), device=device)
    labels = input_ids.clone()

    print(f"      input_ids: {tuple(input_ids.shape)}  dtype={input_ids.dtype}")
    print(f"[1/3] HF forward + backward (with per-layer hooks)...")
    hf_logits, hf_loss, hf_grads, hf_captures = _train_step(hf_model, input_ids, labels)
    print(f"      logits={tuple(hf_logits.shape)}  loss={hf_loss.item():.6f}  "
          f"grads on {len(hf_grads)} params  captures={len(hf_captures)}")

    hf_state = {k: v.detach().clone() for k, v in hf_model.state_dict().items()}

    # =========================================================================
    # Phase 2: Build alloy from HF config (optionally binder-activated),
    #          load HF state_dict, forward + backward
    # =========================================================================
    alloy_cfg = alloy_config_from_qwen3_5_text(hf_cfg)
    alloy_cfg._attn_implementation = args.attn_impl

    if args.prefer != "torch":
        fake = type("M", (), {"config": alloy_cfg})()
        chosen = _binder.activate(fake, prefer=args.prefer)
        print(f"\n[2/3] Activated hf_npu_binder: {chosen}")
    else:
        print("\n[2/3] No binder activation — alloy will use its 'torch' default dispatch.")

    print(f"[2/3] Building AlloyForCausalLM from translated config")
    torch.manual_seed(args.seed)
    alloy_model = AlloyForCausalLM(alloy_cfg).to(device=device, dtype=dtype)

    res = alloy_model.load_state_dict(hf_state, strict=False)
    print(f"      load_state_dict: missing={len(res.missing_keys)} unexpected={len(res.unexpected_keys)}")
    if res.missing_keys:
        for k in res.missing_keys[:5]:
            print(f"        + missing  {k}")
        if len(res.missing_keys) > 5:
            print(f"        + ... and {len(res.missing_keys) - 5} more")
    if res.unexpected_keys:
        for k in res.unexpected_keys[:5]:
            print(f"        - unexpct {k}")
        if len(res.unexpected_keys) > 5:
            print(f"        - ... and {len(res.unexpected_keys) - 5} more")

    print(f"[2/3] alloy forward + backward (with per-layer hooks)...")
    alloy_logits, alloy_loss, alloy_grads, alloy_captures = _train_step(alloy_model, input_ids, labels)
    print(f"      logits={tuple(alloy_logits.shape)}  loss={alloy_loss.item():.6f}  "
          f"grads on {len(alloy_grads)} params  captures={len(alloy_captures)}")

    # =========================================================================
    # Phase 3: Diff
    # =========================================================================
    print("\n[3/3] === DIFF: alloy vs HF reference ===\n")

    # Per-layer forward drift — pinpoint where divergence starts.
    keys_in_order = (
        ["embed"]
        + [f"layer_{i}" for i in range(args.num_layers)]
        + ["final_norm", "logits"]
    )
    layer_types = [""] + list(hf_cfg.layer_types) + ["", ""]

    print("=== Per-capture-point forward drift ===")
    header = f"{'capture':<14} {'layer_type':<22} {'max_abs':>12} {'mean_abs':>12} {'rel_max':>12}"
    print(header)
    print("-" * len(header))
    first_div: Optional[str] = None
    for k, lt in zip(keys_in_order, layer_types):
        if k not in hf_captures or k not in alloy_captures:
            print(f"{k:<14} {lt:<22}  (missing capture)")
            continue
        d = _diff(hf_captures[k], alloy_captures[k])
        if d.get("shape_mismatch"):
            print(f"{k:<14} {lt:<22}  SHAPE MISMATCH ref={d['ref_shape']} ours={d['our_shape']}")
            continue
        marker = ""
        if first_div is None and d["max_abs"] > 0:
            first_div = k
            marker = "  <-- first divergence"
        print(
            f"{k:<14} {lt:<22} {d['max_abs']:>12.3e} {d['mean_abs']:>12.3e} "
            f"{d['relative_max']:>12.3e}{marker}"
        )

    if first_div is None:
        print("\n  byte-exact at every capture point — alloy == HF")
    else:
        print(f"\n  first divergence: {first_div}")
        print("  Suspect (by capture point):")
        print("    embed                  -> Embedding load / weight tying / dtype cast")
        print("    layer_0..N (linear)    -> Qwen35GatedDeltaNet (chunk_rule, conv1d, A_log init)")
        print("    layer_N (full attn)    -> Qwen3Attention (RoPE, eager_attention_forward, mRoPE flag)")
        print("    layer_N (any)          -> Qwen35SparseMoE / experts dispatch / RMSNorm unit_offset")
        print("    final_norm             -> AlloyModel.norm wiring")
        print("    logits                 -> lm_head tying")

    print()
    print("=== Forward logits (final layer output) ===")
    fw = diff_logits(hf_logits, alloy_logits)
    for k, v in fw.items():
        print(f"  {k:14s} {v:.6e}")
    print()
    print(f"Loss:  hf={hf_loss.item():.6f}  alloy={alloy_loss.item():.6f}  "
          f"abs_diff={abs(hf_loss.item() - alloy_loss.item()):.6e}")

    # Per-param grad diff
    print("\nBackward grads (per parameter):")
    common = sorted(set(hf_grads) & set(alloy_grads))
    only_hf = sorted(set(hf_grads) - set(alloy_grads))
    only_alloy = sorted(set(alloy_grads) - set(hf_grads))

    if only_hf:
        print(f"  {len(only_hf)} param(s) had grad on HF but not alloy:")
        for n in only_hf[:5]:
            print(f"    + {n}")
    if only_alloy:
        print(f"  {len(only_alloy)} param(s) had grad on alloy but not HF:")
        for n in only_alloy[:5]:
            print(f"    - {n}")

    diffs = []
    for name in common:
        d = diff_logits(hf_grads[name], alloy_grads[name])
        diffs.append((name, d))

    if not diffs:
        print("  no comparable parameters!")
        return 1

    # Aggregate
    max_max = max(d["max_abs"] for _, d in diffs)
    mean_mean = sum(d["mean_abs"] for _, d in diffs) / len(diffs)
    print(f"\n  Aggregate over {len(diffs)} param(s):"
          f"  max-of-max-abs={max_max:.3e}  mean-of-mean-abs={mean_mean:.3e}")

    # Top-N by max-abs grad diff
    diffs.sort(key=lambda kv: -kv[1]["max_abs"])
    print(f"\n  Top {min(args.top_grad_diffs, len(diffs))} params by max_abs grad diff:")
    print(f"    {'name':50s} {'max_abs':>12s} {'mean_abs':>12s} {'rel_max':>12s}")
    for name, d in diffs[:args.top_grad_diffs]:
        print(f"    {name:50s} {d['max_abs']:>12.3e} {d['mean_abs']:>12.3e} {d['relative_max']:>12.3e}")

    # =========================================================================
    # Phase 4: timing (forward-only + full train step) on the same models /
    # input. The grid summary parser greps lines starting with "TIMING".
    # =========================================================================
    if not args.no_timing:
        print(f"\n=== Timing (n_warmup={args.n_warmup}, n_repeat={args.n_repeat}) ===")
        hf_t = _time_train_step(
            hf_model, input_ids, labels,
            n_warmup=args.n_warmup, n_repeat=args.n_repeat, device=device,
        )
        alloy_t = _time_train_step(
            alloy_model, input_ids, labels,
            n_warmup=args.n_warmup, n_repeat=args.n_repeat, device=device,
        )

        # Print in a parseable format. The bash grid extracts these lines.
        print(f"TIMING hf     forward_ms={hf_t['forward_ms']:.3f}  "
              f"backward_ms={hf_t['backward_ms']:.3f}  train_step_ms={hf_t['train_step_ms']:.3f}")
        print(f"TIMING alloy  forward_ms={alloy_t['forward_ms']:.3f}  "
              f"backward_ms={alloy_t['backward_ms']:.3f}  train_step_ms={alloy_t['train_step_ms']:.3f}")

        # Speedups (HF / alloy; >1 means alloy faster).
        def _ratio(num, den):
            return num / den if den > 0 else 0.0
        sp_fwd  = _ratio(hf_t["forward_ms"],    alloy_t["forward_ms"])
        sp_bwd  = _ratio(hf_t["backward_ms"],   alloy_t["backward_ms"])
        sp_step = _ratio(hf_t["train_step_ms"], alloy_t["train_step_ms"])
        print(f"TIMING speedup forward={sp_fwd:.3f}  backward={sp_bwd:.3f}  train_step={sp_step:.3f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
