"""Per-capture-point drift diagnostic for compare_dsv4_random failures.

Same model construction as :mod:`alloy.tests.compare_dsv4_random` (HF DSV4
reference + alloy port, identical state_dict via copy). Adds forward hooks
on every alloy↔HF boundary point so we can see WHERE drift first exceeds
threshold:

  * embed_tokens                           — pre-MHC-expansion
  * each AlloyMhcDecoderLayer / DSV4DecoderLayer output
  * sub-points inside layer 0:
      - input_layernorm output
      - mixer (self_attn) output
      - post_attention_layernorm output
      - mlp output
      - attn_hc / ffn_hc collapsed (single-stream view)
  * hc_head output                         — final stream collapse
  * model.norm output                      — final RMSNorm
  * lm_head output                         — logits

Run this AFTER ``compare_dsv4_random`` reports a non-zero ``max_abs_diff``;
the printed table tells you where to dig next (which layer, which
sub-step). Same logic / interpretation as ``debug_layerwise_diff.py`` for
qwen3.5.

Usage::

    ALLOY_DISABLE_AUTO_BRIDGE=1 python -m alloy.tests.debug_dsv4_layerwise
    python -m alloy.tests.debug_dsv4_layerwise --threshold 1e-7
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def _capture_hook(store: dict[str, torch.Tensor], key: str):
    def hook(module, args, output):
        tensor = output[0] if isinstance(output, tuple) else output
        if isinstance(tensor, torch.Tensor):
            store[key] = tensor.detach().to("cpu", torch.float32)
    return hook


def _attach_hooks_full(model, num_layers: int):
    """Hook embed → each layer → optional hc_head → norm → lm_head."""
    store: dict[str, torch.Tensor] = {}
    handles: list = []

    handles.append(
        model.model.embed_tokens.register_forward_hook(_capture_hook(store, "embed"))
    )
    for i in range(num_layers):
        layer = model.model.layers[i]
        handles.append(layer.register_forward_hook(_capture_hook(store, f"layer_{i:02d}")))
    if hasattr(model.model, "hc_head") and model.model.hc_head is not None:
        handles.append(
            model.model.hc_head.register_forward_hook(_capture_hook(store, "hc_head"))
        )
    handles.append(model.model.norm.register_forward_hook(_capture_hook(store, "final_norm")))
    handles.append(model.lm_head.register_forward_hook(_capture_hook(store, "logits")))
    return store, handles


def _attach_hooks_layer0_sub(model, side_label: str):
    """Hook every interesting sub-point inside layer 0 for fine-grained
    drill-down. Sub-modules are named the same on both sides.

    Coarse points (block-level):
      input_layernorm, self_attn, post_attention_layernorm, mlp
      attn_hc, ffn_hc (MHC sites)

    Fine points inside self_attn (run when the coarse self_attn DIVERGES,
    to localise which op inside DSV4 attention introduced drift):
      q_a_proj, q_a_norm, q_b_proj, q_b_norm
      kv_proj, kv_norm
      compressor                — HCA / CSA / None
      o_a_proj, o_b_proj
    """
    store: dict[str, torch.Tensor] = {}
    handles: list = []
    layer0 = model.model.layers[0]

    # ----- block-level -----
    coarse = ["input_layernorm", "self_attn", "post_attention_layernorm",
              "mlp", "attn_hc", "ffn_hc"]
    for name in coarse:
        if hasattr(layer0, name):
            handles.append(getattr(layer0, name).register_forward_hook(
                _capture_hook(store, f"L0/{side_label}/{name}")
            ))

    # ----- inside self_attn -----
    if hasattr(layer0, "self_attn"):
        attn = layer0.self_attn
        attn_subs = ["q_a_proj", "q_a_norm", "q_b_proj", "q_b_norm",
                     "kv_proj", "kv_norm", "o_a_proj", "o_b_proj"]
        for name in attn_subs:
            if hasattr(attn, name):
                handles.append(getattr(attn, name).register_forward_hook(
                    _capture_hook(store, f"L0/{side_label}/self_attn.{name}")
                ))
        if hasattr(attn, "compressor") and attn.compressor is not None:
            handles.append(attn.compressor.register_forward_hook(
                _capture_hook(store, f"L0/{side_label}/self_attn.compressor")
            ))
    return store, handles


def _diff(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    if a.shape != b.shape:
        return {"shape_mismatch": True, "a_shape": tuple(a.shape), "b_shape": tuple(b.shape)}
    d = (a - b).abs()
    return {
        "max_abs": d.max().item(),
        "mean_abs": d.mean().item(),
        "max_ref_abs": a.abs().max().item(),
    }


def _fp32_ulp_at(magnitude: float) -> float:
    if magnitude <= 0:
        return 0.0
    return 2.0 ** (math.floor(math.log2(magnitude)) - 23)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--threshold", type=float, default=1e-5,
                        help="max_abs above which a capture point is flagged as 'diverged'.")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    # Reuse the same test config for both sides
    from alloy.tests.compare_dsv4_random import _build_configs
    hf_cfg, alloy_cfg = _build_configs()
    num_layers = hf_cfg.num_hidden_layers

    from transformers.models.deepseek_v4 import DeepseekV4ForCausalLM

    from alloy import AlloyForCausalLM

    print("[1/3] Building HF + alloy, copying state_dict ...", flush=True)
    torch.manual_seed(args.seed)
    hf_model = DeepseekV4ForCausalLM(hf_cfg).to(device=args.device, dtype=torch.float32).eval()
    alloy_model = AlloyForCausalLM(alloy_cfg).to(device=args.device, dtype=torch.float32).eval()
    res = alloy_model.load_state_dict(hf_model.state_dict(), strict=False)
    if res.missing_keys or res.unexpected_keys:
        print(f"      WARNING — missing={len(res.missing_keys)} unexpected={len(res.unexpected_keys)}; "
              "subsequent diff numbers may reflect that, not algorithmic drift.")

    print("[2/3] Hooks: full pass + layer-0 sub-points ...", flush=True)
    hf_full, hf_handles_full = _attach_hooks_full(hf_model, num_layers)
    alloy_full, alloy_handles_full = _attach_hooks_full(alloy_model, num_layers)
    hf_l0, hf_handles_l0 = _attach_hooks_layer0_sub(hf_model, "hf")
    alloy_l0, alloy_handles_l0 = _attach_hooks_layer0_sub(alloy_model, "alloy")

    torch.manual_seed(args.seed)
    input_ids = torch.randint(
        0, hf_cfg.vocab_size, (args.batch_size, args.seq_len), device=args.device
    )

    print(f"[3/3] Forward {args.batch_size}x{args.seq_len}, fp32, single shot ...", flush=True)
    with torch.inference_mode():
        hf_model(input_ids=input_ids)
        alloy_model(input_ids=input_ids)

    for h in hf_handles_full + alloy_handles_full + hf_handles_l0 + alloy_handles_l0:
        h.remove()

    # ------------------------------------------------------------------ #
    # 1. Whole-model capture chain
    # ------------------------------------------------------------------ #
    capture_keys = (
        ["embed"]
        + [f"layer_{i:02d}" for i in range(num_layers)]
        + (["hc_head"] if "hc_head" in hf_full and "hc_head" in alloy_full else [])
        + ["final_norm", "logits"]
    )

    print()
    print("=" * 96)
    print(f"{'capture':<14} {'shape':<28} {'max_abs':>12} {'mean_abs':>12} {'rel_max':>12} {'fp32_ulp@ref':>13}  flag")
    print("-" * 96)
    first_div = None
    for key in capture_keys:
        if key not in hf_full or key not in alloy_full:
            print(f"{key:<14} (missing on one side: hf={key in hf_full}, alloy={key in alloy_full})")
            continue
        ref, ours = hf_full[key], alloy_full[key]
        d = _diff(ref, ours)
        if d.get("shape_mismatch"):
            print(f"{key:<14} SHAPE MISMATCH hf={d['a_shape']} alloy={d['b_shape']}")
            if first_div is None:
                first_div = key
            continue
        ulp = _fp32_ulp_at(d["max_ref_abs"])
        ulp_count = d["max_abs"] / ulp if ulp > 0 else float("inf")
        rel = d["max_abs"] / max(d["max_ref_abs"], 1e-12)
        flag = ""
        if d["max_abs"] > args.threshold and first_div is None:
            first_div = key
            flag = "<-- first divergence"
        print(
            f"{key:<14} {str(tuple(ref.shape)):<28} {d['max_abs']:12.4e} "
            f"{d['mean_abs']:12.4e} {rel:12.4e} {ulp_count:>12.1f}u  {flag}"
        )

    # ------------------------------------------------------------------ #
    # 2. Layer-0 sub-point drill-down
    # ------------------------------------------------------------------ #
    print()
    print("=" * 96)
    print("Layer-0 sub-points (fine-grained drill-down)")
    print("-" * 96)
    # Explicit forward-order grouping (block-level → attention internals
    # → MLP / HC). Anything captured but not listed here gets appended.
    ordered_subs = [
        "input_layernorm",
        "attn_hc",
        "self_attn",                     # coarse — flagged DIVERGES means dig below
        "self_attn.q_a_proj",
        "self_attn.q_a_norm",
        "self_attn.q_b_proj",
        "self_attn.q_b_norm",
        "self_attn.kv_proj",
        "self_attn.kv_norm",
        "self_attn.compressor",
        "self_attn.o_a_proj",
        "self_attn.o_b_proj",
        "post_attention_layernorm",
        "ffn_hc",
        "mlp",
    ]
    captured = set(k.split("/", 2)[-1] for k in hf_l0)
    leftovers = sorted(captured - set(ordered_subs))
    sub_keys = [s for s in ordered_subs if s in captured] + leftovers

    print(f"{'sub-point':<32} {'shape':<28} {'max_abs':>12} {'mean_abs':>12}  flag")
    print("-" * 96)
    for sub in sub_keys:
        hf_key = f"L0/hf/{sub}"
        al_key = f"L0/alloy/{sub}"
        if hf_key not in hf_l0 or al_key not in alloy_l0:
            continue
        ref, ours = hf_l0[hf_key], alloy_l0[al_key]
        d = _diff(ref, ours)
        if d.get("shape_mismatch"):
            print(f"{sub:<32} SHAPE MISMATCH hf={d['a_shape']} alloy={d['b_shape']}")
            continue
        flag = "" if d["max_abs"] <= args.threshold else "DIVERGES"
        print(
            f"{sub:<32} {str(tuple(ref.shape)):<28} "
            f"{d['max_abs']:12.4e} {d['mean_abs']:12.4e}  {flag}"
        )

    # ------------------------------------------------------------------ #
    # 3. Headline + interpretation
    # ------------------------------------------------------------------ #
    print()
    print("=" * 96)
    if first_div is None:
        print(f"All capture points within threshold {args.threshold:.0e}.")
    else:
        print(f"First divergence > {args.threshold:.0e}: {first_div}")
        print()
        print("Interpretation:")
        if first_div == "embed":
            print("  embed_tokens parameters or input handling differ.")
        elif first_div.startswith("layer_"):
            i = int(first_div.split("_")[-1])
            print(f"  layer {i} introduced drift (math is correct up to layer {i-1}).")
            print("  Inspect the layer-0 sub-points above to localise within the layer.")
            if i > 0:
                print(f"  Or: rerun with --seq-len 1 to simplify, then attach sub-hooks on layer {i}.")
        elif first_div == "hc_head":
            print("  HyperHead (final MHC collapse) op differs.")
        elif first_div == "final_norm":
            print("  final RMSNorm differs (check rms_norm_unit_offset / class).")
        elif first_div == "logits":
            print("  lm_head wiring or chunk_loss / logits_to_keep path differs.")
    return 0 if first_div is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
