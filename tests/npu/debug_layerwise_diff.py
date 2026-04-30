"""Per-layer numerical-drift diagnostic vs HF Qwen3_5MoeForCausalLM — NPU edition.

Hooks the embed output, every decoder-layer output, the final norm output, and
the lm_head logits on both sides; runs the same input through HF reference and
through alloy; reports per-capture-point drift in fp32.

Use this to pinpoint *which* layer / module first introduces math drift
relative to HF. The first row in the per-layer table whose ``max_abs`` exceeds
~1 fp32 ulp at that magnitude is where alloy starts diverging from HF — i.e.
where the implementation is not byte-identical.

The script targets fp32 by default precisely so bf16 accumulation noise can't
mask a real math difference. With identical math and identical NPU eager
kernels, fp32 should give ``max_abs == 0`` at every capture point.

Usage:
    python -m alloy.tests.npu.debug_layerwise_diff \\
        --pretrained /path/to/Qwen3.5-35B-A3B \\
        --num-layers 4 --dtype fp32

Layer-type interpretation:

* drift starts at ``layer_0`` (linear_attention)            -> Qwen35GatedDeltaNet port
* drift starts at first ``full_attention`` layer            -> Qwen3Attention / eager_attention_forward
* drift only after the FFN of any layer                     -> Qwen35SparseMoE / experts dispatch / RoPE
* drift only at ``final_norm`` or ``logits``                -> AlloyModel.norm or lm_head wiring
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

# Diagnostic contract: drift at layer_N attributes a math difference to
# alloy's port at layer N. That attribution only makes sense when both
# sides run the same kernel family — both torch. Pin alloy to the in-tree
# torch reference by disabling the hf-npu-binder auto-bridge BEFORE alloy
# is imported (read in alloy/__init__.py at module load time).
os.environ["ALLOY_DISABLE_AUTO_BRIDGE"] = "1"

import torch
import torch_npu  # noqa: F401  registers the npu backend with torch
from torch_npu.contrib import transfer_to_npu  # noqa: F401  monkey-patches torch.cuda -> npu

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from alloy import AlloyForCausalLM
from alloy.tests._compare_utils import (
    alloy_config_from_qwen3_5_text,
    build_skeleton,
    empty_cache,
    load_state_dict_from_disk,
    pick_device,
)

DEFAULT_PROMPT = "The quick brown fox jumps over the lazy dog."


def _dtype_from_str(s: str) -> torch.dtype:
    return {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[s]


def _resolve_text_config(hf_full_cfg):
    text_cfg = getattr(hf_full_cfg, "text_config", None)
    return text_cfg if text_cfg is not None else hf_full_cfg


def _capture_hook(store: dict[str, torch.Tensor], key: str):
    """Return a forward_hook that snapshots the module output to CPU fp32."""

    def hook(module, args, output):
        # HF DecoderLayer may return tuple; alloy returns Tensor.
        tensor = output[0] if isinstance(output, tuple) else output
        store[key] = tensor.detach().to("cpu", torch.float32)

    return hook


def _attach_hooks(model, layer_module_iter) -> tuple[dict[str, torch.Tensor], list]:
    """Hook embed -> each decoder layer -> final norm. Returns (store, handles)."""
    store: dict[str, torch.Tensor] = {}
    handles = [model.model.embed_tokens.register_forward_hook(_capture_hook(store, "embed"))]
    for i, layer in enumerate(layer_module_iter):
        handles.append(layer.register_forward_hook(_capture_hook(store, f"layer_{i}")))
    handles.append(model.model.norm.register_forward_hook(_capture_hook(store, "final_norm")))
    return store, handles


def phase_reference(
    pretrained: str,
    input_ids: torch.Tensor,
    num_layers: int,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    from transformers import AutoConfig
    from transformers.models.qwen3_5_moe import Qwen3_5MoeForCausalLM

    print(f"[phase-ref] Loading Qwen3.5-MoE reference: {pretrained}")
    full_cfg = AutoConfig.from_pretrained(pretrained)
    text_cfg = _resolve_text_config(full_cfg)
    text_cfg._attn_implementation = "eager"

    original_num_layers = text_cfg.num_hidden_layers
    if num_layers > original_num_layers:
        raise ValueError(f"--num-layers={num_layers} > checkpoint layers ({original_num_layers})")
    if num_layers < original_num_layers:
        text_cfg.num_hidden_layers = num_layers
        text_cfg.layer_types = list(text_cfg.layer_types[:num_layers])
        print(f"[phase-ref] truncated to first {num_layers} of {original_num_layers} layers")

    ref = Qwen3_5MoeForCausalLM.from_pretrained(
        pretrained, config=text_cfg, torch_dtype=dtype, attn_implementation="eager"
    ).to(device).eval()

    store, handles = _attach_hooks(ref, ref.model.layers)
    ids = input_ids.to(device)
    with torch.no_grad():
        out = ref(input_ids=ids, use_cache=False)
        store["logits"] = out.logits.detach().to("cpu", torch.float32)
    for h in handles:
        h.remove()

    print(f"[phase-ref] captured {len(store)} tensors  layer_types={text_cfg.layer_types}")

    result = {
        "captures": store,
        "hf_text_config": text_cfg,
        "original_num_layers": original_num_layers,
        "layer_types": list(text_cfg.layer_types),
    }
    del ref
    empty_cache(device)
    print("[phase-ref] released reference from memory")
    return result


def phase_ours(
    pretrained: str,
    input_ids: torch.Tensor,
    num_layers: int,
    device: torch.device,
    dtype: torch.dtype,
    hf_text_cfg,
    original_num_layers: int,
) -> dict[str, Any]:
    print("[phase-ours] Building AlloyForCausalLM from qwen3.5 text config")
    alloy_cfg = alloy_config_from_qwen3_5_text(hf_text_cfg)
    alloy_cfg._attn_implementation = "eager"
    # Belt-and-suspenders: pin GDN to torch even if the auto-bridge somehow
    # got loaded — drift attribution at layer_N only works if both sides
    # run the same kernel family.
    alloy_cfg._qwen3_5_gdn_implementation = "torch"

    with build_skeleton(dtype):
        ours = AlloyForCausalLM(alloy_cfg)
    ours.tie_weights()

    _LM_PREFIX = "model.language_model."

    def _strip_language_model_prefix(k: str) -> str:
        if k.startswith(_LM_PREFIX):
            return "model." + k[len(_LM_PREFIX):]
        return k

    ignore_patterns = [
        r"^mtp\.",
        r"^model\.visual\.",
        r"^model\.mtp\.",
        r".*rotary_emb\.inv_freq$",
    ]
    if alloy_cfg.num_hidden_layers < original_num_layers:
        for i in range(alloy_cfg.num_hidden_layers, original_num_layers):
            ignore_patterns.append(rf"^model\.layers\.{i}\.")
            ignore_patterns.append(rf"^model\.language_model\.layers\.{i}\.")

    sd = load_state_dict_from_disk(
        pretrained, ignore_patterns=ignore_patterns, device="cpu",
        key_remap=_strip_language_model_prefix,
    )
    res = ours.load_state_dict(sd, strict=False)
    print(f"[phase-ours] load_state_dict: missing={len(res.missing_keys)} unexpected={len(res.unexpected_keys)}")
    if res.missing_keys:
        print(f"             missing (first 5): {res.missing_keys[:5]}")
    if res.unexpected_keys:
        print(f"             unexpected (first 5): {res.unexpected_keys[:5]}")
    del sd
    empty_cache(device)

    ours = ours.to(device).eval()

    store, handles = _attach_hooks(ours, ours.model.layers)
    ids = input_ids.to(device)
    with torch.no_grad():
        out = ours(input_ids=ids, use_cache=False)
        store["logits"] = out.logits.detach().to("cpu", torch.float32)
    for h in handles:
        h.remove()

    print(f"[phase-ours] captured {len(store)} tensors")

    del ours
    empty_cache(device)
    print("[phase-ours] released ours from memory")
    return {"captures": store}


def _diff(ref_t: torch.Tensor, our_t: torch.Tensor) -> dict[str, float]:
    ref_t = ref_t.to(torch.float32)
    our_t = our_t.to(torch.float32)
    if ref_t.shape != our_t.shape:
        return {"shape_mismatch": True, "ref_shape": tuple(ref_t.shape), "our_shape": tuple(our_t.shape)}
    diff = (ref_t - our_t).abs()
    ref_abs_max = ref_t.abs().max().clamp_min(1e-12)
    return {
        "max_abs": diff.max().item(),
        "mean_abs": diff.mean().item(),
        "max_ref_abs": ref_t.abs().max().item(),
        "relative_max": (diff.max() / ref_abs_max).item(),
    }


def _fp32_ulp_at(magnitude: float) -> float:
    """Approximate fp32 ulp at the given absolute magnitude."""
    if magnitude <= 0:
        return 0.0
    import math
    return 2.0 ** (math.floor(math.log2(magnitude)) - 23)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pretrained", required=True,
                        help="Local path or HF hub id for the qwen3.5 checkpoint")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--num-layers", type=int, default=4,
                        help="Truncate to first N layers on both sides (default 4 covers GDN+attn).")
    parser.add_argument("--dtype", choices=["fp32", "bf16", "fp16"], default="fp32",
                        help="fp32 (default) removes bf16 noise so any drift is real math difference.")
    parser.add_argument("--device", default="npu")
    args = parser.parse_args()

    device = pick_device(args.device)
    dtype = _dtype_from_str(args.dtype)

    if device.type != "npu":
        print(f"WARNING: requested NPU but pick_device returned {device}.")

    from transformers import AutoTokenizer

    print(f"Loading tokenizer: {args.pretrained}")
    tok = AutoTokenizer.from_pretrained(args.pretrained)
    input_ids = tok(args.prompt, return_tensors="pt").input_ids
    print(f"Prompt: {args.prompt!r}  -> {input_ids.shape[1]} tokens")

    ref_out = phase_reference(args.pretrained, input_ids, args.num_layers, device, dtype)
    our_out = phase_ours(
        args.pretrained, input_ids, args.num_layers, device, dtype,
        ref_out["hf_text_config"], ref_out["original_num_layers"],
    )

    # Walk capture points in execution order:
    keys_in_order = (
        ["embed"]
        + [f"layer_{i}" for i in range(args.num_layers)]
        + ["final_norm", "logits"]
    )
    layer_types = [""] + ref_out["layer_types"] + ["", ""]

    print("\n=== Per-capture-point drift (HF reference vs alloy) ===")
    header = f"{'capture':<14} {'layer_type':<22} {'max_abs':>12} {'mean_abs':>12} {'rel_max':>12} {'ulp@max_ref':>12}"
    print(header)
    print("-" * len(header))

    first_div_key: str | None = None
    for k, lt in zip(keys_in_order, layer_types):
        if k not in ref_out["captures"] or k not in our_out["captures"]:
            print(f"{k:<14} {lt:<22}  (missing capture)")
            continue
        d = _diff(ref_out["captures"][k], our_out["captures"][k])
        if d.get("shape_mismatch"):
            print(f"{k:<14} {lt:<22}  SHAPE MISMATCH ref={d['ref_shape']} ours={d['our_shape']}")
            continue
        ulp = _fp32_ulp_at(d["max_ref_abs"])
        ulp_count = d["max_abs"] / ulp if ulp > 0 else float("inf")
        marker = ""
        if first_div_key is None and d["max_abs"] > 0:
            first_div_key = k
            marker = "  <-- first divergence"
        print(
            f"{k:<14} {lt:<22} {d['max_abs']:>12.3e} {d['mean_abs']:>12.3e} "
            f"{d['relative_max']:>12.3e} {ulp_count:>11.1f}u{marker}"
        )

    print()
    if first_div_key is None:
        print("RESULT: byte-exact at every capture point in fp32. alloy == HF.")
        return 0
    print(f"RESULT: first divergence at '{first_div_key}'.")
    print(
        "Suspect (by capture point):"
        "\n  embed                  -> Embedding load / weight tying"
        "\n  layer_0..N (linear)    -> Qwen35GatedDeltaNet port (chunk/recurrent kernels, init state, conv1d)"
        "\n  layer_N (full attn)    -> Qwen3Attention or eager_attention_forward"
        "\n  layer_N (any)          -> Qwen35SparseMoE / _Experts dispatch / apply_rotary_pos_emb / RMSNorm"
        "\n  final_norm             -> AlloyModel.norm wiring"
        "\n  logits                 -> lm_head tying / head dtype"
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
