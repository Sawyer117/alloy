"""Sub-module-level numerical-drift diagnostic — NPU edition.

Companion to debug_layerwise_diff.py. Where layerwise hooks on the WHOLE
decoder layer's output, this hooks each *sub-module* inside layer 0
(input_layernorm, mixer, post_attention_layernorm, mlp) so we can see which
sub-step first introduces drift relative to HF.

Use this after debug_layerwise_diff.py confirms the first-divergence layer
(typically layer_0 for qwen3.5-MoE on NPU). With this we can pinpoint
whether the residual drift is from RMSNorm, GDN, or MoE.

Defaults to fp32 + num_layers=1 to keep the output focused. With identical
math and the same NPU eager kernels, fp32 should give max_abs == 0 at
every sub-module capture point.

Usage:
    python -m alloy.tests.npu.debug_sublayer_diff \\
        --pretrained /path/to/Qwen3.5-35B-A3B
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
import torch_npu  # noqa: F401
from torch_npu.contrib import transfer_to_npu  # noqa: F401

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from alloy import AlloyForCausalLM
from alloy.tests._compare_utils import (
    alloy_config_from_qwen3_5_text,
    build_skeleton,
    empty_cache,
    load_state_dict_from_disk,
    pick_device,
)


def _capture_hook(store: dict[str, torch.Tensor], key: str):
    def hook(module, args, output):
        tensor = output[0] if isinstance(output, tuple) else output
        store[key] = tensor.detach().to("cpu", torch.float32)
    return hook


def _attach_layer0_hooks(model, side: str) -> tuple[dict[str, torch.Tensor], list]:
    """Hook layer-0 sub-modules + the layer itself.

    `side` is just for logging. Both HF and alloy expose the same attribute
    names on their decoder layer (input_layernorm, post_attention_layernorm,
    mlp, plus mixer at self_attn or linear_attn).
    """
    store: dict[str, torch.Tensor] = {}
    handles = []

    handles.append(model.model.embed_tokens.register_forward_hook(
        _capture_hook(store, "embed")
    ))

    layer0 = model.model.layers[0]

    handles.append(layer0.input_layernorm.register_forward_hook(
        _capture_hook(store, "L0/input_layernorm")
    ))

    # mixer attribute name differs by layer_type:
    #   linear_attention -> linear_attn
    #   full_attention   -> self_attn
    if hasattr(layer0, "linear_attn"):
        mixer = layer0.linear_attn
        mixer_name = "L0/mixer (linear_attn)"
    elif hasattr(layer0, "self_attn"):
        mixer = layer0.self_attn
        mixer_name = "L0/mixer (self_attn)"
    else:
        mixer = None
        mixer_name = None

    if mixer is not None:
        handles.append(mixer.register_forward_hook(_capture_hook(store, mixer_name)))

    handles.append(layer0.post_attention_layernorm.register_forward_hook(
        _capture_hook(store, "L0/post_attention_layernorm")
    ))
    handles.append(layer0.mlp.register_forward_hook(
        _capture_hook(store, "L0/mlp")
    ))
    handles.append(layer0.register_forward_hook(_capture_hook(store, "L0/output")))

    return store, handles


def _resolve_text_config(hf_full_cfg):
    return getattr(hf_full_cfg, "text_config", None) or hf_full_cfg


def phase_reference(pretrained, input_ids, num_layers, device, dtype):
    from transformers import AutoConfig
    from transformers.models.qwen3_5_moe import Qwen3_5MoeForCausalLM

    print(f"[phase-ref] Loading Qwen3.5-MoE reference: {pretrained}")
    full_cfg = AutoConfig.from_pretrained(pretrained)
    text_cfg = _resolve_text_config(full_cfg)
    text_cfg._attn_implementation = "eager"
    original_num_layers = text_cfg.num_hidden_layers
    if num_layers < original_num_layers:
        text_cfg.num_hidden_layers = num_layers
        text_cfg.layer_types = list(text_cfg.layer_types[:num_layers])

    ref = Qwen3_5MoeForCausalLM.from_pretrained(
        pretrained, config=text_cfg, torch_dtype=dtype, attn_implementation="eager"
    ).to(device).eval()

    store, handles = _attach_layer0_hooks(ref, "ref")
    ids = input_ids.to(device)
    with torch.no_grad():
        ref(input_ids=ids, use_cache=False)
    for h in handles:
        h.remove()

    print(f"[phase-ref] captured {len(store)} tensors")

    out = {
        "captures": store,
        "hf_text_config": text_cfg,
        "original_num_layers": original_num_layers,
        "layer_type_0": text_cfg.layer_types[0],
    }
    del ref
    empty_cache(device)
    return out


def phase_ours(pretrained, input_ids, num_layers, device, dtype, hf_text_cfg, original_num_layers):
    print("[phase-ours] Building AlloyForCausalLM")
    alloy_cfg = alloy_config_from_qwen3_5_text(hf_text_cfg)
    alloy_cfg._attn_implementation = "eager"

    with build_skeleton(dtype):
        ours = AlloyForCausalLM(alloy_cfg)
    ours.tie_weights()

    _LM_PREFIX = "model.language_model."
    def _strip(k):
        return ("model." + k[len(_LM_PREFIX):]) if k.startswith(_LM_PREFIX) else k

    ignore_patterns = [
        r"^mtp\.", r"^model\.visual\.", r"^model\.mtp\.",
        r".*rotary_emb\.inv_freq$",
    ]
    if alloy_cfg.num_hidden_layers < original_num_layers:
        for i in range(alloy_cfg.num_hidden_layers, original_num_layers):
            ignore_patterns.append(rf"^model\.layers\.{i}\.")
            ignore_patterns.append(rf"^model\.language_model\.layers\.{i}\.")

    sd = load_state_dict_from_disk(
        pretrained, ignore_patterns=ignore_patterns, device="cpu", key_remap=_strip
    )
    res = ours.load_state_dict(sd, strict=False)
    print(f"[phase-ours] load_state_dict: missing={len(res.missing_keys)} unexpected={len(res.unexpected_keys)}")
    del sd
    empty_cache(device)

    ours = ours.to(device).eval()

    store, handles = _attach_layer0_hooks(ours, "ours")
    ids = input_ids.to(device)
    with torch.no_grad():
        ours(input_ids=ids, use_cache=False)
    for h in handles:
        h.remove()

    print(f"[phase-ours] captured {len(store)} tensors")

    del ours
    empty_cache(device)
    return {"captures": store}


def _diff(ref_t, our_t):
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


def _fp32_ulp_at(magnitude):
    if magnitude <= 0:
        return 0.0
    import math
    return 2.0 ** (math.floor(math.log2(magnitude)) - 23)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pretrained", required=True)
    parser.add_argument("--prompt", default="The quick brown fox jumps over the lazy dog.")
    parser.add_argument("--num-layers", type=int, default=1,
                        help="Default 1: focus on layer_0 sub-modules. The prior "
                             "debug_layerwise_diff.py already isolated layer_0 as the first divergence.")
    parser.add_argument("--dtype", choices=["fp32", "bf16", "fp16"], default="fp32")
    parser.add_argument("--device", default="npu")
    args = parser.parse_args()

    device = pick_device(args.device)
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype]

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.pretrained)
    input_ids = tok(args.prompt, return_tensors="pt").input_ids
    print(f"Prompt: {args.prompt!r}  -> {input_ids.shape[1]} tokens")

    ref_out = phase_reference(args.pretrained, input_ids, args.num_layers, device, dtype)
    our_out = phase_ours(
        args.pretrained, input_ids, args.num_layers, device, dtype,
        ref_out["hf_text_config"], ref_out["original_num_layers"],
    )

    layer_type = ref_out["layer_type_0"]
    print(f"\n=== Sub-module drift inside layer_0 ({layer_type}, dtype={args.dtype}) ===")
    header = f"{'capture':<35} {'max_abs':>12} {'mean_abs':>12} {'rel_max':>12} {'ulp@max_ref':>12}"
    print(header)
    print("-" * len(header))

    keys_in_order = ["embed", "L0/input_layernorm"]
    if layer_type == "linear_attention":
        keys_in_order.append("L0/mixer (linear_attn)")
    else:
        keys_in_order.append("L0/mixer (self_attn)")
    keys_in_order += ["L0/post_attention_layernorm", "L0/mlp", "L0/output"]

    first_div = None
    for k in keys_in_order:
        if k not in ref_out["captures"] or k not in our_out["captures"]:
            print(f"{k:<35} (missing)")
            continue
        d = _diff(ref_out["captures"][k], our_out["captures"][k])
        if d.get("shape_mismatch"):
            print(f"{k:<35} SHAPE MISMATCH ref={d['ref_shape']} ours={d['our_shape']}")
            continue
        ulp = _fp32_ulp_at(d["max_ref_abs"])
        ulp_count = d["max_abs"] / ulp if ulp > 0 else float("inf")
        marker = ""
        if first_div is None and d["max_abs"] > 0:
            first_div = k
            marker = "  <-- first divergence"
        print(
            f"{k:<35} {d['max_abs']:>12.3e} {d['mean_abs']:>12.3e} "
            f"{d['relative_max']:>12.3e} {ulp_count:>11.1f}u{marker}"
        )

    print()
    if first_div is None:
        print("RESULT: byte-exact at every sub-module. alloy == HF in fp32.")
        return 0
    print(f"RESULT: first divergence at '{first_div}'.")
    print(
        "Sub-module key:"
        "\n  embed                          -> nn.Embedding lookup"
        "\n  L0/input_layernorm             -> Qwen3_5MoeRMSNorm(unit_offset=True) vs alloy RMSNorm"
        "\n  L0/mixer (linear_attn)         -> Qwen3_5MoeGatedDeltaNet vs alloy Qwen35GatedDeltaNet"
        "\n  L0/mixer (self_attn)           -> Qwen3_5MoeAttention vs alloy Qwen3Attention"
        "\n  L0/post_attention_layernorm    -> same RMSNorm class as input_layernorm"
        "\n  L0/mlp                         -> Qwen3_5MoeSparseMoeBlock vs alloy Qwen35SparseMoE"
        "\n  L0/output                      -> the residual + add wiring on AlloyDecoderLayer.forward"
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
