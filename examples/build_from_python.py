"""Example 2 — build an alloy model inline in Python, load a checkpoint, generate.

Same end-to-end goal as ``build_from_config.py`` but the config is constructed
programmatically rather than read from a JSON file. This path is the one you
use when iterating on architectures in a notebook / script — tweak a field,
rebuild, rerun.

The config is derived at runtime from the target checkpoint's *own* ``config.json``,
so this script works out-of-the-box with any Qwen3 variant (4B / 8B / 14B / 32B …).
For a hybrid architecture that doesn't have a ready ckpt, pass ``--toy`` to see
the build + generate path exercised on a small random-init mix.

This example uses :func:`alloy.build_skeleton` for the checkpoint-loading
case — skip the wasted CPU ``randn_`` init since ``load_state_dict`` will
overwrite everything. For a from-scratch training run, swap to
:func:`alloy.build_on_device` instead; it keeps ``_init_weights`` active and
runs all allocation + init ops on the accelerator.

Run:

    python -m alloy.examples.build_from_python --pretrained /path/to/Qwen3-4B

Or, for a hybrid architecture without a matching ckpt:

    python -m alloy.examples.build_from_python --toy
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from alloy import (
    AlloyConfig,
    AlloyForCausalLM,
    build_skeleton,
    load_state_dict_from_disk,
    strip_language_model_prefix,
)


# --------------------------------------------------------------------------- #
# Build AlloyConfig programmatically, optionally seeded by an HF Qwen3 config
# --------------------------------------------------------------------------- #


def build_alloy_config_from_qwen3_hf_dir(ckpt_dir: Path) -> AlloyConfig:
    """Translate a Qwen3 HuggingFace ``config.json`` into an ``AlloyConfig``.

    Equivalent to writing every field by hand — it's just reading values from
    the checkpoint's own config instead of hard-coding them, so the same code
    path works for Qwen3-4B, Qwen3-8B, Qwen3-14B, etc.

    To express a hybrid, replace the ``layer_types`` / ``ffn_types`` assignment
    below with your own pattern. The rest stays compatible with the Qwen3 ckpt.
    """
    with open(ckpt_dir / "config.json", encoding="utf-8") as f:
        hf_cfg = json.load(f)

    num_layers = hf_cfg["num_hidden_layers"]
    head_dim = hf_cfg.get("head_dim") or hf_cfg["hidden_size"] // hf_cfg["num_attention_heads"]
    layer_types = hf_cfg.get("layer_types") or ["full_attention"] * num_layers

    return AlloyConfig(
        # Global shape
        vocab_size=hf_cfg["vocab_size"],
        hidden_size=hf_cfg["hidden_size"],
        num_hidden_layers=num_layers,
        max_position_embeddings=hf_cfg["max_position_embeddings"],
        hidden_act=hf_cfg.get("hidden_act", "silu"),
        tie_word_embeddings=hf_cfg.get("tie_word_embeddings", False),
        # Per-layer architecture
        layer_types=layer_types,
        ffn_types=["mlp"] * num_layers,
        # Norm — Qwen3 uses plain RMSNorm (w * x, ones init)
        rms_norm_eps=hf_cfg["rms_norm_eps"],
        rms_norm_unit_offset=False,
        # Rotary — pass through whatever rope_parameters the ckpt specified
        rope_parameters=hf_cfg.get(
            "rope_parameters", {"rope_type": "default", "rope_theta": 10000.0}
        ),
        # Attention (GQA, no output gate in plain Qwen3)
        num_attention_heads=hf_cfg["num_attention_heads"],
        num_key_value_heads=hf_cfg.get("num_key_value_heads", hf_cfg["num_attention_heads"]),
        head_dim=head_dim,
        attention_bias=hf_cfg.get("attention_bias", False),
        attention_dropout=hf_cfg.get("attention_dropout", 0.0),
        attn_output_gate=False,
        sliding_window=hf_cfg.get("sliding_window"),
        # FFN
        intermediate_size=hf_cfg["intermediate_size"],
    )


def build_toy_hybrid_config() -> AlloyConfig:
    """Small demo AlloyConfig showcasing layer-type mixing.

    Not matched to any real checkpoint — useful for plumbing tests and for
    seeing how the registry dispatches different layer types.
    """
    num_layers = 8
    return AlloyConfig(
        vocab_size=32000,
        hidden_size=512,
        num_hidden_layers=num_layers,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=64,
        intermediate_size=2048,
        max_position_embeddings=4096,
        # 3 linear-attention layers per 1 full-attention layer, MoE FFN throughout.
        layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"] * 2,
        ffn_types=["moe"] * num_layers,
        rms_norm_unit_offset=True,
        attn_output_gate=True,
        linear_num_key_heads=4,
        linear_num_value_heads=8,
        linear_key_head_dim=64,
        linear_value_head_dim=64,
        num_experts=8,
        num_experts_per_tok=2,
        moe_intermediate_size=512,
        shared_expert_intermediate_size=512,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10000.0,
            "partial_rotary_factor": 0.25,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--pretrained",
        default=None,
        help="Path to a Qwen3 ckpt directory. AlloyConfig is derived from its "
        "config.json; the ckpt's safetensors are then loaded into the alloy model.",
    )
    parser.add_argument(
        "--toy",
        action="store_true",
        help="Skip --pretrained entirely and build a small hybrid-attention demo "
        "config. Useful for seeing the registry + generation path without any ckpt.",
    )
    parser.add_argument("--prompt", default="The theory of relativity is")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bf16", choices=["fp32", "bf16", "fp16"])
    args = parser.parse_args()

    if not args.toy and args.pretrained is None:
        parser.error("Pass either --pretrained <ckpt_dir> or --toy")

    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype]
    device = torch.device(args.device)

    # ------ 1. build AlloyConfig in Python ------------------------------ #
    if args.toy:
        print("[1/4] Building toy hybrid AlloyConfig in Python")
        config = build_toy_hybrid_config()
    else:
        print(f"[1/4] Deriving AlloyConfig from {args.pretrained}/config.json")
        config = build_alloy_config_from_qwen3_hf_dir(Path(args.pretrained))
    config._attn_implementation = "eager"
    print(
        f"      num_hidden_layers={config.num_hidden_layers} "
        f"hidden_size={config.hidden_size} "
        f"layer_types[0]={config.layer_types[0]!r} "
        f"ffn_types[0]={config.ffn_types[0]!r}"
    )

    # ------ 2. build skeleton (no init, directly in target dtype) ------- #
    print(f"[2/4] Instantiating AlloyForCausalLM skeleton in {dtype}")
    with build_skeleton(dtype):
        model = AlloyForCausalLM(config)
    model.tie_weights()  # alias lm_head for tied-embedding configs
    model.to(device).eval()

    # ------ 3. load weights --------------------------------------------- #
    if args.toy:
        print("[3/4] Toy mode — skipping weight load (weights stay uninitialized under no_init)")
        # Uninitialized bf16 tensors would make forward produce NaN garbage.
        # Drop to full init here so the plumbing demo produces finite output.
        print("      re-running _init_weights so the demo forward stays finite")
        model.apply(model._init_weights)
    else:
        ckpt_dir = Path(args.pretrained)
        print(f"[3/4] Loading state_dict from {ckpt_dir}")
        sd = load_state_dict_from_disk(
            ckpt_dir,
            ignore_patterns=[
                r".*rotary_emb\.inv_freq$",
                r"^model\.visual\.",
                r"^model\.mtp\.",
                r"^mtp\.",
            ],
            key_remap=strip_language_model_prefix,
        )
        result = model.load_state_dict(sd, strict=False)
        print(
            f"      missing={len(result.missing_keys)} "
            f"unexpected={len(result.unexpected_keys)} "
            f"(lm_head.weight missing is expected when tied)"
        )

    # ------ 4. generate ------------------------------------------------- #
    if args.toy:
        print("[4/4] Toy mode — generating from random input ids")
        input_ids = torch.randint(0, config.vocab_size, (1, 8), device=device)
        with torch.no_grad():
            output = model.generate(
                input_ids=input_ids, max_new_tokens=args.max_new_tokens, do_sample=False
            )
        print("\n----- Output token ids -----")
        print(output[0].tolist())
        return

    print(f"[4/4] Generating continuation (max_new_tokens={args.max_new_tokens})")
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained)
    inputs = tokenizer(args.prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            num_beams=1,
            pad_token_id=tokenizer.eos_token_id,
        )
    print("\n----- Output -----")
    print(tokenizer.decode(output[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
