"""Example 1 — build an alloy model from a JSON config, load a checkpoint, generate.

This path is for workflows where the architecture is described declaratively in
a config file (easy to version, diff, share on Hub, etc.) and the runtime just
instantiates whatever the config says.

Flow:

    1. Read ``configs/qwen3_4b.json`` and parse it into an ``AlloyConfig``.
    2. Build an ``AlloyForCausalLM`` using ``build_skeleton`` — skip ``_init_weights``
       since we're about to overwrite every parameter anyway. For a 4B model
       this turns ~30-60 s of CPU ``randn_`` into a few hundred ms of shape
       allocation. For training from scratch, use ``build_on_device`` instead
       (keeps init correct, runs it on the accelerator).
    3. Stream real Qwen3-4B ``safetensors`` shards from the ckpt directory
       straight into the skeleton via ``load_state_dict_from_disk``.
    4. Tokenize a prompt and run greedy ``generate`` to show a continuation.

Run:

    python -m alloy.examples.build_from_config --pretrained /path/to/Qwen3-4B

The provided ``qwen3_4b.json`` targets Qwen3-4B's published hparams (36 layers
of full-attention GQA with tied word embeddings). For a different Qwen3 variant
(e.g. Qwen3-8B), copy the JSON, adjust ``num_hidden_layers`` / ``hidden_size``
/ ``intermediate_size`` / etc., and extend the per-layer lists.

Pass ``--no-load-ckpt`` to exercise the build and generation path with random
weights — useful for plumbing a new hybrid layer_types pattern before you have
a matching checkpoint.
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


def load_alloy_config(json_path: Path) -> AlloyConfig:
    """Parse a JSON file into an ``AlloyConfig``.

    Unknown keys (e.g. section markers from a human-edited file) are ignored:
    ``AlloyConfig.__init__`` picks up recognized fields and stashes the rest.
    """
    with open(json_path, encoding="utf-8") as f:
        config_dict = json.load(f)
    return AlloyConfig(**config_dict)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--config",
        default=str(Path(__file__).parent / "configs" / "qwen3_4b.json"),
        help="Path to an AlloyConfig JSON file (default: configs/qwen3_4b.json)",
    )
    parser.add_argument(
        "--pretrained",
        default=None,
        help="Path to a Qwen3-style checkpoint directory containing "
        "config.json + safetensors shards. Required unless --no-load-ckpt.",
    )
    parser.add_argument(
        "--no-load-ckpt",
        action="store_true",
        help="Skip state_dict loading — generate from random weights (plumbing test).",
    )
    parser.add_argument("--prompt", default="The theory of relativity is")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bf16", choices=["fp32", "bf16", "fp16"])
    args = parser.parse_args()

    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype]
    device = torch.device(args.device)

    # ------ 1. parse JSON -> AlloyConfig --------------------------------- #
    print(f"[1/4] Loading AlloyConfig from {args.config}")
    config = load_alloy_config(Path(args.config))
    # Use eager attention so numerical results are reproducible and hardware-agnostic.
    config._attn_implementation = "eager"
    print(
        f"      num_hidden_layers={config.num_hidden_layers} "
        f"hidden_size={config.hidden_size} "
        f"tie_word_embeddings={config.tie_word_embeddings}"
    )

    # ------ 2. build skeleton (no init, directly in target dtype) -------- #
    print(f"[2/4] Instantiating AlloyForCausalLM skeleton in {dtype}")
    with build_skeleton(dtype):
        model = AlloyForCausalLM(config)
    # build_skeleton runs under no_init_weights, which also bypasses HF's
    # post-init tie_weights. Do the tie ourselves before loading: the ckpt
    # only stores model.embed_tokens.weight for tied configs, and lm_head
    # needs to share that storage to be correct after load.
    model.tie_weights()
    model.to(device).eval()

    # ------ 3. load checkpoint ------------------------------------------ #
    if args.no_load_ckpt:
        print("[3/4] --no-load-ckpt set — skipping weight load (random init will be used)")
    else:
        if args.pretrained is None:
            parser.error("--pretrained is required unless --no-load-ckpt is passed")
        ckpt_dir = Path(args.pretrained)
        print(f"[3/4] Loading state_dict from {ckpt_dir}")
        sd = load_state_dict_from_disk(
            ckpt_dir,
            ignore_patterns=[
                # Non-persistent buffer — recomputed from config at RotaryEmbedding.__init__.
                r".*rotary_emb\.inv_freq$",
                # Vision / multimodal heads stored in ...ForConditionalGeneration
                # checkpoints (e.g. Qwen3.5-MoE). Safe no-op for pure text ckpts.
                r"^model\.visual\.",
                r"^model\.mtp\.",
                r"^mtp\.",
            ],
            # Strip the model.language_model.* wrapper prefix used by multimodal
            # ConditionalGeneration checkpoints — HF's from_pretrained does this
            # automatically via base_model_prefix; raw safetensors reads don't.
            # No-op for plain ...ForCausalLM checkpoints (Qwen3-4B etc.).
            key_remap=strip_language_model_prefix,
        )
        result = model.load_state_dict(sd, strict=False)
        print(
            f"      missing={len(result.missing_keys)} "
            f"unexpected={len(result.unexpected_keys)} "
            f"(lm_head.weight missing is expected when tied)"
        )

    # ------ 4. generate ------------------------------------------------- #
    if args.pretrained is None:
        print("[4/4] No tokenizer path (no --pretrained) — done.")
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
