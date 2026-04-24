"""Example 1 — build an alloy model from a JSON config, load a checkpoint, generate.

This path is for workflows where the architecture is described declaratively in
a config file (easy to version, diff, share on Hub, etc.) and the runtime just
instantiates whatever the config says.

Flow:

    1. Read ``configs/qwen3_4b.json`` and parse it into an ``AlloyConfig``.
    2. Build an ``AlloyForCausalLM`` with that config. Weights are random —
       the config only describes shape / layer composition, not parameters.
    3. Stream real Qwen3-4B ``safetensors`` shards from the ckpt directory
       and load them into the model.
    4. Tokenize a prompt and run greedy ``generate`` to show a continuation.

Run:

    python -m alloy.examples.build_from_config --pretrained /path/to/Qwen3-4B

The provided ``qwen3_4b.json`` targets Qwen3-4B's published hparams (36 layers
of full-attention GQA with tied word embeddings). For a different Qwen3 variant
(e.g. Qwen3-8B), copy the JSON, adjust ``num_hidden_layers`` / ``hidden_size``
/ ``intermediate_size`` / etc. to match, and extend the per-layer lists.

To explore a hybrid architecture that you *don't* have a pretrained ckpt for,
the same script also works with ``--no-load-ckpt`` — it will skip the state
dict loading step and generate from random weights (output will be gibberish
but the end-to-end pipeline runs, useful for plumbing / debugging).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer

from alloy import AlloyConfig, AlloyForCausalLM


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def load_alloy_config(json_path: Path) -> AlloyConfig:
    """Parse a JSON file into an ``AlloyConfig``.

    Any section markers or commented-out keys are ignored: ``AlloyConfig.__init__``
    only picks up the fields it knows about and stashes the rest.
    """
    with open(json_path, encoding="utf-8") as f:
        config_dict = json.load(f)
    return AlloyConfig(**config_dict)


def load_safetensors_shards(ckpt_dir: Path) -> dict:
    """Read all safetensors shards in a HuggingFace-style checkpoint directory.

    Handles both single-file (``model.safetensors``) and sharded
    (``model-00001-of-00014.safetensors`` + index) layouts. Returns the
    concatenated ``state_dict``.
    """
    index_file = ckpt_dir / "model.safetensors.index.json"
    if index_file.exists():
        with open(index_file, encoding="utf-8") as f:
            shards = sorted(set(json.load(f)["weight_map"].values()))
    else:
        shards = ["model.safetensors"]

    full_sd: dict = {}
    for shard_name in shards:
        full_sd.update(load_file(str(ckpt_dir / shard_name)))
    return full_sd


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


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
    # For real training/inference you'd likely leave this unset and let HF auto-pick sdpa / flash.
    config._attn_implementation = "eager"
    print(
        f"      num_hidden_layers={config.num_hidden_layers} "
        f"hidden_size={config.hidden_size} "
        f"tie_word_embeddings={config.tie_word_embeddings}"
    )

    # ------ 2. build model ---------------------------------------------- #
    print(f"[2/4] Instantiating AlloyForCausalLM on {device} in {dtype}")
    model = AlloyForCausalLM(config).to(dtype).to(device).eval()
    # For tied-embedding configs this aliases lm_head.weight to model.embed_tokens.weight.
    # No-op when tie_word_embeddings is False.
    model.tie_weights()

    # ------ 3. load checkpoint ------------------------------------------ #
    if args.no_load_ckpt:
        print("[3/4] --no-load-ckpt set — skipping weight load (random init will be used)")
    else:
        if args.pretrained is None:
            parser.error("--pretrained is required unless --no-load-ckpt is passed")
        ckpt_dir = Path(args.pretrained)
        print(f"[3/4] Loading state_dict from {ckpt_dir}")
        sd = load_safetensors_shards(ckpt_dir)
        result = model.load_state_dict(sd, strict=False)
        print(
            f"      missing={len(result.missing_keys)} "
            f"unexpected={len(result.unexpected_keys)} "
            f"(lm_head.weight missing is expected when tied)"
        )

    # ------ 4. generate ------------------------------------------------- #
    tokenizer_path = args.pretrained if args.pretrained is not None else None
    if tokenizer_path is None:
        print("[4/4] No tokenizer path (no --pretrained) — done.")
        return

    print(f"[4/4] Generating continuation (max_new_tokens={args.max_new_tokens})")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
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
