# Examples

Two entry points for building an `AlloyForCausalLM` and running it end-to-end
(load a real checkpoint, generate a continuation).

## `build_from_config.py` — declarative

Architecture lives in a JSON file (`configs/qwen3_4b.json` here). The script
parses it into an `AlloyConfig`, builds the model, loads Qwen3-4B weights, and
generates.

```bash
python -m alloy.examples.build_from_config \
    --pretrained /path/to/Qwen3-4B \
    --prompt "The theory of relativity is" \
    --dtype bf16 --max-new-tokens 32
```

Use this path when:
- the architecture should be reviewable / version-controlled separately from code
- you want to share configs on HuggingFace Hub alongside checkpoints
- multiple training runs differ only in config values

To express a new hybrid, copy `configs/qwen3_4b.json`, replace
`layer_types` / `ffn_types` with your mix (e.g. `"linear_attention"` every
3 of 4 positions), point `--pretrained` at a compatible ckpt (or pass
`--no-load-ckpt` to skip the load step and run on random weights for a
plumbing check).

## `build_from_python.py` — programmatic

Same flow but `AlloyConfig` is constructed inline in Python. By default the
script reads the target checkpoint's `config.json` and translates Qwen3 hparams
into `AlloyConfig` fields at runtime — so it works out of the box with any
Qwen3 variant (4B / 8B / 14B / …) without a matching JSON file.

```bash
python -m alloy.examples.build_from_python \
    --pretrained /path/to/Qwen3-4B \
    --prompt "The theory of relativity is" \
    --dtype bf16 --max-new-tokens 32
```

Or run a toy hybrid architecture with no checkpoint at all:

```bash
python -m alloy.examples.build_from_python --toy
```

Use this path when:
- iterating interactively on architecture in a notebook / REPL
- you want the config source of truth to be code (branching logic,
  derived values, etc.)
- you need to bootstrap an `AlloyConfig` from an arbitrary HF Qwen3 ckpt

## Expected output

With a correct Qwen3-4B ckpt and either script:

```
[1/4] Loading/deriving AlloyConfig ...
[2/4] Instantiating AlloyForCausalLM on cuda in torch.bfloat16
[3/4] Loading state_dict ...
      missing=1 unexpected=0 (lm_head.weight missing is expected when tied)
[4/4] Generating continuation ...

----- Output -----
The theory of relativity is a theory of space and time. It was
developed by Albert Einstein in the early 20th century and has
...
```

The `missing=1` line refers to `lm_head.weight`; the checkpoint doesn't
store it because `tie_word_embeddings=true` and the storage is shared with
`model.embed_tokens.weight`. The model still has correct tied values after
load — the missing-key report is cosmetic.
