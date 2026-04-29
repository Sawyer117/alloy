# Examples

Entry points for building an `AlloyForCausalLM` and running it end-to-end
(load a real checkpoint, generate a continuation), plus packaging /
training helpers.

## Quick links

| What I want to do | Go to |
|---|---|
| Build a model from JSON config + load a real ckpt + generate | [`build_from_config.py`](#build_from_configpy--declarative) below |
| Build a model in Python + load an HF Qwen3 ckpt + generate | [`build_from_python.py`](#build_from_pythonpy--programmatic) below |
| Package an alloy config (+ tokenizer) into a Hub-loadable dir | [`package_config_for_hub.py`](package_config_for_hub.py) |
| **Train alloy with MindSpeed-MM (FSDP2)** | **[`train/README_mindspeed_mm.md`](train/README_mindspeed_mm.md)** |

## `build_from_config.py` — declarative

Architecture lives in a JSON file. The script parses it into an `AlloyConfig`,
builds the model, loads matching weights, and generates.

Shipped configs:

| Config file | Target ckpt | Architecture |
|---|---|---|
| `configs/qwen3_4b.json` | Qwen3-4B | 36 × `qwen3_attention` + `qwen3_mlp`, tied embeddings |
| `configs/qwen3_5_35b_a3b.json` | Qwen3.5-35B-A3B | 40 layers, 3:1 `qwen3_5_gdn` to `qwen3_attention`, `qwen3_5_moe` FFN throughout, attn output gate + unit-offset RMSNorm + partial rotary + mRoPE |

```bash
# plain full-attention path:
python -m alloy.examples.build_from_config \
    --config alloy/examples/configs/qwen3_4b.json \
    --pretrained /path/to/Qwen3-4B \
    --dtype bf16 --max-new-tokens 32

# GDN + MoE hybrid path:
python -m alloy.examples.build_from_config \
    --config alloy/examples/configs/qwen3_5_35b_a3b.json \
    --pretrained /path/to/Qwen3.5-35B-A3B \
    --dtype bf16 --max-new-tokens 8
```

Use this path when:
- the architecture should be reviewable / version-controlled separately from code
- you want to share configs on HuggingFace Hub alongside checkpoints
- multiple training runs differ only in config values

To express a new hybrid, copy one of the shipped configs and edit
`layer_types` / `ffn_types` (and the matching `linear_*` / `moe_*` / attention
hyperparameter blocks). Pass `--no-load-ckpt` to skip weight loading and
exercise the build + generate plumbing on random init.

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
