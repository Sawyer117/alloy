# Training alloy under MindSpeed-MM (FSDP2)

MindSpeed-MM has a clean plugin system that loads any HuggingFace-native
model as long as `model_id` resolves through `model_register` and the model
directory is loadable via `AutoConfig.from_pretrained(..., trust_remote_code=True)`.
Alloy meets both. This doc walks through the workflow end-to-end and ships a
concrete worked example for a 340M qwen3-next-style dense model.

> **Prerequisite — apply MindSpeed-MM patches first.** Before training will
> actually run end-to-end, three upstream sites in MindSpeed-MM need small
> patches (parquet dataloader; loss_func `**kwargs` sink; defensive
> `output.aux_loss` access). See
> [`mindspeed_mm_patches.md`](./mindspeed_mm_patches.md) for the diffs and
> rationale. These are not alloy-specific — any HF-native CausalLM under
> MindSpeed-MM needs them — but skipping them gives confusing crashes
> partway into training, not at startup.

## What lives where

| File | Role |
|---|---|
| `alloy/integrations/mindspeed_mm.py` | Plugin shim — registers `AlloyForCausalLM` under `model_id="alloy"` and forwards `_<module>_implementation` yaml fields onto the HF config |
| `alloy/integrations/hf_npu_binder.py` | Binder bridge — registers `triton` / `flash` fast-path kernels into alloy + HF dispatch tables (optional but recommended on NPU) |
| `alloy/examples/package_config_for_hub.py` | Generates the model directory (config.json + 1-line modeling shim + auto_map). Run once per model architecture. |
| `alloy/examples/configs/qwen3_*.json` | Pre-built alloy config JSONs for each model size / architecture flavor |
| `examples/train/pretrain_alloy_qwen3_5_moe_mindspeed.yaml` | Yaml template — qwen3.5-MoE alloy (**GDN + MoE**) |
| `examples/train/pretrain_alloy_qwen3_next_340m_mindspeed.yaml` | Yaml template — qwen3-next 340M alloy (**GDN, no MoE**) |
| `examples/train/pretrain_alloy_qwen3_dense_mindspeed.yaml` | Yaml template — qwen3 dense alloy (**no GDN, no MoE**) |
| `examples/train/mindspeed_mm_patches.md` | Required upstream patches on MindSpeed-MM (parquet dataloader; loss_func kwargs sink; defensive aux_loss access). Apply before training. |

## Quickstart: which template fits your model

Pick the yaml template based on whether your alloy config has GDN linear-attention
layers and/or MoE feed-forward layers:

| your alloy `layer_types` | your `ffn_types` | template |
|---|---|---|
| includes `qwen3_5_gdn` | includes `qwen3_5_moe` | `pretrain_alloy_qwen3_5_moe_mindspeed.yaml` |
| includes `qwen3_5_gdn` | all `qwen3_mlp` | `pretrain_alloy_qwen3_next_340m_mindspeed.yaml` |
| only `qwen3_attention*` | all `qwen3_mlp` | `pretrain_alloy_qwen3_dense_mindspeed.yaml` |

The three differ in FSDP `apply_modules` paths (whether `linear_attn` or
`mlp.experts` entries are needed) and whether `expert_parallel_size` /
`router_aux_loss_coef` are configured.

---

## Concrete walkthrough: qwen3-next-340m-dense

This walks you through the **exact** flow assuming an alloy config matching
the existing `examples/configs/qwen3_next_340m_dense.json` — 24 layers in
3:1 GDN/full-attention pattern, dense MLP, hidden=1024, vocab=32000. The
config corresponds 1:1 to a typical MindSpeed-LLM `pretrain_*.sh`
launcher's `--num-layers 24 --hidden-size 1024 --ffn-hidden-size 2816
--num-attention-heads 16 --num-query-groups 2 --full-attention-interval 3
--linear-{key,value}-head-dim 256 --linear-num-{key,value}-heads 4`.

### 1. Generate the model directory + copy tokenizer

On the NPU machine, after `pip install -e /path/to/alloy`:

```bash
TARGET=./hf_models/alloy_qwen3_next_340m
SRC_TOKENIZER=/path/to/your/tokenizer/dir   # e.g. /home/.../340M-20B-GatedDeltaNet-hybrid-3-1

python -m alloy.examples.package_config_for_hub \
    --config /path/to/alloy/examples/configs/qwen3_next_340m_dense.json \
    --target $TARGET \
    --tokenizer-src $SRC_TOKENIZER
```

This writes into `$TARGET`:
- `config.json` — the JSON form of the architecture
- `modeling_alloy.py` — 1-line shim, `from alloy import AlloyForCausalLM`
- `auto_map` injected into config.json so AutoModelForCausalLM resolves
- Tokenizer files (`tokenizer.json`, `tokenizer_config.json`,
  `special_tokens_map.json`, plus any of `tokenizer.model` / `vocab.json` /
  `merges.txt` / `added_tokens.json` / `chat_template.jinja` /
  `generation_config.json` that exist in the source). Files absent from
  the source are skipped silently.

`--tokenizer-src` is optional. Without it, the script still produces the
config + modeling shim and prints a one-line note that you'll need to
copy tokenizer files manually before training.

### 2. Verify the directory loads via HF AutoConfig

```bash
python -c "
from transformers import AutoConfig
cfg = AutoConfig.from_pretrained('$TARGET', trust_remote_code=True)
print(f'OK: {type(cfg).__name__}, layers={cfg.num_hidden_layers}, '
      f'hidden={cfg.hidden_size}, vocab={cfg.vocab_size}')
print(f'layer_types head: {cfg.layer_types[:6]}')
"
```

Expected output:
```
OK: AlloyConfig, layers=24, hidden=1024, vocab=32000
layer_types head: ['qwen3_5_gdn', 'qwen3_5_gdn', 'qwen3_5_gdn', 'qwen3_attention', 'qwen3_5_gdn', 'qwen3_5_gdn']
```

### 3. Pick the yaml template + tweak data paths

```bash
cp /path/to/alloy/examples/train/pretrain_alloy_qwen3_next_340m_mindspeed.yaml \
   /path/to/qwen3.5_omni_creative/scripts_qwen3_5/configs/my_qwen3_next_340m.yaml
```

Edit the copied yaml — the lines marked with `# set me`:

```yaml
data:
  dataset_param:
    basic_parameters:
      dataset_dir: /path/to/your/data       # absolute path to your dataset directory
      dataset: train_part1.json             # filename(s); comma-separated for multiple shards
training:
  save: ./intermediate_ckpt                 # checkpoint save directory
```

If your `model_name_or_path` differs from `./hf_models/alloy_qwen3_next_340m`,
update both occurrences and **make sure they are identical**:

- `data.dataset_param.preprocess_parameters.model_name_or_path`
- `model.model_name_or_path`

Both must point at the **same directory** — the one produced by `package_config_for_hub`,
containing `config.json` (with `model_type: alloy` + `auto_map`), `modeling_alloy.py`,
and the tokenizer files. The data side calls `AutoConfig.from_pretrained(...)` on this
path during `load_tokenizer`, so a path that doesn't contain a valid alloy `config.json`
(or pointing one level too high / too low) will fail with
`ValueError: Unrecognized model in <path>. Should have a model_type key in its config.json.`
*before* training even starts.

### 4. Map MindSpeed-LLM bash hyperparameters → yaml

The yaml ships with the values from a typical 340M qwen3-next launcher;
double-check against your bash:

| MindSpeed-LLM bash | yaml field | notes |
|---|---|---|
| `MBS=6` | `training.micro_batch_size: 6` | direct |
| `GBS=48` | `training.gradient_accumulation_steps: <accum>` | `accum = GBS / (MBS × world_size)`. **For 1 node × 8 NPUs, accum = 48/(6×8) = 1**. The yaml ships with 8 (single-card calc) — adjust to your actual world_size. |
| `LR=2e-3` | `training.lr: 2.0e-3` | direct |
| `MIN_LR=2e-4` | `training.lr_min: 2.0e-4` | direct |
| `TRAIN_ITERS=101726` | `training.train_iters: 101726` | direct |
| `SAVE_ITERS=19235` | `training.save_interval: 19235` | direct |
| `SEQ_LENGTH=4096` | `data.basic_parameters.cutoff_len: 4096` | direct |
| `--use-triton-gdn` | `model._qwen3_5_gdn_implementation: triton` | yaml ships `torch`; flip to `triton` to use binder's GDN fast path |
| `--lr-warmup-iters 1024` | `training.lr_warmup_ratio: 0.01` | 1024 / 101726 ≈ 0.01 |

### 5. Adapt the launcher script

Take an existing MindSpeed-MM launcher (e.g.
`scripts_qwen3_5/pretrain-exp4_qwen3_5_10b-a2b_optimized.sh`), change only
the final `torchrun` line to point at your yaml:

```bash
torchrun $DISTRIBUTED_ARGS mindspeed_mm/fsdp/train/trainer.py \
    scripts_qwen3_5/configs/my_qwen3_next_340m.yaml \
    2>&1 | tee ${LOG_DIR}/qwen3_next_340m_${WORLD_SIZE}P_${datename}.log
```

Then launch:

```bash
bash my_qwen3_next_340m_launcher.sh localhost 1 0
```

### 6. Post-training: convert DCP checkpoint → HF format

mindspeed-mm saves checkpoints as torch DCP shards (`<save>/release/__*_*.distcp`),
not HF safetensors. To get an `AutoModelForCausalLM.from_pretrained`-loadable dir,
run:

```bash
python -m alloy.tools.dcp_to_hf \
    --dcp-dir   ./intermediate_ckpt/release \
    --hub-dir   ./hf_models/alloy_qwen3_next_340m \
    --target    ./hf_models/alloy_qwen3_next_340m_trained
```

Loads the DCP shards in-process, repacks as standard sharded safetensors via
`huggingface_hub.split_torch_state_dict_into_shards`, and copies config /
modeling_alloy.py / tokenizer files from `--hub-dir`. Does not need a
pre-existing safetensors index in `--hub-dir` (unlike mindspeed-mm's own
`checkpoint/common/merge_dcp_to_hf.py`, which keys its sharding off one).

---

## General workflow (any architecture)

For a new alloy architecture (different vocab, layer count, mix of layer
types, MoE config):

### 1. Write or pick a config JSON

Use one of `examples/configs/qwen3_*.json` as a starting point, or build
one programmatically:

```python
from alloy import AlloyConfig
cfg = AlloyConfig(
    vocab_size=151936,
    hidden_size=2048,
    num_hidden_layers=24,
    num_attention_heads=16,
    num_key_value_heads=2,
    head_dim=128,
    layer_types=["qwen3_5_gdn"] * 18 + ["qwen3_attention"] * 6,    # canonical 3:1
    ffn_types=["qwen3_5_moe"] * 24,
    num_experts=64,
    num_experts_per_tok=8,
    moe_intermediate_size=512,
    shared_expert_intermediate_size=512,
    rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.25},
    rms_norm_unit_offset=True,
    attn_output_gate=True,
)
import json
with open("my_config.json", "w") as f:
    json.dump(cfg.to_dict(), f, indent=2)
```

### 2. Run `package_config_for_hub` against it

Same command as in step 1 of the walkthrough — feeds the JSON in, drops
config.json + modeling shim + auto_map at the target. Pass
``--tokenizer-src /path/to/tokenizer/dir`` to also copy tokenizer files
into the target in one shot; omit the flag if you want to copy tokenizer
files yourself later.

### 3. Pick the matching yaml template, adjust `hook_modules` ranges

Yaml templates ship with `model.layers.{0-N}` ranges that need to match
your `num_hidden_layers`:

```yaml
parallel:
  fsdp_plan:
    hook_modules:
      - model.layers.{0-23}     # for 24 layers — adjust to your N-1
  recompute_plan:
    apply_modules:
      - model.layers.{0-23}     # same
```

---

## What each yaml field controls (and how it reaches alloy)

| yaml field | Reaches alloy via | Meaning |
|---|---|---|
| `model.model_id: alloy` | `model_register.get("alloy")` → `AlloyForCausalLMPlugin` | which model class to build |
| `model.model_name_or_path` | `AutoConfig.from_pretrained(...)` | where `config.json` + `modeling_alloy.py` live |
| `model.trust_remote_code: true` | `AutoConfig.from_pretrained(..., trust_remote_code=True)` | required to load `auto_map` |
| `model.attn_implementation` | `AutoConfig.from_pretrained(_attn_implementation=...)` | HF standard attn dispatch (`eager` / `sdpa` / `flash_attention_2`) |
| `model._qwen3_5_gdn_implementation` | `overwrite_transformer_config` → `transformer_config._qwen3_5_gdn_implementation` | alloy GDN backend (`torch` / `triton` / `flash`) |
| `model._experts_implementation` | same | HF / alloy MoE dispatch (`eager` / `grouped_mm` / `batched_mm` / `flash`) |
| `parallel.fsdp_plan.apply_modules` | mindspeed FSDP wrapping | which alloy submodules get sharded |
| `parallel.ep_plan.apply_modules` | mindspeed EP wrapping | which alloy submodules participate in expert parallel |

The plugin shim's pattern for `_<key>_implementation` means you can drop
new alloy modules with their own dispatch surfaces in alloy core, then
expose them in yaml without editing the shim.

---

## Switching backends without retraining

Once the model is frozen in `./hf_models/alloy_*/`:

```yaml
# binder OFF (alloy default torch dispatch — byte-exact baseline)
model:
  _qwen3_5_gdn_implementation: torch
  _experts_implementation: eager

# binder ON (NPU fast path)
model:
  _qwen3_5_gdn_implementation: flash      # GDN fused: ascendc + triton
  _experts_implementation: flash          # MoE fused: permute + GMM + swiglu + GMM + unpermute

# pure triton GDN (fallback when ascendc kernels unavailable)
model:
  _qwen3_5_gdn_implementation: triton
  _experts_implementation: eager
```

These are pure yaml knobs. **No alloy code edits, no model dir regeneration.**
That's the whole point of having alloy's `_<module>_implementation` fields
runtime-only and never serialised into `config.json`.

---

## FSDP plan caveats

Alloy's module paths differ from `Qwen3_5MoeForConditionalGeneration` (which
wraps a vision encoder + a language_model). Alloy is text-only, so:

- `model.layers.{*}` — not `model.language_model.layers.{*}`
- `model.layers.{*}.linear_attn` — alloy's GDN attribute name (matches the
  registry's `attr_name`)
- `model.layers.{*}.self_attn` — full-attention layers
- `model.layers.{*}.mlp` — FFN (dense MLP or `Qwen35SparseMoE` depending on
  layer's `ffn_types[i]`)
- `model.layers.{*}.mlp.experts` — only on MoE layers
- `model.embed_tokens`, `model.norm`, `lm_head` — top-level

The `hook_modules` and `recompute_plan.apply_modules` ranges
(`model.layers.{0-N}`) must match `num_hidden_layers` in your
`./hf_models/alloy_*/config.json`.

To verify the patterns against an actual constructed model:

```python
from transformers import AutoConfig, AutoModelForCausalLM
cfg = AutoConfig.from_pretrained("./hf_models/alloy_qwen3_next_340m", trust_remote_code=True)
m = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True)
for name, _ in m.named_modules():
    print(name)
```

---

## Troubleshooting

**`ValueError: Unrecognized model in <path>. Should have a model_type key in
its config.json.`** — raised from `load_tokenizer` → `AutoConfig.from_pretrained`
on the data side. Two common causes:

1. `data.dataset_param.preprocess_parameters.model_name_or_path` and
   `model.model_name_or_path` point at different directories. The data side
   reads the alloy `config.json` from its own path; if it points one level
   above (or below) the dir produced by `package_config_for_hub`, there's no
   valid `config.json` there. Fix: make both yaml fields **identical**.
2. The directory exists but its `config.json` lacks `model_type: alloy` /
   `auto_map`. Re-run `package_config_for_hub` to regenerate, then verify
   with `python -c "import json; print(json.load(open('<dir>/config.json')).get('model_type'))"`.

**`KeyError: 'alloy' is not registered`** — the plugin import never ran.
Check `training.plugin` includes `alloy.integrations.mindspeed_mm` and
that alloy is `pip install -e`'d into the env.

**`ValueError: model_id 'alloy' is not registered in MODEL_MAPPINGS`** — same
cause, plugin not imported. Verify by running

```bash
python -c "from alloy.integrations.mindspeed_mm import AlloyForCausalLMPlugin; print('OK')"
```

**`KeyError: 'flash' is not a valid experts implementation`** — bytedance fork's
`ExpertsInterface.get_interface` is strict; the binder bridge needs to import
*before* the model is built. Add `alloy.integrations.hf_npu_binder` to the
yaml's `training.plugin` list (mindspeed imports plugins in order during
`Trainer.initialize()`).

**`ValueError: Specified experts_implementation="flash" is not supported`** —
older transformers fork (5.2.0.dev0) has a hardcoded whitelist in
`get_correct_experts_implementation`. alloy's `AlloyPreTrainedModel` already
overrides this, but only for transformers v5.7+ — upgrade if seen.

**RuntimeError: Input dtype mismatch in conv1d (input fp32, weight bf16)** —
also a stale transformers fork. v5.7+ fixes the autocast path.

**Alloy module paths don't match `apply_modules` patterns** — print
`named_modules()` of the constructed model (snippet under "FSDP plan caveats")
and update the yaml patterns to match exactly.
