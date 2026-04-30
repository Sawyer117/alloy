# MindSpeed-MM patches required to train alloy

Alloy is HF-native (HF transformers conventions, `auto_map`, `trust_remote_code`).
ByteDance's MindSpeed-MM FSDP2 trainer mostly works with HF-native models as-is,
but three places in the trainer pre-date or violate HF conventions and need
small patches before alloy training can run end-to-end. All three are upstream
gaps, not alloy-specific issues — any HF-native CausalLM under MindSpeed-MM
hits the same.

Document maintained against MindSpeed-MM at the version shipped in
`qwen3.5_omni_creative` (April 2026). Re-verify after a sync from upstream:
if either patch site has been refactored or already fixed, drop the entry here.

---

## Patch 1 — dataloader: detect file format from extension

**File**: `mindspeed_mm/fsdp/data/datasets/huggingface/qwen2vl_dataset.py`
**Site**: ~line 55, the `train_dataset = load_dataset(...)` call (and the
`val_dataset` block right below it, if validation is used).

### Why

The current code hardcodes `path="json"` in HF `load_dataset`, so any data
format other than `.json` / `.jsonl` (e.g. parquet shards from fineweb-edu,
nemotron, dolma) cannot be loaded — the JSON parser tries to read parquet
bytes and fails.

### Patch

Add a tiny extension sniffer and dispatch to the matching HF loader:

```python
def _detect_path(files):
    if isinstance(files, str):
        files = files.split(",")
    if any(f.strip().endswith(".parquet") for f in files):
        return "parquet"
    return "json"   # default keeps existing .json / .jsonl behavior

train_dataset = load_dataset(
    path=_detect_path(data_args.dataset),       # ← was: path="json"
    data_files=data_args.dataset,
    split="train",
    cache_dir=data_args.cache_dir,
    streaming=data_args.streaming,
)
```

Apply the same change to the `val_dataset` block immediately below if you use
validation.

### Yaml after patch

`dataset` accepts a glob; HF expands it and sorts. 100 parquet shards is one line:

```yaml
basic_parameters:
  dataset_dir: /path/to/parquet_folder       # informational; not used for text
  dataset: "/path/to/parquet_folder/*.parquet"
```

Mixing .parquet with .json/.jsonl in a single `dataset` value is not supported
(the sniffer returns one path type). Keep one format per glob.

### What it doesn't cover

`.csv` and `.arrow` still fall through to the `"json"` fallback and fail. If
you need those, extend `_detect_path` with explicit branches; for parquet+json
only, the above is sufficient.

---

## Patch 2 — loss_func: accept HF's calling-convention kwargs

**File**: `mindspeed_mm/fsdp/loss/loss_func.py`
**Site**: line 85, the non-chunk-loss `loss_func` closure inside
`build_loss_func`.

### Why

HF transformers' standard CausalLM forward (Qwen2 / Qwen3 / LLaMA / all of
them) calls the loss as:

```python
loss = self.loss_function(
    logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs
)
```

where `**kwargs` carries trainer-side flags the loss may or may not need
(`num_items_in_batch`, `output_router_logits`, `shift_labels`, …). HF's
default `ForCausalLMLoss` declares `**kwargs` as a sink, so the convention
is "the loss accepts and ignores extras it doesn't care about".

MindSpeed-MM's `loss_func` replaces `self.loss_function` but its signature
is strict:

```python
def loss_func(logits, labels=None, vocab_size=None):     # ← no **kwargs
```

Any HF-native model under MindSpeed-MM crashes at the first batch with:

```
TypeError: build_loss_func.<locals>.loss_func() got an unexpected keyword
argument 'output_router_logits'
```

### Patch

Add a `**kwargs` sink — accept and ignore. No semantic change for non-MoE
models; for MoE, alloy's aux loss is configured via `model.loss_cfg.router_aux_loss_coef`
in yaml, not passed via this path, so swallowing `output_router_logits` here
is safe.

```python
def loss_func(logits, labels=None, vocab_size=None, **kwargs):     # ← + **kwargs
    logits = logits.view(-1, logits.shape[-1]).contiguous().float()
    labels = shift_labels.view(-1)
    return fixed_cross_entropy(
        logits, labels,
        alpha=alpha,
        ...
    )
```

If you use `enable_chunk_loss: true`, also add `**kwargs` to the
chunk-loss branch (`def loss_func(hidden_states, head_weight, head_bias):`
in the same file's other arm) for symmetry — alloy's default templates ship
with `enable_chunk_loss: false`, so we have only seen the non-chunk branch
fail in practice.

---

## Patch 3 — train_engine: don't assume `output.aux_loss` exists

**File**: `mindspeed_mm/fsdp/train/train_engine.py`
**Site**: line 102, inside `train_step`'s gradient-accumulation loop.

### Why

The current code is:

```python
output = self.model(**batch_data)
loss = output.loss / args.training.gradient_accumulation_steps
if output.aux_loss is not None:                     # ← assumes attr exists
    aux_loss = output.aux_loss / args.training.gradient_accumulation_steps
else:
    aux_loss = 0
```

The `is not None` check is written as if `aux_loss` were always present and
just optionally `None`. In HF transformers, `aux_loss` lives on
`MoeCausalLMOutputWithPast` (MoE-only), not on the standard
`CausalLMOutputWithPast` that dense models return. Any HF-native dense
model — including alloy's qwen3-next-340m / qwen3-dense templates —
crashes here with:

```
AttributeError: 'CausalLMOutputWithPast' object has no attribute 'aux_loss'
```

This is the same shape of bug as Patch 2 (assuming HF convention guarantees
something it doesn't): the model is allowed to return any output dataclass
matching its task, and the trainer must read fields defensively.

### Patch

Use `getattr` with a default. One-line change:

```python
output = self.model(**batch_data)
loss = output.loss / args.training.gradient_accumulation_steps
aux = getattr(output, "aux_loss", None)            # ← defensive
if aux is not None:
    aux_loss = aux / args.training.gradient_accumulation_steps
else:
    aux_loss = 0
```

No semantic change for MoE models that *do* return `aux_loss` — `getattr`
returns the actual value when the attribute exists. Dense models that
return plain `CausalLMOutputWithPast` get `aux=None` and the else-branch.

### Note for alloy MoE training

Alloy's `AlloyForCausalLM.forward` currently returns
`CausalLMOutputWithPast` regardless of MoE-ness — the router-aux loss is
expected to be configured via yaml's `model.loss_cfg.router_aux_loss_coef`
and computed inside the loss path, not propagated up via the model output.
If MoE training shows aux-loss = 0 in logs even though
`router_aux_loss_coef > 0`, the loss path isn't picking it up — separate
investigation, not covered by these patches.

---

## Verification

After both patches are applied, restart the trainer:

```bash
bash <your_launcher>.sh localhost 1 0
```

Expected progression past the original failure points:

1. `[Rank 0] Prepare data` — past Patch 1 (dataloader doesn't crash on
   parquet glob).
2. First iteration's loss printed within ~1–2 minutes of dataloader warmup
   — past Patch 2 (loss_func absorbs trainer kwargs).
3. Healthy from-scratch pretraining loss for vocab_size=32000 is roughly
   `ln(32000) ≈ 10.4`. Numbers in the 10–11 range = sane init; <8 or >12
   = something off (data column mismatch, weight init scale, etc.).

If you crash at the loss site with a *different* `TypeError` keyword, the
trainer is passing yet another HF-convention kwarg the closure doesn't
accept — same fix, the `**kwargs` sink already added in Patch 2 should
cover it. If it doesn't, the kwarg is being injected somewhere other than
`**batch_data` and warrants a separate investigation.

---

## Long-term direction

These are upstream fixes. When ByteDance accepts patches on the
`qwen2vl_dataset.py` and `loss_func.py` sites (or refactors them away),
this document can be deleted. Until then, anyone training an alloy or
other HF-native model with MindSpeed-MM FSDP2 needs both patches.
