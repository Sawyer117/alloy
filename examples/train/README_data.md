# Dataset notes for MindSpeed-MM training

Notes on how to fill `data.dataset_param.basic_parameters.{dataset_dir,dataset}`
in the trainer yaml, what file formats the loader actually supports, and how
to handle parquet input (which the default loader does not support).

## What `dataset_dir` and `dataset` actually mean

Looking at the code in
`mindspeed_mm/fsdp/data/datasets/huggingface/qwen2vl_dataset.py`:

```python
train_dataset = load_dataset(
    path="json",                          # ← HARDCODED
    data_files=data_args.dataset,         # ← yaml `dataset` field, passed verbatim
    split="train",
    cache_dir=data_args.cache_dir,
    streaming=data_args.streaming,
)
```

So:

- **`dataset`** is passed straight into HuggingFace `load_dataset(data_files=...)`.
  It can be:
  - a single absolute / relative file path: `dataset: /abs/path/file.jsonl`
  - a comma-separated list of paths: `dataset: /a/part_1.jsonl,/a/part_2.jsonl`
  - a glob (if HF datasets accepts it for the path type — works for files,
    not always for directories): `dataset: /a/part_*.jsonl`
  - a single filename relative to wherever the trainer's CWD is (avoid this —
    use absolute paths).

- **`dataset_dir`** is **NOT prepended to `dataset`**. Search for `dataset_dir`
  in the loader code and you'll only find it under
  `_find_media_files` in `convert.py:76` — i.e. it's a directory prefix used
  exclusively for **multimedia** files (images / videos / audios). For
  text-only pretraining it does nothing functional. Many example yamls set
  it to a value anyway as documentation; the trainer doesn't consume it.

- **`path="json"`** is hardcoded. The loader will try to parse every file
  in `data_files` as JSON. **It does NOT auto-detect parquet / csv / arrow.**

## File formats: what works and what doesn't

| Format | Supported by default? | Notes |
|---|---|---|
| `.json` | ✅ | One JSON object per file (with a top-level array) or whole-file dict. |
| `.jsonl` | ✅ | One JSON object per line. The recommended format for pretraining text. |
| `.parquet` | ❌ | `path="json"` will fail to parse. See "Handling parquet input" below. |
| `.csv` | ❌ | Same reason. |
| `.arrow` | ❌ | Same reason. |

The reference yaml under `scripts_qwen3_5/configs/qwen3_5_10B-A2B_exp4_config.yaml`
ships:

```yaml
dataset_dir: /mnt/.../data/300B
dataset: 300B_part1_of_9.json,300B_part2_of_9.json,300B_part3_of_9.json
```

`dataset_dir` here is purely informational. `dataset` lists three JSON files
(absolute paths in practice — relative to wherever the trainer is launched).

## Handling parquet input

If you have a folder of `.parquet` files (e.g. fineweb-edu-20BT released as
parquet shards), the default loader cannot read them. Three options, listed
from least to most invasive.

### Option A — convert parquet → jsonl (recommended for first runs)

One-time preprocessing. Stable, zero code changes, and you end up with a
format that matches the example yamls.

```bash
SRC=/path/to/your/parquet_folder      # 20 .parquet files in here
DST=/path/to/your/jsonl_folder        # output jsonl shards

mkdir -p "$DST"
python <<'PY'
import os, glob, json, pyarrow.parquet as pq
src = os.environ["SRC"]; dst = os.environ["DST"]
for f in sorted(glob.glob(f"{src}/*.parquet")):
    name = os.path.basename(f).replace(".parquet", ".jsonl")
    out  = os.path.join(dst, name)
    table = pq.read_table(f)
    with open(out, "w", encoding="utf-8") as fp:
        for row in table.to_pylist():
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"{f} -> {out} ({table.num_rows} rows)")
PY
```

Inspect the column names first so you know what to set as the text key in
yaml (`attr.prompt`):

```bash
python -c "
import pyarrow.parquet as pq, glob
f = sorted(glob.glob('$SRC/*.parquet'))[0]
t = pq.read_table(f)
print('columns:', t.column_names)
print('first row keys:', list(t.to_pylist()[0].keys()))
"
```

If the text column is named `"text"`, the yaml's default `attr.prompt: text`
already matches. If it's named e.g. `"content"`, set `attr.prompt: content`.

Then write yaml:

```yaml
data:
  dataset_param:
    dataset_type: huggingface
    attr:
      formatting: alpaca
      pretrain: true
      prompt: text                       # ← column name in your jsonl
    basic_parameters:
      dataset_dir: /path/to/your/jsonl_folder    # purely informational
      dataset: /path/to/your/jsonl_folder/part_1.jsonl,/path/to/your/jsonl_folder/part_2.jsonl
      cutoff_len: 4096
      packing: true
      stage: pretrain
      ...
```

If the comma-separated list gets unwieldy with 20 files, concatenate them
into one big file:

```bash
cat /path/to/your/jsonl_folder/*.jsonl > /path/to/your/all.jsonl
# yaml then becomes:  dataset: /path/to/your/all.jsonl
```

### Option B — patch one line in mindspeed-mm

Edit `mindspeed_mm/fsdp/data/datasets/huggingface/qwen2vl_dataset.py` line 55
to detect the file extension:

```python
def _detect_path(files):
    if isinstance(files, str):
        files = files.split(",")
    if any(f.strip().endswith(".parquet") for f in files):
        return "parquet"
    return "json"

train_dataset = load_dataset(
    path=_detect_path(data_args.dataset),
    data_files=data_args.dataset,
    split="train",
    cache_dir=data_args.cache_dir,
    streaming=data_args.streaming,
)
```

Apply the same change to the `val_dataset` block right below if you use
validation. Pros: one-line behavior change. Cons: you maintain a patch on
mindspeed-mm; updates upstream may conflict.

### Option C — write an alloy plugin that registers a parquet loader

The cleanest separation. mindspeed-mm has the same `data_register` mechanism
as `model_register`, and yaml's `data.dataset_param.dataset_type` field
selects which registered loader to use. We would ship something like
`alloy/integrations/mindspeed_mm_parquet.py`:

```python
from datasets import load_dataset
from mindspeed_mm.fsdp.utils.register import data_register

@data_register.register("alloy_parquet")
def get_alloy_parquet_dataset(basic_param, preprocess_param, dataset_param, **kwargs):
    # Same flow as get_qwen2vl_dataset but with path="parquet" and glob
    # support so dataset_dir + dataset can be combined to scan a directory.
    ...
```

yaml then says:

```yaml
data:
  dataset_param:
    dataset_type: alloy_parquet           # ← registered by the plugin import
    basic_parameters:
      dataset_dir: /path/to/your/parquet_folder
      dataset: "*.parquet"
training:
  plugin:
    - alloy.integrations.mindspeed_mm
    - alloy.integrations.hf_npu_binder
    - alloy.integrations.mindspeed_mm_parquet     # ← add this
```

This is ~150 lines (most copied/adapted from `qwen2vl_dataset.py`). Worth
doing once parquet becomes a recurring need.

## Sanity checks before launching

After preparing the data, verify with HF datasets in isolation — same call
the trainer will make:

```bash
python -c "
from datasets import load_dataset
ds = load_dataset(path='json', data_files='/path/to/your/all.jsonl', split='train', streaming=False)
print(f'rows: {len(ds)}')
print(f'columns: {ds.column_names}')
print(f'first row: {ds[0]}')
"
```

Common failure modes:

- **Empty `text`**: the loader runs but every row tokenizes to an empty
  sequence; loss starts at 0 and stays there. Check `attr.prompt` matches
  the actual column name.
- **Doc lengths much shorter than `cutoff_len`**: with `packing: true`, this
  is normal — the dataloader concatenates short docs to fill `cutoff_len`.
  With `packing: false`, you'd waste a lot of compute on padding.
- **Loader hangs on first epoch**: usually means the cache directory is
  on a slow / NFS-mounted volume and the first-pass tokenization is slow.
  Set `cache_dir` to a local SSD.
- **`KeyError` on column lookup**: a few rows have `null` for the text
  column. Filter beforehand:

  ```python
  ds = ds.filter(lambda r: isinstance(r.get('text'), str) and r['text'].strip() != '')
  ```

## Quick reference: yaml fields for data

| field | meaning | typical value |
|---|---|---|
| `dataset_type` | which `data_register`-ed loader to use | `huggingface` (default) |
| `attr.formatting` | data converter family | `alpaca` for pretrain text |
| `attr.pretrain` | flag on alpaca converter | `true` for pretraining |
| `attr.prompt` | column name holding the text | `text` (or whatever your file uses) |
| `basic_parameters.dataset_dir` | media file root (NOT used for text) | doesn't matter; set to data folder for documentation |
| `basic_parameters.dataset` | comma-separated file paths | absolute paths to your `.json` / `.jsonl` files |
| `basic_parameters.cutoff_len` | max sequence length | match `SEQ_LENGTH` in your launcher / yaml |
| `basic_parameters.packing` | concat short docs to fill `cutoff_len` | `true` for pretraining |
| `basic_parameters.stage` | pipeline stage | `pretrain` |
| `basic_parameters.cache_dir` | tokenization cache | a local writable dir; first pass is slow |
| `basic_parameters.overwrite_cache` | rerun tokenization | `false`; flip to true if you change `attr.prompt` or the data |
| `dataloader_param.collate_param.model_name` | collator family | `llm_pretrain` |
