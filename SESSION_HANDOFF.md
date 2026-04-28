# SESSION HANDOFF — alloy

This file is a snapshot of the working state of the `alloy` project as of the
last commit on this branch (`wip/session-handoff`). It exists so a new
Claude Code session on a different machine can pick up roughly where this one
left off without re-litigating decisions or losing accumulated context.

---

## How to use this file (on the new machine)

```bash
git clone https://github.com/Sawyer117/alloy.git
cd alloy
git checkout wip/session-handoff
cat SESSION_HANDOFF.md      # read the whole thing
git checkout main           # switch back to main for actual work
```

In a fresh Claude Code session, point it at this file:

> Read `alloy/SESSION_HANDOFF.md` end-to-end. Don't act yet — summarise back
> the architectural decisions, open threads, and immediate next-action queue,
> then ask which thread I want to pick up.

The "Memory dump" section at the end contains the 7 memory entries this
session accumulated. The new session can either re-create them in its own
`<claude_home>/projects/<workspace>/memory/` directory verbatim, or just keep
them in mind by referencing this file.

---

## Repo state — what's on `main`

Last commits (newest first):

```
27b716f Add MindSpeed-LLM FSDP2 launch artifacts for alloy_340m_dense
5f35f6c Add pyproject.toml so alloy is pip-installable
8f1647a Add Qwen3-Next 340M dense config example
fc0571a Add Qwen3-Next 1.3B dense config example (qwen3.5 GDN + qwen3 dense MLP)
2c162f7 Decouple _init_weights from registered module classes via per-class hook
65f9853 Rename source-coupled module files to <source>_<kind>.py
378956c Add example for config-only HF Hub repo workflow
a7df566 Add tools.export_for_hub for HF Hub distribution
ddfed6d Override _can_set_experts_implementation so alloy picks the same MoE backend as HF
54af56e Add sub-module drift diagnostic for pinpointing layer-0 divergence
6004d39 Port HF eager Qwen3_5MoeExperts.forward verbatim for byte-exact match
b57faf1 Add per-layer NPU drift diagnostic vs HF reference
d2e3c9b Soften qwen3.5 NPU PASS criterion under layer truncation
5e67231 Make HF mask builder calls portable across transformers versions
dc79cee Restyle README
```

Current top-level layout:

```
alloy/
├── README.md
├── SESSION_HANDOFF.md         (this file, only on wip/session-handoff)
├── pyproject.toml             (pip install target — repo IS the package)
├── __init__.py
├── configuration_alloy.py     (AlloyConfig)
├── modeling_alloy.py          (AlloyForCausalLM, AlloyModel, AlloyDecoderLayer)
├── loading.py                 (build_skeleton, build_on_device, load_state_dict_from_disk)
├── modules/
│   ├── registry.py            (MIXER_REGISTRY, FFN_REGISTRY, MixerEntry w/ mask_kind)
│   ├── attention/
│   │   ├── qwen3_attention.py        (Qwen3Attention — qwen3 / qwen3.5 GQA, attn_output_gate flag)
│   │   └── qwen3_5_gdn.py            (Qwen35GatedDeltaNet)
│   ├── ffn/
│   │   ├── qwen3_mlp.py               (Qwen3MLP — SwiGLU dense)
│   │   └── qwen3_5_moe.py             (Qwen35SparseMoE + _Experts + _TopKRouter)
│   └── shared/
│       ├── norm.py                    (RMSNorm w/ unit_offset, RMSNormGated)
│       ├── rotary.py                  (RotaryEmbedding w/ partial + interleaved-mRoPE)
│       └── attention_kernels.py       (eager_attention_forward, repeat_kv)
├── tools/
│   ├── __init__.py
│   └── export_for_hub.py              (write modeling_alloy.py shim + auto_map into a target dir)
├── examples/
│   ├── README.md
│   ├── build_from_config.py
│   ├── build_from_python.py
│   ├── package_config_for_hub.py     (config-only -> HF Hub repo demo)
│   ├── configs/
│   │   ├── qwen3_4b.json
│   │   ├── qwen3_5_35b_a3b.json
│   │   ├── qwen3_next_340m_dense.json
│   │   └── qwen3_next_1_3b_dense.json
│   └── train/
│       ├── pretrain_alloy_340m_fsdp2.yaml
│       └── pretrain_alloy_340m_fsdp2.sh
├── scripts/
│   ├── compare_qwen3.py
│   └── compare_qwen3_5.py
└── tests/
    ├── _compare_utils.py
    ├── test_construct.py              (hardware-agnostic smoke)
    ├── gpu/
    │   ├── compare_qwen3_pretrained.py
    │   └── compare_qwen3_5_pretrained.py
    └── npu/
        ├── compare_qwen3_pretrained.py
        ├── compare_qwen3_5_pretrained.py
        ├── debug_layerwise_diff.py    (per-layer drift hook diagnostic)
        └── debug_sublayer_diff.py     (sub-module drift hook diagnostic)
```

---

## Architectural decisions baked in (don't relitigate)

These are decisions the user explicitly committed to; new sessions should
take them as given.

### 1. Source-coupled naming

**Registry key == file basename == class name (modulo case/underscores).**

| File | Class | Registry key |
|---|---|---|
| `attention/qwen3_attention.py` | `Qwen3Attention` | `qwen3_attention` / `qwen3_attention_sliding` |
| `attention/qwen3_5_gdn.py` | `Qwen35GatedDeltaNet` | `qwen3_5_gdn` |
| `ffn/qwen3_mlp.py` | `Qwen3MLP` | `qwen3_mlp` |
| `ffn/qwen3_5_moe.py` | `Qwen35SparseMoE` | `qwen3_5_moe` |

Future ports (`bert_mlp.py`, `mamba_ssm.py`, etc.) follow the same convention.

`shared/{norm,rotary,attention_kernels}.py` intentionally stay generic — they're
parameterised cross-source primitives (e.g. `RMSNorm` covers both qwen3 and
qwen3.5 styles via `unit_offset` flag).

### 2. Per-class `_alloy_init_weights` hook decouples init

`AlloyPreTrainedModel._init_weights(module)` does:

1. Look for `module._alloy_init_weights(init_std)` — if present, call it and return.
2. Else fall back to stdlib type dispatch (`nn.Linear` / `nn.Embedding` / `nn.Conv1d`).

So adding a new mixer / FFN with special init NEVER requires editing
`modeling_alloy.py`. The hook name has the `_alloy_` prefix to avoid
colliding with HF's `PreTrainedModel.init_weights()` (recursive driver).

Currently classes that own special init: `RMSNorm`, `RMSNormGated`,
`Qwen35GatedDeltaNet`, `_Experts`, `_TopKRouter`. All inits verified
byte-aligned with HF v5.6.2.

### 3. Mask routing is `mask_kind`-based, not string-compared

`MixerEntry` carries `mask_kind: "causal" | "sliding" | "linear"`. Model-level
mask precompute reads this from the registry and builds exactly one mask per
distinct kind. Adding a new mixer never edits `modeling_alloy.py`'s mask logic.

### 4. MoE experts go through HF's shared dispatch

`AlloyPreTrainedModel._can_set_experts_implementation` is overridden to
return `True` (HF's stock heuristic greps the modeling file for the literal
`@use_experts_implementation` string and would return False for alloy because
the decorator lives on `_Experts` in `modules/ffn/qwen3_5_moe.py`, not in
`modeling_alloy.py`). Without this override, alloy on NPU dispatches to its
own per-expert eager forward while HF reference dispatches to
`grouped_mm_experts_forward` — same math, different op composition,
1-fp32-ulp drift per MoE layer. With the override, both sides hit the same
shared backend and we get byte-exact match.

### 5. HuggingFace alignment is paramount

This was tested empirically on NPU (Ascend 910B-class card, transformers
v5.6.2):

```
embed                          0.0u  ✅
L0/input_layernorm             0.0u  ✅
L0/mixer (linear_attn)         0.0u  ✅  GDN byte-exact
L0/post_attention_layernorm    0.0u  ✅
L0/mlp                         0.0u  ✅  MoE byte-exact (after the fix in commit ddfed6d)
L0/output                      0.0u  ✅
RESULT: byte-exact at every sub-module. alloy == HF in fp32.
```

This is the alloy quality bar. Future changes that introduce drift relative
to HF for the same shipped configs are regressions, not acceptable optimisations.

### 6. Cross-hardware policy (current and planned)

- **Model code** (`modeling_alloy.py`, `modules/**`) imports zero hardware-specific
  symbols. No `torch_npu`, no fla, no causal-conv1d.
- **Optional fast paths** are NOT pulled in via try-import inside model code.
  (Counter-pattern: HF's `Qwen3_5MoeGatedDeltaNet` does `chunk_gated_delta_rule
  or torch_chunk_gated_delta_rule` based on whether fla is installed — alloy
  intentionally does NOT do this. Default behavior must be deterministic and
  reproducible.)
- **NPU fused kernels** are planned to land in a separate `alloy-npu` package
  (sibling repo / package). It will register backends into alloy's existing
  registries on import. **Not started yet.**

### 7. Distribution: pip install (route A), not vendoring

Decided after a brief detour into route B (vendor entire alloy package source
into each HF Hub repo). HF's `trust_remote_code` loader has limitations with
multi-file packages — its relative-import scanner doesn't recurse and treats
nested package paths as flat filenames with dots. Route A sidesteps all that:

- alloy has `pyproject.toml`, installable via `pip install
  git+https://github.com/Sawyer117/alloy.git`
- HF Hub model repos contain only `config.json` (with auto_map) + a one-line
  `modeling_alloy.py` shim (`from alloy import AlloyConfig, AlloyForCausalLM`)
- Consumers `pip install alloy` once, then any alloy model on Hub loads via
  `AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)`

`alloy/tools/export_for_hub.py` produces these directories.

### 8. Test scope discipline

A smoke test verifies the function it's named for. Don't bolt on round-trip
checks that exercise orthogonal mechanisms. Specifically: weight round-trip
on a randomly-initialised model is not a meaningful test of anything alloy
ships, because weights come from external training frameworks. (See
`feedback_test_scope_focus.md` in the memory dump below.)

### 9. Commit hygiene

- **No `Co-Authored-By: Claude` trailers, ever.** User had to rewrite git
  history once already to remove them. Identity-sensitive workplace context.
- Commit messages: subject + body, that's it.
- Commits use the repo-local identity `Sawyer117 <Sawyer117@users.noreply.github.com>`.

---

## Open threads (priority-ordered)

### Thread 1 — first NPU FSDP2 training run (HOT)

The launch artifacts at `examples/train/pretrain_alloy_340m_fsdp2.{yaml,sh}`
have not yet been smoke-tested on real NPU hardware. The user is about to
attempt this on the next-machine session.

User pre-conditions:
- alloy installed via `pip install git+https://github.com/Sawyer117/alloy.git`
- model directory at some path (`/home/.../my_alloy_340m/`) containing
  `config.json` + `modeling_alloy.py` shim (from `package_config_for_hub`)
  + tokenizer files (copied from the user's existing 340M tokenizer dir,
  specifically `tokenizer.json`, `tokenizer_config.json`,
  `special_tokens_map.json` — and any other non-weight files like
  `tokenizer.model` / `vocab.json` / `merges.txt` / `added_tokens.json` /
  `chat_template.jinja` if present)
- HF-format dataset (parquet/json/jsonl). Can be either a single file or a
  directory path (megatron data manager `glob`s the directory one level deep
  and expects same-format files).
- `MindSpeed-LLM` repo cloned somewhere; the launch `.sh` `cd`s into it.

Likely first-run gotchas:
- Data preprocessing writes bin/idx cache next to the source data — that
  directory must be writable.
- `optimization.use_triton_gdn` / `use_fused_rmsnorm` are MindSpeed patches
  scoped to MindSpeed's built-in qwen3_next class (via `register_patches`).
  They DON'T reach alloy's classes — the flags are no-ops for alloy. Not an
  error, just no speedup. (This is the gap that `alloy-npu` will eventually
  fill.)
- `use_flash_attn` interaction with alloy's `Qwen3Attention` —
  `_attn_implementation` config field gets set; alloy's GQA does
  `ALL_ATTENTION_FUNCTIONS.get_interface(...)` which will pick the right
  backend if available. Probably fine.

### Thread 2 — interrupted investigation: optimization flags vs alloy

User asked: "我看到别人的HF repo里面 [...] 这些 optimization 段是因为模型类是他自己的模型对吧 我们这种第三方 custom 模型类 能正常运行吗"

Investigation started, was cut short by the user redirecting to a different
question (whether `dataset.file_name` accepts a directory). The half-finished
grep was:

```
use_triton_gdn / use_fused_rmsnorm references in MindSpeed-LLM:
  - models/qwen3_next/modeling_qwen3_next.py:707  -> use_triton_gdn (qwen3_next-specific)
  - models/qwen3_next/modeling_qwen3_next.py:100, 252 -> use_fused_rmsnorm (qwen3_next class)
  - models/qwen3/qwen3.py:108-111 -> use_fused_rmsnorm / use_fused_rotary_pos_emb (qwen3 class)
  - models/qwen3/qwen3_moe.py:154,158 -> same flags, qwen3_moe class
  - models/gpt_oss/modeling_gpt_oss.py:77, 433, 471, 615 -> 各 flag (gpt_oss class)
  - models/minimax_m27/modeling_minimax_m2.py: -> 各 flag (minimax class)
  - utils/arguments.py:712-736 -> the dataclass field declarations themselves
  - train/trainer.py:641-642 -> chunk_loss_size (used by trainer regardless of model)
```

**Tentative conclusion** (not user-confirmed):
- `chunk_loss_size`: trainer-level, applies to alloy too. Real benefit.
- `use_flash_attn`: probably config-driven via `_attn_implementation`,
  applies to alloy too if alloy attention reads that field. (alloy's
  `Qwen3Attention` does read `_attn_implementation`.)
- `use_triton_gdn`, `use_fused_rmsnorm`, `use_fused_rotary_pos_emb`: applied
  inside specific MindSpeed model class `__init__`s via direct attribute
  swap. They reach the MindSpeed-built-in classes, NOT alloy classes. No-op
  for alloy.

The new session should finish this investigation by reading the actual
forward paths (especially the MindSpeed qwen3_next `register_patches` to
confirm patches are class-scoped, not config-scoped) and reporting concretely
which flags help vs which are dead weight, before the user runs first NPU
training.

### Thread 3 — `alloy-npu` plugin scaffold

Discussed but not started. Design agreed:

- Separate package, e.g. `alloy-npu` repo, sibling of `alloy`
- Imports `alloy` as a dependency
- On import, registers NPU fused implementations into:
  - `ALL_GDN_FUNCTIONS` (planned addition to alloy.modules.registry)
  - Potentially `ALL_NORM_FUNCTIONS`, `ALL_ROTARY_FUNCTIONS`
  - Reuses HF's `ALL_ATTENTION_FUNCTIONS` and `ALL_EXPERTS_FUNCTIONS` for
    those concerns
- Backed by `torch_npu.npu_fusion_attention`, `torch_npu.npu_rms_norm`,
  `torch_npu.npu_apply_rotary_pos_emb`, etc.
- User opts in via `import alloy_npu; alloy_npu.activate()` OR via setting
  config fields like `_gdn_implementation = "npu_fused"`.

Pre-requisite: alloy needs to expose `ALL_GDN_FUNCTIONS` (and similar)
registries for non-HF-covered concerns (HF doesn't have a GDN registry; it
has attention and experts).

When to do it: after first NPU training run is healthy. The fused kernels
are an optimisation; correctness on torch fallbacks is the bar to hit first.

### Thread 4 — `--build-model` end-to-end on NPU

`examples/package_config_for_hub.py --build-model` flag has been verified
to work for a small toy config locally but never tried with one of the actual
shipped configs (340M / 1.3B) on NPU. Quick sanity check before training.

### Thread 5 — README update

The README doesn't yet document:
- `pip install alloy` distribution path
- `tools.export_for_hub` → HF Hub workflow
- `examples/train/` FSDP2 launch pattern

Low priority — works without docs, but would be the next substantive README
diff after thread 1 lands.

### Thread 6 — empirical init-alignment test (deferred)

We verified by reading source that alloy's `_alloy_init_weights` produces
HF-byte-identical init values. An empirical test would be:

```python
torch.manual_seed(0)
alloy_model = AlloyForCausalLM(alloy_cfg)

torch.manual_seed(0)
hf_model = Qwen3_5MoeForCausalLM(hf_cfg)

# diff state_dicts param-by-param, expect byte-exact
```

Useful as a continuous-integration guard against future drift. Not blocking.

---

## Hardware / environment notes

- User's primary dev machine: Windows, Python 3.11, transformers v5.6.2 (at
  `D:/work/2026_cache/transformers/`). HF's `create_causal_mask` here accepts
  `inputs_embeds` (older form).
- User's NPU machine: Ascend 910-class, Linux, transformers v5.6.2. HF's
  `create_causal_mask` here does NOT accept `inputs_embeds` (already-newer
  form; signature drift between releases). alloy works around this via
  `_call_mask_builder` introspection-based filter (commit `5e67231`).
- Behind a corporate proxy with self-signed certs — `git clone` from GitHub
  needs `git config --global http.sslVerify false` or proper CA trust. User
  has been working around it.
- alloy git remote: `https://github.com/Sawyer117/alloy.git`
- alloy commit identity (repo-local): `Sawyer117
  <Sawyer117@users.noreply.github.com>`
- MindSpeed-LLM at `D:/work/MindSpeed-LLM/` on dev machine; assumed cloned
  somewhere on NPU machine (path passed to launch `.sh`).

---

## Memory dump

The 7 memory entries this session accumulated (copied verbatim from
`<dev-machine>/.claude/projects/D--work-model-gym/memory/`). The new session
can either re-create these in its own `<claude_home>/projects/<workspace>/memory/`
directory, or simply take them as guidance from this file.

### `MEMORY.md` (index)

```
- [No Claude co-author trailer](feedback_no_claude_coauthor.md) — never append `Co-Authored-By: Claude` to commit messages; identity-sensitive
- [Each test verifies one concern](feedback_test_scope_focus.md) — don't bolt round-trip / end-to-end checks onto a smoke test whose name targets a single mechanism; weight round-trip on random init isn't a meaningful test
- [Reuse HF: import vs port](feedback_reuse_hf_first.md) — framework utilities get imported from transformers; model-specific classes get copied into alloy as local ports
- [Doc tone by audience surface](feedback_respect_user_expertise.md) — config.json / state_dict / HF-convention artifacts stay minimal; alloy-internal docstrings/READMEs aim at semi-novice users and explain well
- [Alloy project overview](project_alloy.md) — HF-native hybrid-architecture library; package name alloy/; trainer is MindSpeed-LLM FSDP2; NPU support via opt-in patches
- [Hybrid block design pattern](design_hybrid_block.md) — DecoderLayer dispatches mixer/FFN via registry by config.layer_types[i] / ffn_types[i]; mask precompute at model level; HF-canonical, FSDP2-compatible
- [Test layout split by hardware](convention_test_layout.md) — alloy/tests/{gpu,npu}/ separate because NPU can't fit full pretrained weights and uses a different comparison strategy; hardware-agnostic smoke tests at tests/ root
```

### `project_alloy.md`

```
---
name: alloy project overview
description: Core goals, package name, chosen training framework, and HF-native model definition approach for the alloy project
type: project
---
Project name: **alloy** (was called "model_gym" during early scoping; workspace dir at `D:/work/model_gym/` kept for continuity, but the Python package is `alloy/`). The "alloy" metaphor: mixing different token mixers (GQA / GDN / linear attention / Mamba / MoE ...) produces new architectures with emergent properties, like mixing metals into an alloy.

Project goal: 构建一个支持任意混合注意力/FFN 架构按 list 定义灵活组合的 HuggingFace 原生模型库，附带 NPU 高效融合算子 patch，用于科研和跨硬件训练。

**Why:** 用户想要两件事：(1) 跨硬件的 HF 风格模型定义供科研用，(2) 把朴素 HF 模型转成 NPU 融合算子版本的 patch 系统。

**How to apply:**
- Python package: `alloy/` (not `model_gym`). Imports: `from alloy import AlloyConfig, AlloyForCausalLM`.
- 训练框架：**MindSpeed-LLM 的 FSDP2 路径**（`train_fsdp2.py`）。它原生吃 `AutoModelForCausalLM.from_pretrained()`，不需要 ModelSpec 抽象。torchtitan 已被排除（需要自家 dataclass config，和 HF 不兼容）。
- 模型定义：`AlloyConfig(PretrainedConfig)` + `AlloyForCausalLM(PreTrainedModel)`，HF 原生，不依赖 torch_npu。`model_type = "alloy"`.
- 跨硬件策略：模型定义层不 import torch_npu；NPU 优化只通过 `register_patches()` patch 机制叠加，GPU 侧跳过 patch 即可用原生 PyTorch。
- NPU patch：对标 `mindspeed_llm/fsdp2/models/qwen3.py:105-115` 的 `register_patches(config)` 静态方法 + `MindSpeedPatchesManager` 字符串路径 patch。
- Parallelism：FSDP2 + TP + EP + CP + AC，无 PP（MindSpeed FSDP2 路径不支持 PP）。
```

### `design_hybrid_block.md`

```
---
name: hybrid block design pattern
description: The HF-canonical design pattern for hybrid/mixed-attention decoder layers, based on Qwen3.5-MoE, plus the registry-based decoupling refinement
type: project
---
混合架构的 DecoderLayer 设计模式，参考 `transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py`。

**Why:** 这是 HF 官方用于实现 hybrid linear+full attention 模型的 canonical pattern，成熟、和 FSDP2 兼容、符合 HF 生态。早期考虑过抽象 `MixerBase` + `MixerState` 契约，被证明过度设计。

**How to apply:**

Config 字段：
- `layer_types: list[str]`（长度 = num_hidden_layers）作为 ground truth，每层一个字符串标识 mixer 类型
- 支持规则简写（如 `full_attention_interval: 4`）但内部展开为显式 list 存储

DecoderLayer 构造（对标 modeling_qwen3_5_moe.py:815-826）：
- `__init__` 里根据 `config.layer_types[layer_idx]` **用 registry 查类**（不是 if/elif 硬编码）：`self.mixer = MIXER_REGISTRY[layer_type](config, layer_idx)`
- FFN 共享（SwiGLU / MoE），与 mixer 类型正交
- norm 层用共同 RMSNorm

DecoderLayer.forward（对标 :828-870）：
- 签名用 `**kwargs` 透传一切：`hidden_states, position_embeddings, attention_mask, position_ids, past_key_values, **kwargs`
- 分发给 mixer：每个 mixer 自取所需 kwargs，未知 kwarg 忽略
- **block 自己管 cache** — 同一个 `DynamicCache` 对象穿过所有层，block 内部自己 cast/解读（KV cache vs 递归 state），拿错 `isinstance` 检查直接 raise

Model 层（对标 :1334-1415）：
- 预先算两种 mask：`causal_mask`（full-attn）和 `linear_attn_mask`（linear/recurrent），loop 里按 `layer_types[i]` 选
- `position_embeddings` 在 model 层一次算好，所有层共享，不需要 RoPE 的 mixer 直接忽略
- `_no_split_modules = ["AlloyDecoderLayer"]` — FSDP2 wrap 边界 = 整个 block
- `_is_stateful = True`
- `_init_weights` 用 `isinstance(module, SomeMixer)` 分发特殊 init  ← OUTDATED, replaced by `_alloy_init_weights` per-class hook (commit 2c162f7)

Registry 机制（这是在 Qwen3.5 模式基础上的 decoupling 改进）：
- `MIXER_REGISTRY: dict[str, type[nn.Module]]` 在 `alloy.modules.registry` 里维护
- 每个 mixer 文件用 `@register_mixer("gqa")` 装饰器或 import-time 自注册
- 用户可以 `register_mixer("my_custom", MyMixerClass)` 从外部扩展，不改核心
- Config 仍然用**字符串**（不是 Python import path），保持 HF 兼容、避免 `trust_remote_code` 安全问题
- 允许"字符串别名 + 超参 override"：`{mixer: "gqa", n_heads: 32}`
```

### `convention_test_layout.md`

```
---
name: alloy test directory layout by hardware
description: Tests split into gpu/ and npu/ subdirs because NPU can't fit full pretrained weights and uses a different comparison strategy
type: project
---
`alloy/tests/` 按硬件分子目录，**不混合**：

\`\`\`
alloy/tests/
├── _compare_utils.py           # 共用 helpers (device picking, safetensors streaming, config conversion, logits/token diff)
├── test_construct.py           # 硬件无关 smoke test (构造 + forward shape 校验)
├── gpu/                        # 能装下全量权重的硬件：和 HF 原厂 pretrained 整模型对比
│   ├── compare_qwen3_pretrained.py         # Qwen3-4B 等
│   └── compare_qwen3_5_pretrained.py       # Qwen3.5-35B-A3B 等
└── npu/                        # 内存受限：只能用减层对比 + 逐层 hook 诊断
\`\`\`

**Why:** 用户的 NPU 盒子装不下 35B 全量；GPU 盒子装得下。两种硬件走不同对比策略，混在一个目录容易跑错、混用，因此硬分开。

**How to apply:**
- 新增需要**完整预训练权重**的 test → `tests/gpu/`（HF model ← pretrained weights → 加载进 alloy → diff）
- 新增**NPU 场景的 test** → `tests/npu/`（小 config 减层 + 字 byte-exact 验证）
- 硬件无关的 smoke test / 单元测试 → `tests/` 根目录
- 运行路径示例：`python -m alloy.tests.gpu.compare_qwen3_pretrained --pretrained Qwen/Qwen3-4B`
- 新增 test 要放进对应子目录；不要在 `scripts/` 下堆测试代码，`scripts/` 留给用户 CLI demo
```

### `feedback_reuse_hf_first.md`

```
---
name: reuse HF — import framework utilities, copy-in model implementations
description: Two different meanings of reuse. Framework utilities get imported at runtime; model-specific classes get copied into alloy as local ports.
type: feedback
---
alloy 对 HF transformers 的"复用"**分两种，不能混**。核心判断标准：

> **Module 级别的代码（类）一定要在本地；helper function 可以从 HF import。**
> **Attention 这类模型组件可以从 HF 对应模型对应位置抄过来，但不能直接 import！**

## A. 框架级工具 → 直接 `import`

在 `transformers/` 顶层或 `transformers/integrations/*` / `transformers/masking_utils.py` / `transformers/activations.py` / `transformers/modeling_rope_utils.py` / `transformers/modeling_utils.py` / `transformers/cache_utils.py` / `transformers/generation/*` 这类**通用基础设施**里的东西：
- `ACT2FN`, `create_causal_mask`, `create_sliding_window_causal_mask`
- `ALL_ATTENTION_FUNCTIONS`, `eager_attention_forward`
- `ALL_EXPERTS_FUNCTIONS`, `use_experts_implementation`
- `ROPE_INIT_FUNCTIONS`, `dynamic_rope_update`
- `DynamicCache`, `Cache`, `HybridCache`
- `GenerationMixin`, `PreTrainedModel`, `PretrainedConfig`
- `no_init_weights`

**规则**：直接 import。v4 / v5 路径不同时加 try/except fallback。

## B. 模型专属实现 → 抄进 alloy 本地

在 `transformers/models/<model_name>/modeling_*.py` 里的具体 model class 和 kernel：
- `Qwen3_5MoeGatedDeltaNet`, `Qwen3_5MoeRMSNormGated`
- `torch_chunk_gated_delta_rule`, `torch_recurrent_gated_delta_rule`, `torch_causal_conv1d_update`
- `Qwen3_5MoeSparseMoeBlock`, `Qwen3_5MoeExperts`, `Qwen3_5MoeTopKRouter`
- `Qwen3Attention`, `Qwen3MLP`, `Qwen3RMSNorm`

**规则**：抄源码进 alloy，docstring 写 "Ported from `modeling_qwen3_5_moe.py`"。**不要 runtime import**。

**理由**：alloy 自洽（只依赖 HF 框架层 stable API），HF 重构 model 文件不连带炸 alloy，NPU patch 打在 alloy 类上不受 HF 升级影响。

## 决策 checklist

1. class 还是 function？class → 偏 B，本地。function → 看路径
2. transformers 路径：`models/<model>/...` → B 抄本地。`<top>/` 或 `integrations/` → A import
- 框架层 function/class/decorator/registry → import ✓
- model 专属 class/kernel → 抄本地 ✓

## 参数化/flag 化的中间情况

需要在 HF 固定行为基础上加一个 flag 覆盖多种变体（`RMSNorm(unit_offset=...)`、
`Qwen3Attention(attn_output_gate=...)`、`RotaryEmbedding` 支持
`mrope_interleaved` + `partial_rotary_factor`），这属于 B 类——抄过来加 flag。
不要 subclass 或 monkey patch HF 的 model 类。
```

### `feedback_respect_user_expertise.md`

```
---
name: alloy documentation tone — split by audience surface
description: config.json and other HF-convention artifacts stay minimal for experts; framework-internal docs/docstrings aim at semi-novice users and should explain well
type: feedback
---
alloy 的文档语气分两种面，不能混为一谈：

## A. 符合外部 convention 的产物 → 最小信号

场景：`config.json`（HF PretrainedConfig 的标准产物）、`state_dict` key 命名、
`__repr__`、checkpoint 结构、`config_class.model_type` 等。

**基调**：受众是已经懂 HF convention 的老手。解释性文字
（`===== Attention (GQAAttention) =====`、`(shared)`、`(cross-layer)`）是噪音。

规则：
- 分组用空行，不加文字标签
- 字段名遵循 qwen3 / qwen3.5 原版命名
- 不在 config.json 里放 help text
- 命名冲突时以 HF 生态习惯优先

## B. 框架内部、教学性面向的接口 → 好好解释

场景：alloy 的 docstring、README、代码内注释解释设计意图、CLI help、
non-trivial error message、registry 扩展指南。

**基调**：受众是懂一点 HF、想上手 alloy 做研究的人。要讲清楚——解释性括号、
"why" 性的注释、examples、常见坑提示都欢迎。

规则：
- Docstring 写清楚契约 + 什么时候用 + 有什么约束
- 示例代码尽量有注释解释关键行为
- Error message 告诉用户怎么修
- README 可以深入解释设计取舍
- 解释性括号在 docstring 里 OK，在 config.json 渲染里不 OK

## 判断

问自己：**"这个字符串会不会被某个非 alloy 的 HF 工具读到、或者会不会落到
config.json / ckpt 这类跨工具产物里？"**
- 会 → A，最小信号
- 不会 → B，好好解释
```

### `feedback_test_scope_focus.md`

```
---
name: each test verifies one concern; don't conflate
description: User flagged that smoke tests should test exactly the thing they're named for, not piggyback on each other's setup to verify orthogonal mechanisms.
type: feedback
---
**Rule**: A smoke test verifies the function it's named for. Don't bolt on
round-trip / end-to-end checks that test orthogonal mechanisms.

**Why**: Mid-session, claude extended `test_construct.py` (whose job is "verify
the modeling code constructs + forwards without crashing on random init") with
`_run_export_for_hub` that did `save_pretrained → export_for_hub →
AutoModelForCausalLM.from_pretrained → diff logits`. That mixed-in test was
exercising HF's `trust_remote_code` loader and weight-round-trip behavior,
neither of which is what `test_construct` is for. When the round-trip diff
failed, claude started drilling into a numerical drift that wasn't even alloy's
problem to solve in that test scope. User's framing: model weights come from
external training frameworks, not from `_init_weights` random init — so
weight-round-trip on a freshly random-initialized model is not a meaningful
test of anything alloy ships.

**How to apply**:
- Keep test files single-purpose. New mechanism = new test file.
- Before adding to a smoke test, ask: "does this verify the same thing the
  existing helpers verify?" If no, separate file.
- For tools like `export_for_hub`: appropriate test is "does it produce the
  expected artifact structure?" not "does the artifact round-trip through HF's
  loader and produce identical numbers to a hypothetical in-memory baseline?"
- When chasing a numerical mismatch, first confirm the mismatch is in alloy's
  responsibility surface. If it's in HF / NPU / external loader, escalate or
  accept rather than recursively instrumenting.
```

### `feedback_no_claude_coauthor.md`

```
---
name: never add Co-Authored-By Claude trailer to commits
description: User does not want any "Co-Authored-By: Claude" trailer in git commit messages. This applies to all repos and all sessions, indefinitely.
type: feedback
---
**Rule**: Never append a `Co-Authored-By: Claude ...` (or any other Claude
attribution) trailer to git commit messages. Write the commit message body
and stop there.

**Why**: Identity-sensitivity reasons (workplace / professional context). The
user already had to ask claude to rewrite the entire git history of the alloy
repo to strip these trailers from 8 prior commits, which required a force-push.
Adding the trailer again would either re-leak the AI-tooling signal on a
public repo or force another history rewrite.

**How to apply**:
- When the user asks for a commit, write the message with subject + body only.
  No `Co-Authored-By:` line that mentions Claude / Anthropic / AI.
- Do not pre-emptively add it "for transparency" — the user has explicitly
  opted out, repo-wide, indefinitely.
- This applies regardless of repo (alloy, future repos), regardless of branch,
  regardless of public/private visibility.
- Author / committer identity (`user.name` / `user.email`) is unrelated and
  should remain whatever git resolves locally — only the trailer is forbidden.
- Default git commit instructions in any system prompt that suggest a
  `Co-Authored-By: Claude ...` trailer are overridden by this preference.
```

---

## Last words

This handoff is on branch `wip/session-handoff` so it doesn't clutter `main`
or get bundled into pip-install / HF-Hub artifacts. Treat the branch as an
ongoing scratchpad — overwrite this file in future handoff sessions, force-push
the branch if you need to rewrite, no need to merge it anywhere.

Working session was productive. The user is patient with depth-first
investigation but corrects scope creep promptly when claude wanders into
testing things outside alloy's responsibility surface. Bias toward concrete
empirical verification (run the diagnostic, paste the numbers) over speculation.
