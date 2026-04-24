"""Public helpers for instantiating alloy models and loading pretrained weights.

Two construction patterns, two distinct scenarios (pick by intent, not by
hardware):

* **Loading an external checkpoint** (inference / fine-tuning / validation)
  ``_init_weights`` is pure waste — load_state_dict overwrites every
  parameter anyway. Use :func:`build_skeleton` to skip init and allocate
  directly in the target dtype. Saves tens of seconds on multi-billion-param
  models (CPU ``nn.init.normal_`` over every Linear is single-threaded and
  surprisingly expensive) and halves peak CPU memory.

* **Training from scratch** (from-scratch pretrain / fresh architecture
  ablation) — init values must be correct. Use :func:`build_on_device` to
  run the same ``_init_weights`` hook but on the target accelerator, where
  the ``randn`` / ``uniform_`` ops execute in parallel on GPU/NPU instead
  of single-threaded CPU.

Plus :func:`load_state_dict_from_disk` for streaming safetensors shards
directly off disk without instantiating a second HF model.
"""
from __future__ import annotations

import contextlib
import gc
import json
import re
from pathlib import Path
from typing import Callable, Iterable

import torch


# --------------------------------------------------------------------------- #
# Construction helpers
# --------------------------------------------------------------------------- #


@contextlib.contextmanager
def build_skeleton(dtype: torch.dtype):
    """Construction context for the checkpoint-loading case.

    Allocates the module tree in ``dtype`` with uninitialized tensor storage,
    skipping HF's ``_init_weights`` hook entirely. The caller is expected to
    run ``load_state_dict`` immediately afterwards; any uninitialized values
    get overwritten.

    Important: HF's ``no_init_weights`` also short-circuits ``tie_weights()``.
    After construction and **before** ``load_state_dict``, call
    ``model.tie_weights()`` if the config has ``tie_word_embeddings=True``
    (e.g. Qwen3-4B). Otherwise ``lm_head.weight`` will be an independent
    uninitialized Parameter and ``load_state_dict`` will report it missing
    since the ckpt only stores ``model.embed_tokens.weight``.

    Example::

        with build_skeleton(torch.bfloat16):
            model = AlloyForCausalLM(config)
        model.tie_weights()                                    # alias lm_head
        model.load_state_dict(state_dict, strict=False)
        model.to("cuda").eval()
    """
    try:
        from transformers.initialization import no_init_weights  # transformers v5
    except ImportError:
        from transformers.modeling_utils import no_init_weights  # transformers v4

    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        with no_init_weights():
            yield
    finally:
        torch.set_default_dtype(old_dtype)


@contextlib.contextmanager
def build_on_device(dtype: torch.dtype, device: str | torch.device):
    """Construction context for the from-scratch training case.

    Keeps ``_init_weights`` enabled (so init is correct — dt_bias ones, A_log
    uniform-log, Experts and router normal with ``initializer_range``,
    unit-offset RMSNorm zeros, etc.) but runs all allocation and init ops on
    ``device``. Orders-of-magnitude faster than the default CPU init path for
    multi-billion-param models.

    Example::

        with build_on_device(torch.bfloat16, "cuda"):
            model = AlloyForCausalLM(config)
    """
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        with torch.device(device):
            yield
    finally:
        torch.set_default_dtype(old_dtype)


# --------------------------------------------------------------------------- #
# State-dict streaming
# --------------------------------------------------------------------------- #


def load_state_dict_from_disk(
    model_path: str | Path,
    ignore_patterns: Iterable[str] = (),
    device: str | torch.device = "cpu",
    key_remap: Callable[[str], str | None] | None = None,
) -> dict[str, torch.Tensor]:
    """Stream a full ``state_dict`` from ``safetensors`` shards.

    Handles both single-file (``model.safetensors``) and sharded layouts
    (``model-NNNNN-of-NNNNN.safetensors`` + index). Memory-maps shards so
    unused tensors aren't materialized in RAM all at once.

    Parameters
    ----------
    model_path:
        Directory containing ``config.json`` + one or more ``.safetensors``
        shards.
    ignore_patterns:
        Iterable of regex patterns. Any checkpoint key matching one of these
        is skipped. Useful to drop vision / mtp heads or rotary buffers when
        loading a multimodal checkpoint into a text-only model.
    device:
        Where to place each tensor as it's loaded. Default "cpu"; set to
        "cuda" / "npu" to avoid an extra copy if you're going straight to
        the accelerator.
    key_remap:
        Optional ``(old_key) -> new_key | None`` callable. Applied to every
        non-ignored key. Returning ``None`` skips the key. Useful when the
        checkpoint layout doesn't match the target module tree
        (e.g. stripping a ``model.language_model.`` prefix from a
        ``...ForConditionalGeneration`` checkpoint when loading into a
        text-only ``...ForCausalLM``). HF's own ``from_pretrained`` handles
        this via ``base_model_prefix``; raw safetensors reads need it done
        explicitly.
    """
    from safetensors.torch import load_file

    model_path = Path(model_path)
    index_file = model_path / "model.safetensors.index.json"

    if index_file.exists():
        with open(index_file, encoding="utf-8") as f:
            shards = sorted(set(json.load(f)["weight_map"].values()))
    else:
        single = model_path / "model.safetensors"
        if not single.exists():
            raise FileNotFoundError(f"No safetensors index or single file in {model_path}")
        shards = [single.name]

    ignore_re = [re.compile(p) for p in ignore_patterns]
    device_str = str(device)

    full_sd: dict[str, torch.Tensor] = {}
    for shard in shards:
        shard_sd = load_file(str(model_path / shard), device=device_str)
        for k, v in shard_sd.items():
            if any(r.match(k) for r in ignore_re):
                continue
            if key_remap is not None:
                new_k = key_remap(k)
                if new_k is None:
                    continue
                k = new_k
            full_sd[k] = v
    return full_sd


# --------------------------------------------------------------------------- #
# Small utility — release accelerator memory between phases
# --------------------------------------------------------------------------- #


def empty_cache(device: torch.device | str = "cuda") -> None:
    """Release unused memory on the given accelerator, best-effort."""
    device = torch.device(device) if isinstance(device, str) else device
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "npu":
        try:
            torch.npu.empty_cache()
        except AttributeError:
            pass
    gc.collect()


def strip_language_model_prefix(key: str) -> str:
    """Strip the ``model.language_model.`` prefix HF uses for multimodal wrappers.

    Intended as a ``key_remap`` for :func:`load_state_dict_from_disk` when
    loading a ``...ForConditionalGeneration`` checkpoint (e.g. Qwen3.5-MoE,
    Qwen2.5-VL, Llama-4-MM, …) into a text-only ``...ForCausalLM`` — the
    multimodal checkpoint nests the text backbone under
    ``model.language_model.*`` to make room for ``model.visual.*`` /
    ``model.audio.*`` siblings, while the text-only class expects everything
    directly under ``model.*``. HF's ``from_pretrained`` auto-strips via
    ``base_model_prefix``; raw safetensors reads need it explicit.

    Safe to apply unconditionally — keys without the prefix pass through.

    Typical use::

        sd = load_state_dict_from_disk(
            ckpt_dir,
            ignore_patterns=[
                r"^model\\.visual\\.",        # vision tower, not in text model
                r"^model\\.mtp\\.",           # multi-token-prediction head
                r".*rotary_emb\\.inv_freq$",  # recomputed from config
            ],
            key_remap=strip_language_model_prefix,
        )
    """
    prefix = "model.language_model."
    if key.startswith(prefix):
        return "model." + key[len(prefix):]
    return key


__all__ = [
    "build_skeleton",
    "build_on_device",
    "load_state_dict_from_disk",
    "empty_cache",
    "strip_language_model_prefix",
]
