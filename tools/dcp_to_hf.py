"""Merge a mindspeed-mm DCP checkpoint into an HF-loadable directory.

After training alloy under mindspeed-mm FSDP2, the ``save`` directory contains
a torch DCP shard tree (``release/__*_*.distcp`` + ``.metadata``) plus a
``latest_checkpointed_iteration.txt``. This tool unpacks that tree, repackages
the weights as standard HF sharded safetensors, and combines them with the
``config.json`` / ``modeling_alloy.py`` / tokenizer files from the alloy hub
directory you originally trained from.

Output is loadable directly:

    AutoModelForCausalLM.from_pretrained(target, trust_remote_code=True)

Why ``alloy/tools/dcp_to_hf`` instead of mindspeed-mm's
``checkpoint/common/merge_dcp_to_hf.py``
-----------------------------------------------------------------------
The upstream script keys its safetensors sharding off a pre-existing
``*.safetensors.index.json`` in a "model assets dir" — useful when you started
training from real published HF weights, but a chicken-and-egg problem for
alloy where the hub dir produced by ``package_config_for_hub`` is
config-only (no weights, no index). It also bakes in a hard
``model.<prefix> + key`` lookup convention that doesn't fit alloy's
canonical state_dict naming.

This tool sidesteps both: shards are computed via
``huggingface_hub.split_torch_state_dict_into_shards`` from the loaded
state_dict alone, and key prefix handling is a clean opt-in (default empty
is correct for alloy + mindspeed-mm).

Usage
-----
::

    python -m alloy.tools.dcp_to_hf \\
        --dcp-dir   ./intermediate_ckpt/release \\
        --hub-dir   ./hf_models/alloy_qwen3_next_340m \\
        --target    ./hf_models/alloy_qwen3_next_340m_trained

Use ``--strip-prefix`` only if your trainer wrote keys under an extra
namespace (most setups don't); use ``--max-shard-size`` if you want
something other than 5GB shards.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch


# Files we copy alongside the weights into the target dir. Mirrors the set
# package_config_for_hub.py knows about, plus config.json + modeling shim.
# Files absent from --hub-dir are skipped silently.
_HUB_AUX_FILES: tuple[str, ...] = (
    "config.json",
    "modeling_alloy.py",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "tokenizer.model",
    "vocab.json",
    "merges.txt",
    "added_tokens.json",
    "chat_template.jinja",
    "generation_config.json",
)


def _load_dcp_state_dict(dcp_dir: Path) -> dict[str, torch.Tensor]:
    """Read a torch DCP shard directory into an in-memory state_dict.

    mindspeed-mm's checkpoint save wraps the whole state under
    ``{"model": ...}``; we unwrap so callers see canonical HF-style keys.
    Uses the same private ``torch.distributed.checkpoint`` APIs that
    mindspeed-mm's own merge_dcp_to_hf calls — kept matched on purpose so
    a torch upgrade affects both in lockstep.
    """
    from torch.distributed.checkpoint import FileSystemReader
    from torch.distributed.checkpoint.state_dict_loader import _load_state_dict
    from torch.distributed.checkpoint.default_planner import _EmptyStateDictLoadPlanner

    sd: dict = {}
    _load_state_dict(
        sd,
        storage_reader=FileSystemReader(str(dcp_dir)),
        planner=_EmptyStateDictLoadPlanner(),
        no_dist=True,
    )
    return sd["model"] if "model" in sd else sd


def _strip_prefix(state_dict: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    if not prefix:
        return state_dict
    return {(k[len(prefix):] if k.startswith(prefix) else k): v for k, v in state_dict.items()}


def _copy_aux_files(hub_dir: Path, target: Path) -> list[str]:
    copied: list[str] = []
    for name in _HUB_AUX_FILES:
        src = hub_dir / name
        if src.is_file():
            shutil.copy2(src, target / name)
            copied.append(name)
    return sorted(copied)


def dcp_to_hf(
    dcp_dir: str | Path,
    hub_dir: str | Path,
    target: str | Path,
    *,
    strip_prefix: str = "",
    max_shard_size: str = "5GB",
) -> Path:
    """Convert a DCP shard directory into an HF-loadable directory.

    Parameters
    ----------
    dcp_dir
        Path to the DCP shard directory. For mindspeed-mm checkpoints this
        is typically ``<your save dir>/release/`` (the ``release/`` subdir,
        not the parent that holds ``latest_checkpointed_iteration.txt``).
    hub_dir
        Path to the alloy HF hub directory produced by
        ``package_config_for_hub`` — provides config.json, modeling_alloy.py,
        and the tokenizer files.
    target
        Output directory. Created if missing.
    strip_prefix
        Leading prefix to peel off DCP keys before saving. Empty string is
        correct for alloy under mindspeed-mm.
    max_shard_size
        Threshold for sharding via ``split_torch_state_dict_into_shards``.

    Returns
    -------
    Path
        The resolved target directory.
    """
    from safetensors.torch import save_file
    from huggingface_hub import split_torch_state_dict_into_shards

    dcp_dir = Path(dcp_dir).resolve()
    hub_dir = Path(hub_dir).resolve()
    target = Path(target).resolve()

    if not dcp_dir.is_dir():
        raise NotADirectoryError(f"--dcp-dir does not exist or is not a directory: {dcp_dir}")
    if not hub_dir.is_dir():
        raise NotADirectoryError(f"--hub-dir does not exist or is not a directory: {hub_dir}")
    if not (hub_dir / "config.json").is_file():
        raise FileNotFoundError(
            f"--hub-dir lacks config.json — does it point at a package_config_for_hub output? "
            f"Looked in: {hub_dir}"
        )

    target.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] Loading DCP shards from {dcp_dir}")
    state_dict = _load_dcp_state_dict(dcp_dir)
    if not state_dict:
        raise RuntimeError(f"DCP load returned an empty state_dict from {dcp_dir}")
    total = sum(t.numel() for t in state_dict.values() if isinstance(t, torch.Tensor))
    sample_keys = list(state_dict.keys())[:3]
    print(f"      Loaded {len(state_dict)} tensors, ~{total / 1e6:.2f}M params")
    print(f"      Sample keys: {sample_keys}")

    if strip_prefix:
        n_before = sum(1 for k in state_dict if k.startswith(strip_prefix))
        state_dict = _strip_prefix(state_dict, strip_prefix)
        print(f"      Stripped prefix {strip_prefix!r} from {n_before} key(s)")

    print(f"[2/4] Sharding state_dict (max_shard_size={max_shard_size})")
    split = split_torch_state_dict_into_shards(
        state_dict,
        max_shard_size=max_shard_size,
        filename_pattern="model{suffix}.safetensors",
    )

    metadata = {"format": "pt"}
    items = sorted(split.filename_to_tensors.items())
    for i, (filename, tensor_keys) in enumerate(items, start=1):
        shard = {k: state_dict[k] for k in tensor_keys}
        save_file(shard, str(target / filename), metadata=metadata)
        n_params = sum(t.numel() for t in shard.values())
        print(f"      [{i}/{len(items)}] {filename}  ({n_params / 1e6:.2f}M params, {len(shard)} tensors)")

    if split.is_sharded:
        with open(target / "model.safetensors.index.json", "w", encoding="utf-8") as f:
            json.dump(
                {"metadata": split.metadata, "weight_map": split.tensor_to_filename},
                f, indent=2, sort_keys=True,
            )
        print(f"      Wrote model.safetensors.index.json")

    print(f"[3/4] Copying config / modeling / tokenizer from {hub_dir}")
    copied = _copy_aux_files(hub_dir, target)
    print(f"      Copied: {copied}")
    missing = [n for n in ("config.json", "modeling_alloy.py") if n not in copied]
    if missing:
        print(f"      WARNING — these expected files were missing in --hub-dir: {missing}")

    print(f"[4/4] Done. Target: {target}")
    print(f"      Verify with:")
    print(f"        from transformers import AutoModelForCausalLM")
    print(f"        m = AutoModelForCausalLM.from_pretrained('{target}', trust_remote_code=True)")
    return target


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--dcp-dir", required=True,
        help="Path to the DCP shard directory (typically '<save_dir>/release').",
    )
    parser.add_argument(
        "--hub-dir", required=True,
        help="Path to the alloy HF hub directory produced by package_config_for_hub. "
             "Provides config.json + modeling_alloy.py + tokenizer files.",
    )
    parser.add_argument(
        "--target", required=True,
        help="Output directory. Will be created if missing.",
    )
    parser.add_argument(
        "--strip-prefix", default="",
        help="Optional prefix to strip from DCP keys before saving. Default empty "
             "(correct for alloy under mindspeed-mm — alloy stores keys like "
             "'model.embed_tokens.weight' already, and the outer '{\"model\": ...}' "
             "wrapper is unwrapped at load time).",
    )
    parser.add_argument(
        "--max-shard-size", default="5GB",
        help="Threshold for sharding the safetensors output. Default '5GB'.",
    )
    args = parser.parse_args()

    dcp_to_hf(
        dcp_dir=args.dcp_dir,
        hub_dir=args.hub_dir,
        target=args.target,
        strip_prefix=args.strip_prefix,
        max_shard_size=args.max_shard_size,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
