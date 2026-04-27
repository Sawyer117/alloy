"""End-to-end demo of the config-only HF Hub publishing flow.

What it does
------------
1. Loads an ``AlloyConfig`` from a JSON file (or builds one in Python — edit
   the ``--config`` argument or this script's body).
2. Materializes a HF-Hub-loadable directory at ``--target``: writes
   ``config.json`` + a one-line ``modeling_alloy.py`` shim + injects
   ``auto_map``.
3. Verifies the directory by re-loading via
   ``AutoConfig.from_pretrained(target, trust_remote_code=True)`` and asserting
   the round-trip preserves layer / FFN types.
4. (Optional) ``--build-model`` instantiates ``AlloyForCausalLM`` from the
   loaded config via ``AutoModelForCausalLM.from_config``. Costs HBM equal to
   one fp32 random-init copy of the model — for big configs (35B-A3B) skip
   this step, the config round-trip is the actual deliverable.

What's *not* in this script
---------------------------
Weight loading. This is intentional — the workflow demonstrated here is
"publish a config so someone with weights can load them later". For
publishing weights too, do ``model.save_pretrained(target)`` *before*
calling ``export_for_hub`` (steps in the README's HF-Hub section).

Consumer side
-------------
After uploading the target dir to HF Hub, any consumer with
``pip install alloy`` can do::

    from transformers import AutoConfig, AutoModelForCausalLM
    cfg = AutoConfig.from_pretrained("user/my-alloy-model", trust_remote_code=True)
    model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True)
    # ...load weights from somewhere, then forward / generate...

Usage
-----
::

    python -m alloy.examples.package_config_for_hub \\
        --config alloy/examples/configs/qwen3_5_35b_a3b.json \\
        --target /tmp/my_alloy_repo

Add ``--build-model`` to also try instantiating from the round-tripped
config. Skip it for big-config sanity checks.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from alloy import AlloyConfig
from alloy.tools.export_for_hub import export_for_hub


def _load_config(path: Path) -> AlloyConfig:
    with open(path, encoding="utf-8") as f:
        cfg_dict = json.load(f)
    # Pop fields HF will set itself / fields meant for serialization metadata.
    cfg_dict.pop("model_type", None)
    cfg_dict.pop("transformers_version", None)
    return AlloyConfig(**cfg_dict)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True,
                        help="Path to a JSON config file (e.g. "
                             "alloy/examples/configs/qwen3_5_35b_a3b.json).")
    parser.add_argument("--target", required=True,
                        help="Output directory. Will be created if it doesn't exist.")
    parser.add_argument("--build-model", action="store_true",
                        help="Also instantiate AlloyForCausalLM via "
                             "AutoModelForCausalLM.from_config to verify the round-trip "
                             "produces a usable model class. Costs HBM = one fp32 copy.")
    args = parser.parse_args()

    target = Path(args.target).resolve()
    target.mkdir(parents=True, exist_ok=True)

    # ----- 1. Build AlloyConfig --------------------------------------- #
    config = _load_config(Path(args.config))
    print(f"[1/3] Built AlloyConfig from {args.config}")
    print(f"      num_hidden_layers={config.num_hidden_layers}  "
          f"hidden_size={config.hidden_size}  vocab_size={config.vocab_size}")
    print(f"      layer_types[:4]={config.layer_types[:4]}  "
          f"ffn_types[:4]={config.ffn_types[:4]}")

    # ----- 2. Save + export_for_hub ----------------------------------- #
    config.save_pretrained(str(target))
    export_for_hub(target)
    files = sorted(p.name for p in target.iterdir())
    print(f"[2/3] Materialized HF-Hub-loadable repo at {target}")
    print(f"      Files: {files}")

    # ----- 3. Round-trip verification --------------------------------- #
    from transformers import AutoConfig

    loaded = AutoConfig.from_pretrained(str(target), trust_remote_code=True)
    if type(loaded).__name__ != "AlloyConfig":
        raise AssertionError(
            f"Expected AlloyConfig from AutoConfig.from_pretrained, got {type(loaded).__name__}. "
            f"auto_map injection probably failed."
        )
    if loaded.layer_types != config.layer_types:
        raise AssertionError("layer_types did not survive round-trip")
    if loaded.ffn_types != config.ffn_types:
        raise AssertionError("ffn_types did not survive round-trip")

    print(f"[3/3] Round-trip OK: AutoConfig.from_pretrained returned {type(loaded).__name__}, "
          f"layer_types/ffn_types preserved.")

    if args.build_model:
        from transformers import AutoModelForCausalLM

        print(f"[opt] Instantiating model from round-tripped config "
              f"(may take a while, allocates one fp32 copy)...")
        model = AutoModelForCausalLM.from_config(loaded, trust_remote_code=True)
        nparam = sum(p.numel() for p in model.parameters()) / 1e9
        print(f"      Built {type(model).__name__}: {nparam:.2f}B params  "
              f"(random-init, no weights loaded)")

    print()
    print(f"Ready to upload {target} to HF Hub. Consumer command:")
    print(f"    pip install alloy")
    print(f"    AutoConfig.from_pretrained('<repo>', trust_remote_code=True)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
