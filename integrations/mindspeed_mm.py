"""Plugin shim — register :class:`AlloyForCausalLM` into MindSpeed-MM.

Importing this module (via the trainer yaml's ``training.plugin`` list) does
two things:

  1. Decorates :class:`AlloyForCausalLM` with
     ``@model_register.register("alloy")`` so ``model.model_id: alloy`` in
     yaml resolves to the alloy model class via mindspeed's ``ModelHub``.

  2. Defines :meth:`AlloyForCausalLMPlugin.overwrite_transformer_config` —
     mindspeed's pre-construction hook for plumbing yaml ``model_args`` onto
     the HF ``transformer_config`` object before the model is built. We
     forward every yaml field whose name matches the alloy runtime
     dispatch convention ``_<module_key>_implementation`` (e.g.
     ``_qwen3_5_gdn_implementation``, ``_experts_implementation``) onto
     the config so :class:`AlloyForCausalLM` and its sub-modules pick the
     requested fast-path backends at construction.

The pattern lets new alloy modules with their own dispatch surfaces auto-plumb
without editing this shim — add a new yaml field, ship a new alloy module,
done.

Usage in a mindspeed-mm trainer yaml:

.. code-block:: yaml

    model:
      model_id: alloy
      model_name_or_path: ./hf_models/your_alloy_dir   # produced by tools/export_for_hub
      trust_remote_code: true
      attn_implementation: sdpa                        # standard HF flag (no underscore)
      _qwen3_5_gdn_implementation: triton              # alloy-native runtime field
      _experts_implementation: flash                   # alloy-native runtime field

    training:
      plugin:
        - alloy.integrations.mindspeed_mm              # ← triggers this module's registration
        - alloy.integrations.hf_npu_binder             # ← optional: register binder fast paths
"""
from __future__ import annotations

from mindspeed_mm.fsdp.utils.register import model_register

from alloy import AlloyForCausalLM


@model_register.register("alloy")
class AlloyForCausalLMPlugin(AlloyForCausalLM):
    """Thin subclass: pure inheritance + the mindspeed-required static hook.

    No model behaviour is changed — every forward / backward / state_dict
    interaction is identical to :class:`AlloyForCausalLM`. The subclass
    exists only to attach :meth:`overwrite_transformer_config` and to be
    the symbol that the registry decorator owns.
    """

    @staticmethod
    def overwrite_transformer_config(transformer_config, model_args):
        """Forward alloy-native runtime fields from yaml onto the HF config.

        Recognises any attribute on ``model_args`` matching the pattern
        ``_<module_key>_implementation``. That covers alloy's full set of
        dispatch fields (``_qwen3_5_gdn_implementation``,
        ``_experts_implementation``, future ``_mamba3_ssm_implementation`` …)
        without requiring this shim to enumerate them — adding a new alloy
        module just needs the matching yaml field, no code edit here.

        Standard HF fields like ``attn_implementation`` (no leading
        underscore) are not handled here; mindspeed already forwards them
        to ``AutoConfig.from_pretrained(_attn_implementation=...)`` upstream.
        """
        for attr_name in vars(model_args):
            if not (attr_name.startswith("_") and attr_name.endswith("_implementation")):
                continue
            val = getattr(model_args, attr_name)
            if val is None:
                continue
            setattr(transformer_config, attr_name, val)
        return transformer_config


__all__ = ["AlloyForCausalLMPlugin"]
