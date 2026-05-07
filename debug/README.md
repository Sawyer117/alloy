# debug/ — throwaway diagnostics

Short-lived helpers for tracking down a specific issue. Delete the whole
directory once the issue is resolved (don't leave stale debug code in
the tree).

## Current investigation

`compare_dsv4_random.py` fails at `load_state_dict` with size mismatches
on `mlp.gate.weight` / `mlp.experts.gate_up_proj` etc. — HF builds the
MoE block with 256 experts but alloy builds with 4. Hypothesis: the
`num_experts` kwarg passed to `DeepseekV4Config` doesn't reach the
field that DSV4 modeling actually reads (`num_local_experts`, aliased
to `n_routed_experts`).

## Run

```bash
cd /path/to/alloy
bash debug/run.sh
```

Paste the output back to confirm root cause; the fix to
`compare_dsv4_random.py:_build_configs()` lands afterward.

## Cleanup

Once the test is fixed:

```bash
rm -rf debug/
git add -A && git commit -m "Remove debug/ — DSV4 config kwarg mismatch fixed"
```
