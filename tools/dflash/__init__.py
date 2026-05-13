"""DFlash analysis utilities — workload sweeps + speculative-decoding math.

Operational scripts that consume the roofline package to answer questions
about DSV4 + DFlash draft model combinations: where the latency budget
goes, when memory bandwidth dominates vs compute, what context length a
draft can keep up with at a given hardware spec, etc.

Examples / individual model configs live under ``examples/roofline/``;
this package is the cross-model batch-sweep companion. Putting it under
``tools/`` rather than ``examples/`` keeps the per-model example files
single-purpose (one model, one roofline run) while letting these
multi-model analyses live separately.
"""
