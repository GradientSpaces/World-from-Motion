# Distillation and evaluation

Follow [REPRODUCTION.md](../../REPRODUCTION.md) and [ENVIRONMENT.md](ENVIRONMENT.md).
`run.py` fits, evaluates and aggregates explicit run manifests; `compare.py`
reports signed per-scene and mean differences. The raw runner supplies manifests
automatically. `runs.example.json` and `retained_runs.json` describe portable
layouts only; the referenced scene assets are not shipped.

`reference.json` retains the older archive's values and checkpoint fingerprints.
`public-pool-reference.json` contains the September 14, 2026 comparison values.
They are separate from the paper's rounded values. `aggregate --reference`
recalculates saved values and does not measure a new model.
