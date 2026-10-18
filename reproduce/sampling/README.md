# Direct video sampling

Follow [REPRODUCTION.md](../../REPRODUCTION.md) for the complete workflow.
The entry point is `src/wfm/dycheck_sampling.py` with stages `align`, `render`,
`generate`, and `score`. It uses the source run's initial monocular reconstruction.
`collect.py` checks stage success, recipe consistency and held-out frame identities;
`preview.py` exports display videos from a completed run.
