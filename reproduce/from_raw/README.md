# Independent reconstruction

Follow [REPRODUCTION.md](../../REPRODUCTION.md) for setup and the complete workflow.
`download.py` verifies pinned inputs; `run.py` executes nine isolated stages and
records failures and timings. Outputs must be fresh and separate from inputs.
`collect.py` accepts only complete two-method, seven-scene campaigns.

The optional `run.sbatch` wrapper accepts `WFM_REPO_ROOT`, `WFM_RAW_INPUTS`,
`WFM_RAW_OUTPUT` and `WFM_PYTHON`. Array indices 6 and 13 select the MS/WT wheel
pilots; 0–5 and 7–12 select the remaining scenes. Set account, partition, memory
and wall time for your scheduler. The default two-task concurrency limit applies
only to that array. Local commands in the main guide do not require Slurm.
