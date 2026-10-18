"""Import each pipeline in isolation without loading weights or submitting GPU work."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PROBES = {
    "priors": ([REPO / "src", REPO / "vendor/mosca", REPO / "vendor/mosca/lib_prior",
                HERE / "runtime/cotracker3"],
               "import mosca_precompute; import tracking.bootstapir_wrapper; import cotracker.predictor"),
    "ms_reconstruction": ([REPO / "reproduce/table1/backends/ms"], "import mosca_reconstruct"),
    "wt_reconstruction": ([REPO / "reproduce/table1/backends/wt"], "import mosca_reconstruct"),
    "sam3": ([REPO / "vendor/mosca/lib_prior/sam3"], "import sam3.model_builder"),
    "generation": ([HERE / "runtime", REPO / "vendor/diffsynth"],
                   "from gen4dgs.inference.diffsynth_bridge import load_wan_train_module; "
                   f"load_wan_train_module(Path({str(REPO / 'vendor/diffsynth')!r}))"),
}


def main():
    results = {}
    with tempfile.TemporaryDirectory(prefix="wfm-import-check-") as cache:
        env = dict(os.environ, GS_BACKEND="native_add3", JAX_PLATFORMS="cpu", MPLCONFIGDIR=cache,
                   HF_HUB_OFFLINE="1", DIFFSYNTH_SKIP_DOWNLOAD="true", PYTHONDONTWRITEBYTECODE="1")
        for name, (paths, code) in PROBES.items():
            expanded = []
            for path in paths:
                expanded.extend([str(path), str(path / "lib_moca"), str(path / "lib_mosca")])
            command = f"import sys; from pathlib import Path; sys.path[:0] = {expanded!r}; {code}"
            result = subprocess.run([sys.executable, "-c", command], cwd=REPO, env=env,
                                    capture_output=True, text=True)
            results[name] = dict(ok=result.returncode == 0, returncode=result.returncode)
            if result.returncode:
                results[name]["diagnostic"] = (result.stderr + result.stdout)[-6000:]
            print(f"{name}: {'PASS' if result.returncode == 0 else 'FAIL'}", flush=True)
    print(json.dumps(results, indent=2))
    return 0 if all(r["ok"] for r in results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
