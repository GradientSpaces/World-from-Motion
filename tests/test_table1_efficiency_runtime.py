"""Evaluation must ignore fitting switches even when exported by the caller."""
import json
import os
import sys
from types import SimpleNamespace

import pytest

from reproduce.table1 import run, worker
from reproduce.table1.efficiency import runtime


@pytest.mark.parametrize("action,preset", (("fit", None), ("fit", "reference"),
                                         ("fit", "efficient"),
                                         ("evaluate", None), ("metrics", None)))
@pytest.mark.parametrize("method", ("ms", "wt"))
def test_worker_isolates_evaluation_before_backend_import(tmp_path, monkeypatch, action, preset, method):
    requested = {key: "cached" if key.endswith("SSIM") else "1" for key in runtime.VARIABLES}
    if action == "fit":
        requested["WFM_TABLE1_GRAPH_LPIPS"] = "0"
    reference = {key: "reference" if key.endswith("SSIM") else "0" for key in runtime.VARIABLES}
    expected = dict(requested if action == "fit" and preset != "reference" else reference)
    if action == "fit" and preset == "efficient" and method == "ms":
        expected["WFM_TABLE1_FUSED_GAUSSIAN"] = "0"
    for key, value in requested.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setitem(sys.modules, "run", run)
    monkeypatch.setitem(sys.modules, "efficiency.runtime", runtime)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True)))
    output = tmp_path / "output"
    monkeypatch.setattr(sys, "argv", ["worker.py", "--action", action, "--method", method,
                                     "--scene", "wheel", "--mode", "fit" if action == "fit" else "static",
                                     "--workspace", str(tmp_path / "inputs"),
                                     "--metric-workspace", str(tmp_path / "inputs"),
                                     "--checkpoint", str(tmp_path / "checkpoint"),
                                     "--predictions", str(tmp_path / "predictions"), "--output", str(output)])
    if preset:
        sys.argv.extend(["--fit-preset", preset])
    seen = []

    def backend(backend_method):
        # Several renderer flags are captured at import time. Resetting them
        # only inside evaluate(), after backend configuration, is too late.
        # Both methods use the shared MoSca scorer for static predictions.
        assert backend_method == ("ms" if action == "metrics" else method)
        assert {key: os.environ[key] for key in requested} == expected
        seen.append("backend")
        return tmp_path / backend_method

    def perform(*args, **kwargs):
        assert seen == ["backend"]
        seen.append(action)

    monkeypatch.setattr(worker, "configure_backend", backend)
    monkeypatch.setattr(worker, "fit", perform)
    monkeypatch.setattr(worker, "evaluate", perform)
    monkeypatch.setattr(worker, "score", perform)
    worker.main()
    assert seen == ["backend", action]
    if action != "fit":
        assert json.loads((output / "efficiency_settings.json").read_text()) == reference


def test_public_entry_point_rejects_fitting_preset_on_evaluation(capsys):
    with pytest.raises(SystemExit) as error:
        run.main(["evaluate", "--fit-preset", "reference"])
    assert error.value.code == 2
    assert "only valid for fit" in capsys.readouterr().err


def test_default_fitting_preset_and_environment_overrides(monkeypatch):
    for key in runtime.VARIABLES:
        monkeypatch.delenv(key, raising=False)
    assert runtime.configure_fit_environment() == runtime.FIT_PRESETS["efficient"]
    monkeypatch.setenv("WFM_TABLE1_BATCH_LPIPS", "0")
    resolved = runtime.configure_fit_environment()
    assert resolved["WFM_TABLE1_BATCH_LPIPS"] == "0"
    assert resolved["WFM_TABLE1_FUSED_GAUSSIAN"] == "1"
    assert runtime.configure_fit_environment("efficient") == runtime.FIT_PRESETS["efficient"]
    assert runtime.configure_fit_environment("reference") == runtime.FIT_PRESETS["reference"]


def test_invalid_fitting_switch_is_rejected_without_partial_environment_update(monkeypatch):
    for key in runtime.VARIABLES:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("WFM_TABLE1_BATCH_LPIPS", "typo")
    with pytest.raises(ValueError, match="WFM_TABLE1_BATCH_LPIPS must be 0 or 1"):
        runtime.configure_fit_environment()
    assert "WFM_TABLE1_MATRIX_MATH" not in os.environ


@pytest.mark.parametrize("method,fusion", [("ms", "0"), ("wt", "1")])
def test_method_defaults_preserve_explicit_experiment_overrides(monkeypatch, method, fusion):
    for key in runtime.VARIABLES:
        monkeypatch.delenv(key, raising=False)
    expected = dict(runtime.FIT_PRESETS["efficient"], WFM_TABLE1_FUSED_GAUSSIAN=fusion)
    assert runtime.configure_fit_environment(method=method) == expected
    # Benchmark plans deliberately select every switch, independent of defaults.
    monkeypatch.setenv("WFM_TABLE1_FUSED_GAUSSIAN", "1" if fusion == "0" else "0")
    assert runtime.configure_fit_environment(method=method)["WFM_TABLE1_FUSED_GAUSSIAN"] != fusion
    assert runtime.configure_fit_environment("efficient", method=method) == expected
    assert runtime.configure_evaluation() == runtime.FIT_PRESETS["reference"]


@pytest.mark.parametrize("method,batch,error", [
    ("ms", "0", None),
    ("ms", "1", "cannot be enabled together"),
    ("wt", "0", "only for MoSca"),
])
def test_graph_lpips_opt_in_rejects_incompatible_workloads(monkeypatch, method, batch, error):
    for key in runtime.VARIABLES:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("WFM_TABLE1_GRAPH_LPIPS", "1")
    monkeypatch.setenv("WFM_TABLE1_BATCH_LPIPS", batch)
    if error:
        with pytest.raises(ValueError, match=error):
            runtime.configure_fit_environment(method=method)
        assert "WFM_TABLE1_MATRIX_MATH" not in os.environ
    else:
        settings = runtime.configure_fit_environment(method=method)
        assert settings["WFM_TABLE1_GRAPH_LPIPS"] == "1"
        assert settings["WFM_TABLE1_BATCH_LPIPS"] == "0"
    # Explicit reference/evaluation must ignore even conflicting inherited flags.
    assert runtime.configure_evaluation() == runtime.FIT_PRESETS["reference"]
    assert runtime.configure_fit_environment("efficient", method="wt")["WFM_TABLE1_GRAPH_LPIPS"] == "0"
