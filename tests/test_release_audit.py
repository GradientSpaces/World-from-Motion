from __future__ import annotations

from pathlib import Path

from scripts.audit_source import audit, load_denylist


def test_release_tree_contains_no_weights_credentials_or_lustre_paths() -> None:
    root = Path(__file__).resolve().parents[1]
    result = audit(root, tracked_only=(root / ".git").exists())
    assert result["ok"], result["failures"]


def test_audit_checks_slurm_scripts_and_external_symlinks(tmp_path: Path) -> None:
    script = tmp_path / "run.sbatch"
    script.write_text("MODEL=" + "/" + "lustre/private/model.pt\n")
    external = tmp_path.parent / (tmp_path.name + "-outside.txt")
    external.write_text("external input")
    try:
        (tmp_path / "input.txt").symlink_to(external)
        result = audit(tmp_path)
        assert not result["ok"]
        assert any("hardcoded Lustre path: run.sbatch" in s for s in result["failures"])
        assert any("symlink leaves source tree: input.txt" in s for s in result["failures"])
    finally:
        external.unlink()


def test_audit_flags_scheduler_host_and_storage_records(tmp_path: Path) -> None:
    # Fixture strings are assembled so this file does not trip the audit itself.
    (tmp_path / "job.sbatch").write_text("#!/bin/bash\n#SBATCH --" + "account=team_a\n")
    (tmp_path / "launch.sbatch").write_text('srun --container-mounts="${WFM_CONTAINER_MOUNTS:-' + '/data:/data}"\n')
    (tmp_path / "allocation.json").write_text('{"dump": "UserId=' + 'alice(1000) GroupId=' + 'staff(20)"}\n')
    (tmp_path / "runtime.json").write_text(
        '{\n  "platform": "Linux-5.15.0-1032-' + 'oracle-x86_64",\n  "SLURM_JOB_' + 'ID": "1234567"\n}\n')
    (tmp_path / "assets.json").write_text('{"endpoint": "' + 'https://objects.example"}\n')
    (tmp_path / "parsers.py").write_text(
        'parser.add_argument("--wandb_' + 'entity", type=str, default="' + 'team")\n')
    failures = audit(tmp_path, denylist=[])["failures"]
    for expected in ("site-specific #SBATCH directive: job.sbatch:2",
                     "hardcoded container mount: launch.sbatch:1",
                     "Slurm job record: allocation.json:1",
                     "cloud host kernel string: runtime.json:2",
                     "recorded Slurm job ID: runtime.json:3",
                     "object-store endpoint: assets.json:1",
                     "W&B entity default: parsers.py:1"):
        assert expected in failures


def test_audit_accepts_placeholders_and_portable_records(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text(
        "sbatch --account=your_account --partition=your_gpu_partition run.sbatch\n"
        "export WFM_CONTAINER_MOUNTS=/path/to/data:/path/to/data\n")
    (tmp_path / "run.sbatch").write_text(
        '#SBATCH --time=01:00:00\n: "${WFM_CONTAINER_MOUNTS:?Set mounts}"\n'
        'srun --container-mounts="$WFM_CONTAINER_MOUNTS" true\n')
    (tmp_path / "runtime.json").write_text(
        '{"platform": "Linux-x86_64-with-glibc2.35", "allocation": "alloc-0123456789ab"}\n')
    assert audit(tmp_path, denylist=[])["ok"]


def test_reviewed_upstream_exception_is_scoped_to_its_check_and_path(tmp_path: Path, monkeypatch) -> None:
    import scripts.audit_source as audit_source

    reason, reviewed = "cloud host kernel string", "vendor/upstream/cluster.py"
    monkeypatch.setattr(audit_source, "REVIEWED_UPSTREAM", {(reason, reviewed)})
    kernel = '"5.4.0-1051-' + 'aws"\n'
    (tmp_path / reviewed).parent.mkdir(parents=True)
    (tmp_path / reviewed).write_text(kernel)
    (tmp_path / "other.py").write_text(kernel)
    assert audit(tmp_path, denylist=[])["failures"] == [f"{reason}: other.py:1"]


def test_private_denylist_reports_line_not_term(tmp_path: Path) -> None:
    denylist = tmp_path.parent / (tmp_path.name + "-denylist.txt")
    denylist.write_text("# private terms\n\nsecret-cluster\\d+\n")
    (tmp_path / "notes.md").write_text("ran on\nsecret-cluster7\n")
    try:
        terms = load_denylist(denylist)
        assert audit(tmp_path, denylist=[])["ok"]
        result = audit(tmp_path, denylist=terms)
        assert result["failures"] == [f"denylisted term ({denylist.name}:3): notes.md:2"]
        assert "secret" not in result["failures"][0]
    finally:
        denylist.unlink()


def test_audit_retains_internal_symlinks_and_finds_weights(tmp_path: Path) -> None:
    (tmp_path / "module.py").write_text("value = 1\n")
    (tmp_path / "alias.py").symlink_to("module.py")
    assert audit(tmp_path)["ok"]
    (tmp_path / "model.pth").write_bytes(b"model")
    assert not audit(tmp_path)["ok"]
