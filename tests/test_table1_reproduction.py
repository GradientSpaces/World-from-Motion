"""CPU checks for incomplete results, aggregation errors, and asset protection."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from openpyxl import Workbook

from reproduce.table1.run import (
    COLUMNS, HERE, SCENES, average_scenes, fresh_output, read_manifest,
    jobs, read_workbook, validate_fit_workspace, validate_tree, verify_runs,
)
from reproduce.table1.collect_slurm import collect
from reproduce.table1.compare import compare
from reproduce.table1.slurm.fit_batch import assignments, run_batch, validate_task


class Table1Tests(unittest.TestCase):
    def test_backend_origin_metadata_matches_release_files(self):
        provenance = json.loads((HERE / "backends/provenance.json").read_text())
        destinations = set()
        for record in provenance["files"]:
            destination = record["destination"]
            with self.subTest(destination=destination):
                self.assertNotIn(destination, destinations)
                destinations.add(destination)
                fingerprint = hashlib.sha256((HERE / destination).read_bytes()).hexdigest()
                self.assertEqual(record["integrated_sha256"], fingerprint)
                self.assertEqual(record["modified_for_release"],
                                 fingerprint != record["source_sha256"])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def workbook(self, rows, columns=("fn", "mpsnr")):
        path = self.root / "metrics.xlsx"
        book = Workbook()
        sheet = book.active
        sheet.append(columns)
        for row in rows:
            sheet.append(row)
        book.save(path)
        return path

    def test_workbook_validates_all_frame_values(self):
        p = self.workbook([("AVE", 21), ("1_000.png", 20), ("1_001.png", 22)])
        self.assertEqual(read_workbook(p, ("mpsnr",), 2), {"mpsnr": 21})
        p = self.workbook([("AVE", 25), ("1_000.png", 20), ("1_001.png", 22)])
        with self.assertRaisesRegex(ValueError, "AVE differs"):
            read_workbook(p, ("mpsnr",), 2)

    def test_rejects_missing_frames_duplicates_and_placeholder_scores(self):
        examples = [
            ([("AVE", 20), ("a", 20)], "frames"),
            ([("AVE", 20), ("a", 20), ("a", 20)], "Duplicate"),
            ([("AVE", 0), ("a", 0), ("b", 0)], "placeholder"),
            ([("AVE", 20), ("a", 20), ("b", None)], "Invalid numeric"),
        ]
        for rows, error in examples:
            with self.subTest(error=error):
                with self.assertRaisesRegex(ValueError, error):
                    read_workbook(self.workbook(rows), ("mpsnr",), 2)

    def test_equal_scene_weights_and_complete_scene_set(self):
        records = [{"method": "ms", "scene": scene, "frame_rows": 10 ** i,
                    "metrics": dict.fromkeys(COLUMNS, float(i))}
                   for i, scene in enumerate(SCENES)]
        self.assertEqual(average_scenes(records)["ms"]["mpsnr"], 3.0)
        with self.assertRaisesRegex(ValueError, "seven scenes"):
            average_scenes(records[:-1])
        with self.assertRaisesRegex(ValueError, "seven scenes"):
            average_scenes(records[:-1] + [records[0]])
        records[0]["metrics"]["mpsnr"] = float("nan")
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            average_scenes(records)

    def test_archived_reference_keeps_the_agreed_camera_ready_correction(self):
        reference = json.loads((HERE / "reference.json").read_text())
        records = [{"method": method, "scene": scene, **entry}
                   for method, scenes in reference["scenes"].items()
                   for scene, entry in scenes.items()]
        rows = average_scenes(records)
        self.assertEqual(f"{rows['ms']['mpsnr']:.2f}", "19.74")
        self.assertEqual(f"{rows['wt']['pose_mpsnr']:.2f}", "20.26")
        for method, values in rows.items():
            for column, value in values.items():
                self.assertAlmostEqual(value, reference["rows"][method][column], places=12)

    def test_no_existing_or_overlapping_output(self):
        source = self.root / "inputs"
        source.mkdir()
        alias = self.root / "alias"
        alias.symlink_to(source, target_is_directory=True)
        for candidate in (source, self.root, source / "new", alias / "new"):
            with self.assertRaises((ValueError, FileExistsError)):
                fresh_output(candidate, [source])
        self.assertEqual(fresh_output(self.root / "outputs", [source]), self.root / "outputs")

    def test_missing_nested_ancestor_is_detected(self):
        (self.root / "photometric_cam.pth").write_bytes(b"checkpoint")
        reference = {"checkpoints": {"ms/apple": [
            {"path": "photometric_cam.pth"},
            {"path": "dst_L000_R003/dst_L000_R007/photometric_d_model_native_add3.pth"},
        ]}}
        with self.assertRaisesRegex(FileNotFoundError, "dst_L000_R007"):
            verify_runs([{"method": "ms", "scene": "apple", "checkpoint": self.root}], reference)

    def test_tree_split_consistency(self):
        for lo, hi, split in [(0, 7, 4), (0, 3, 2), (4, 7, 6),
                              (0, 1, 1), (2, 3, 3), (4, 5, 5), (6, 7, 7)]:
            p = self.root / f"dst_L{lo:03}_R{hi:03}"
            p.mkdir()
            (p / "split_ind.txt").write_text(str(split))
        validate_tree(self.root)
        (self.root / "dst_L000_R007/split_ind.txt").write_text("3")
        with self.assertRaisesRegex(ValueError, "Invalid split"):
            validate_tree(self.root)

    def test_manifests_are_portable_and_have_no_duplicate_runs(self):
        for filename in ("retained_runs.json", "runs.example.json"):
            runs = read_manifest(HERE / filename, self.root)
            self.assertEqual(len(runs), 14)
            raw = json.loads((HERE / filename).read_text())
            self.assertTrue(all(not Path(r["checkpoint"]).is_absolute() for r in raw["runs"]))
            raw["runs"].append(raw["runs"][0])
            p = self.root / filename
            p.write_text(json.dumps(raw))
            with self.assertRaisesRegex(ValueError, "duplicate"):
                read_manifest(p, self.root)

    def test_new_fits_do_not_inherit_archived_evaluation_metrics(self):
        workspace = self.root / "workspace"
        workspace.mkdir()
        run = {"method": "ms", "scene": "apple", "checkpoint": self.root / "old-model",
               "workspace": workspace, "metric_workspace": workspace,
               "static_metrics": self.root / "old-static.xlsx",
               "per_frame_metrics": self.root / "old-pose.xlsx"}
        args = SimpleNamespace(action="fit", output=self.root / "new-fit", mode="both",
                               python="python", dry_run=False, fit_preset="reference")
        with patch("reproduce.table1.run.subprocess.run"), \
                patch("reproduce.table1.run.validate_fit_workspace"):
            jobs(args, [run], {})
        generated = json.loads((args.output / "runs.json").read_text())["runs"][0]
        command = json.loads((args.output / "plan.json").read_text())[0]["command"]
        self.assertEqual(command[-2:], ["--fit-preset", "reference"])
        for field in ("static_metrics", "per_frame_metrics"):
            self.assertNotEqual(generated[field], str(run[field]))
            self.assertTrue(Path(generated[field]).is_relative_to(args.output))

    def test_fit_requires_all_eight_virtual_cameras(self):
        manifest = self.root / "multiview_manifest.json"
        manifest.write_text(json.dumps({"cameras": [{"camera_name": "cam0"}]}))
        with self.assertRaisesRegex(ValueError, "exactly cam100"):
            validate_fit_workspace({"method": "ms", "workspace": self.root})

    def test_slurm_collection_requires_all_fresh_scene_results(self):
        for method in ("ms", "wt"):
            for scene in SCENES:
                task = self.root / f"{method}_{scene}"
                task.mkdir()
                row = {"method": method, "scene": scene, "checkpoint": "checkpoint",
                       "workspace": "workspace", "metric_workspace": "metric_workspace"}
                for mode in ("static", "per_frame"):
                    path = task / f"{mode}.xlsx"
                    path.write_bytes(b"metric placeholder for manifest-only test")
                    row[f"{mode}_metrics"] = str(path)
                (task / "runs.json").write_text(json.dumps({"schema_version": 1, "runs": [row]}))
        self.assertEqual(len(collect(self.root)["runs"]), 14)
        manifest = self.root / "wt_wheel/runs.json"
        payload = json.loads(manifest.read_text())
        payload["runs"][0]["per_frame_metrics"] = str(self.root / "missing.xlsx")
        manifest.write_text(json.dumps(payload))
        with self.assertRaisesRegex(ValueError, "Missing fresh"):
            collect(self.root)

    def test_comparison_reports_optimization_differences_without_rejecting_them(self):
        reference = json.loads((HERE / "reference.json").read_text())
        scenes = [{"method": "ms", "scene": scene, "metrics": dict(entry["metrics"])}
                  for scene, entry in reference["scenes"]["ms"].items()]
        wheel = next(r for r in scenes if r["scene"] == "wheel")
        wheel["metrics"]["psnr"] += 0.07
        results = {"scenes": scenes, "rows": average_scenes(scenes)}
        report = compare(results, reference)
        self.assertAlmostEqual(report["max_absolute_delta"]["table"]["psnr"], 0.01)
        self.assertAlmostEqual(report["max_absolute_delta"]["scene"]["psnr"], 0.07)
        self.assertTrue(any(not r["rounded_equal"] for r in report["records"]))
        results["rows"]["ms"]["psnr"] += 1
        with self.assertRaisesRegex(ValueError, "differs from scene means"):
            compare(results, reference)

    def test_concurrent_scenes_have_disjoint_gpu_and_cpu_assignments(self):
        for method, count in (("ms", 6), ("wt", 3), ("ms", 7), ("wt", 4)):
            tasks = assignments(method, SCENES[:count], list("01234567"), list(range(124)), 16)
            width = 1 if method == "ms" else 2
            self.assertEqual(len({gpu for task in tasks for gpu in task["gpus"]}), count * width)
            self.assertEqual(len({cpu for task in tasks for cpu in task["cpus"]}), count * 16)
        for scenes, gpus, cpus in ((["apple", "apple"], ["0", "1"], [0, 1]),
                                   (["apple", "block"], ["0"], [0, 1]),
                                   (["apple", "block"], ["0", "1"], [0]),
                                   (["apple"], ["0", "0"], [0])):
            with self.assertRaises(ValueError):
                assignments("ms", scenes, gpus, cpus, 1)

    def test_batch_fit_failure_skips_evaluation_and_other_scene_completes(self):
        repo = self.root / "repo"
        source = repo / "reproduce/table1"
        source.mkdir(parents=True)
        (source / "provenance.py").write_text("# CPU-only subprocess fixture\n")
        (source / "run.py").write_text('''import json, os, pathlib, sys
args = sys.argv
output = pathlib.Path(args[args.index("--output") + 1])
output.mkdir(parents=True)
(output / "observed.json").write_text(json.dumps({
    "gpus": os.environ["CUDA_VISIBLE_DEVICES"], "cpus": sorted(os.sched_getaffinity(0))}))
if args[1] == "fit" and args[args.index("--scene") + 1] == "apple":
    sys.exit(3)
if args[1] == "evaluate":
    assert pathlib.Path(args[args.index("--manifest") + 1]).is_file()
(output / "runs.json").write_text("{}")
''')
        cpus = sorted(os.sched_getaffinity(0))
        if len(cpus) < 2:
            self.skipTest("Needs two available CPU IDs to test affinity separation")
        tasks = assignments("ms", ["apple", "block"], ["2", "5"], cpus, 1)
        for task in tasks:
            task.update(output=str(self.root / task["scene"]),
                        cache=str(self.root / "cache" / task["scene"]))
        plan = {"repo": str(repo), "manifest": "unused.json", "assets_root": str(self.root),
                "tasks": tasks}
        directory = self.root / "batch"
        self.assertEqual(run_batch(plan, directory), 1)
        self.assertFalse((self.root / "apple/evaluate").exists())
        self.assertFalse((self.root / "apple/runs.json").exists())
        self.assertTrue((self.root / "block/runs.json").is_file())
        for task in tasks:
            observed = json.loads((Path(task["output"]) / "fit/observed.json").read_text())
            self.assertEqual(observed, {"gpus": ",".join(task["gpus"]), "cpus": task["cpus"]})
        states = json.loads((directory / "status.json").read_text())["tasks"]
        self.assertNotEqual(states[0]["returncode"], 0)
        self.assertEqual(states[1]["returncode"], 0)
        failed = json.loads((self.root / "cache/apple/timing.json").read_text())["stages"]
        passed = json.loads((self.root / "cache/block/timing.json").read_text())["stages"]
        self.assertEqual([stage["name"] for stage in failed], ["provenance", "fit"])
        self.assertEqual(failed[-1]["returncode"], 3)
        self.assertEqual([stage["name"] for stage in passed], ["provenance", "fit", "evaluate"])
        for stage in failed + passed:
            self.assertGreater(stage["seconds"], 0)
            self.assertLessEqual(stage["started_utc"], stage["finished_utc"])
        self.assertTrue(all(stage["returncode"] == 0 for stage in passed))

    def test_separate_fit_and_evaluation_allocations_keep_the_fresh_checkpoint(self):
        repo = self.root / "repo"
        source = repo / "reproduce/table1"
        source.mkdir(parents=True)
        (source / "provenance.py").write_text("# CPU-only subprocess fixture\n")
        (source / "run.py").write_text('''import json, pathlib, sys
args = sys.argv
output = pathlib.Path(args[args.index("--output") + 1])
output.mkdir(parents=True, exist_ok=False)
with (output.parent / "actions.txt").open("a") as handle:
    handle.write(args[1] + "\\n")
if args[1] == "fit":
    assert args[args.index("--fit-preset") + 1] == "reference"
    payload = {"runs": [{"method": "ms", "scene": "block", "checkpoint": str(output)}]}
else:
    assert "--fit-preset" not in args
    payload = json.loads(pathlib.Path(args[args.index("--manifest") + 1]).read_text())
(output / "runs.json").write_text(json.dumps(payload))
''')
        task = assignments("ms", ["block"], ["3"], sorted(os.sched_getaffinity(0)), 1)[0]
        output = self.root / "ms_block"
        task.update(output=str(output), cache=str(self.root / "cache/fit"))
        plan = {"repo": str(repo), "manifest": "unused.json", "assets_root": str(self.root),
                "phase": "fit", "fit_preset": "reference", "tasks": [task]}
        self.assertEqual(run_batch(plan, self.root / "fit-batch"), 0)
        self.assertTrue((output / "fit/runs.json").is_file())
        self.assertFalse((output / "runs.json").exists())
        self.assertFalse((output / "evaluate").exists())
        task["cache"] = str(self.root / "cache/evaluate")
        plan["phase"] = "evaluate"
        validate_task(task, "evaluate")
        self.assertEqual(run_batch(plan, self.root / "evaluation-batch"), 0)
        self.assertEqual((output / "actions.txt").read_text().splitlines(), ["fit", "evaluate"])
        self.assertTrue((output / "runs.json").is_file())
        fitting = json.loads((self.root / "cache/fit/timing.json").read_text())
        evaluation = json.loads((self.root / "cache/evaluate/timing.json").read_text())
        self.assertEqual([stage["name"] for stage in fitting["stages"]], ["provenance", "fit"])
        self.assertEqual([stage["name"] for stage in evaluation["stages"]], ["provenance", "evaluate"])
        self.assertEqual(fitting["fit_preset"], "reference")
        with self.assertRaises(FileExistsError):
            validate_task(task, "evaluate")

    def test_evaluation_phase_rejects_an_archived_checkpoint_manifest(self):
        output = self.root / "ms_block"
        (output / "fit").mkdir(parents=True)
        manifest = output / "fit/runs.json"
        manifest.write_text(json.dumps({"runs": [
            {"method": "ms", "scene": "block", "checkpoint": str(self.root)}]}))
        task = {"method": "ms", "scene": "block", "output": str(output),
                "cache": str(self.root / "cache")}
        with self.assertRaisesRegex(ValueError, "newly fitted checkpoint"):
            validate_task(task, "evaluate")


if __name__ == "__main__":
    unittest.main()
