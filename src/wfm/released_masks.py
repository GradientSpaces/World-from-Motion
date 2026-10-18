# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Package and install exact evaluation masks with frame-bound content hashes."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import shutil
import zipfile

from PIL import Image


def digest(data):
    return hashlib.sha256(data).hexdigest()


def image_identity(data, *, rgb=False):
    with Image.open(io.BytesIO(data)) as image:
        if image.format != "PNG":
            raise ValueError("Expected a PNG image")
        image.load()
        if rgb:
            image = image.convert("RGB")
        return {"size": list(image.size), "mode": image.mode,
                "pixels_sha256": digest(image.tobytes())}


def png_files(directory):
    files = {p.name: p for p in directory.glob("*.png") if p.is_file()}
    if not files:
        raise ValueError(f"No PNG files in {directory}")
    return files


def frame_record(mask, rgb):
    data = mask.read_bytes()
    identity = image_identity(data)
    reference = image_identity(rgb.read_bytes(), rgb=True)
    if identity["size"] != reference["size"]:
        raise ValueError(f"Mask/RGB dimensions differ: {mask.name}")
    return {"name": mask.name, "bytes": len(data), "sha256": digest(data),
            **identity, "reference_rgb": reference}


def build_bundle(runs_manifest, assets_root, output, label):
    """Read an explicit run manifest; publish only masks and portable identities.

    The label identifies the selected source; hashing current workspace files does
    not by itself prove they are unchanged since the historical experiment.
    """
    output = Path(output).resolve()
    assets_root = Path(assets_root).resolve()
    if output.exists():
        raise FileExistsError(output)
    raw = Path(runs_manifest).read_bytes()
    rows = json.loads(raw)["runs"]
    records, sources, seen = [], [], set()
    for row in rows:
        method, scene = row["method"], row["scene"]
        if method not in ("ms", "wt") or not scene or Path(scene).name != scene or scene in (".", ".."):
            raise ValueError("Invalid method or scene")
        if (method, scene) in seen:
            raise ValueError(f"Duplicate method/scene: {method}/{scene}")
        seen.add((method, scene))
        workspace = (assets_root / row["workspace"]).resolve()
        if output == workspace or workspace in output.parents or output in workspace.parents:
            raise ValueError("Bundle output must be separate from source workspaces")
        masks = png_files(workspace / "test_dynamic")
        images = png_files(workspace / "test_images")
        if masks.keys() != images.keys():
            raise ValueError(f"Mask/frame coverage mismatch: {method}/{scene}")
        frames = [frame_record(masks[name], images[name]) for name in sorted(images)]
        # TTO uses workspace, whereas static scoring can use metric_workspace.
        # A single installed set represents both roles only if their inputs match.
        metric_workspace = (assets_root / row.get("metric_workspace", row["workspace"])).resolve()
        if output == metric_workspace or metric_workspace in output.parents or output in metric_workspace.parents:
            raise ValueError("Bundle output must be separate from metric workspaces")
        if metric_workspace != workspace:
            metric_masks = png_files(metric_workspace / "test_dynamic")
            metric_images = png_files(metric_workspace / "test_images")
            if metric_masks.keys() != images.keys() or metric_images.keys() != images.keys():
                raise ValueError(f"Alignment/scoring frame coverage differs: {method}/{scene}")
            metric_frames = [frame_record(metric_masks[name], metric_images[name])
                             for name in sorted(images)]
            if metric_frames != frames:
                raise ValueError(f"Alignment/scoring mask or RGB identities differ: {method}/{scene}; "
                                 "separate historical roles cannot be collapsed into one mask set")
        records.append({"method": method, "scene": scene, "frames": frames,
                        "source_role": "workspace", "metric_workspace_identity_checked": True})
        sources.append((method, scene, masks, frames))
    if not records:
        raise ValueError("Empty run manifest")
    manifest = {"schema_version": 1, "label": label,
                "source_runs_sha256": digest(raw), "sets": records,
                "scope": "Exact selected workspace test_dynamic PNGs; no RGB, priors or models. "
                         "Historical identity requires a separate source audit."}
    output.mkdir(parents=True)
    for method, scene, masks, frames in sources:
        folder = output / method / scene / "test_dynamic"
        folder.mkdir(parents=True)
        for frame in frames:
            data = masks[frame["name"]].read_bytes()
            if digest(data) != frame["sha256"]:
                raise ValueError("Source mask changed during packaging")
            with (folder / frame["name"]).open("xb") as handle:
                handle.write(data)
    manifest_data = (json.dumps(manifest, indent=2) + "\n").encode()
    (output / "manifest.json").write_bytes(manifest_data)
    return {"manifest_sha256": digest(manifest_data), "sets": len(records),
            "files": sum(len(row["frames"]) for row in records),
            "bytes": sum(f["bytes"] for row in records for f in row["frames"])}


def write_archive(bundle, archive):
    """Store a bundle as one reproducible ZIP so it downloads as a single file."""
    bundle, archive = Path(bundle).resolve(), Path(archive)
    files = sorted(p for p in bundle.rglob("*") if p.is_file())
    with zipfile.ZipFile(archive, "x", zipfile.ZIP_STORED) as zf:
        for path in files:
            # Fixed timestamps and permissions keep the archive hash reproducible.
            info = zipfile.ZipInfo(f"{bundle.name}/{path.relative_to(bundle).as_posix()}", (1980, 1, 1, 0, 0, 0))
            info.external_attr = 0o644 << 16
            zf.writestr(info, path.read_bytes())
    return {"archive_sha256": digest(archive.read_bytes()), "bytes": archive.stat().st_size,
            "members": len(files)}


def extract_archive(archive, destination):
    """Unpack only regular files below the archive's single top-level bundle directory."""
    destination = Path(destination)
    with zipfile.ZipFile(archive) as zf:
        members = zf.infolist()
        tops = {PurePosixPath(item.filename).parts[0] for item in members if item.filename}
        if len(tops) != 1:
            raise ValueError("Mask archive must hold exactly one bundle directory")
        for item in members:
            path = PurePosixPath(item.filename)
            if path.is_absolute() or ".." in path.parts or "\\" in item.filename:
                raise ValueError(f"Unsafe mask archive member: {item.filename}")
            if (item.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError(f"Mask archive member cannot be a symlink: {item.filename}")
            if item.is_dir():
                continue
            target = destination.joinpath(*path.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(item) as source, target.open("xb") as handle:
                shutil.copyfileobj(source, handle)
    return destination / tops.pop()


def verify_bundle_files(bundle, manifest_sha256):
    """Check a downloaded bundle file by file against its pinned manifest; no observations needed."""
    bundle = Path(bundle).resolve()
    data = (bundle / "manifest.json").read_bytes()
    if not manifest_sha256 or digest(data) != manifest_sha256:
        raise ValueError("Released-mask manifest SHA-256 mismatch")
    manifest = json.loads(data)
    if manifest["schema_version"] != 1:
        raise ValueError("Unsupported released-mask schema")
    expected, total = {"manifest.json"}, 0
    for row in manifest["sets"]:
        method, scene = row["method"], row["scene"]
        if method not in ("ms", "wt") or not scene or Path(scene).name != scene or scene in (".", ".."):
            raise ValueError("Invalid method or scene")
        for frame in row["frames"]:
            name = frame["name"]
            if Path(name).name != name or not name.endswith(".png"):
                raise ValueError("Invalid mask filename")
            relative = f"{method}/{scene}/test_dynamic/{name}"
            if relative in expected:
                raise ValueError(f"Duplicate mask: {relative}")
            expected.add(relative)
            mask_data = (bundle / relative).read_bytes()
            if len(mask_data) != frame["bytes"] or digest(mask_data) != frame["sha256"]:
                raise ValueError(f"Released-mask SHA-256 mismatch: {relative}")
            total += len(mask_data)
    present = {p.relative_to(bundle).as_posix() for p in bundle.rglob("*") if not p.is_dir()}
    if present != expected:
        raise ValueError(f"Released-mask bundle has unlisted files: {sorted(present - expected)[:5]}")
    return {"label": manifest["label"], "sets": len(manifest["sets"]),
            "files": len(expected) - 1, "bytes": total}


def verify_bundle(bundle, manifest_sha256, method, scene, workspace):
    """Read-only verification; return provenance and the verified mask bytes."""
    bundle, workspace = Path(bundle).resolve(), Path(workspace).resolve()
    if method not in ("ms", "wt") or not scene or Path(scene).name != scene or scene in (".", ".."):
        raise ValueError("Invalid method or scene")
    data = (bundle / "manifest.json").read_bytes()
    if not manifest_sha256 or digest(data) != manifest_sha256:
        raise ValueError("Released-mask manifest SHA-256 mismatch")
    manifest = json.loads(data)
    if manifest["schema_version"] != 1:
        raise ValueError("Unsupported released-mask schema")
    selected = [row for row in manifest["sets"] if (row["method"], row["scene"]) == (method, scene)]
    if len(selected) != 1:
        raise ValueError("Expected one mask set for the requested method/scene")
    frames = selected[0]["frames"]
    names = [f["name"] for f in frames]
    if any(Path(name).name != name or not name.endswith(".png") for name in names):
        raise ValueError("Invalid mask filename")
    images = png_files(workspace / "test_images")
    if len(set(names)) != len(names) or set(names) != set(images):
        raise ValueError("Released-mask frame coverage mismatch")
    verified = []
    for frame in frames:
        source = (bundle / method / scene / "test_dynamic" / frame["name"]).resolve()
        if not source.is_relative_to(bundle):
            raise ValueError("Mask source escapes bundle")
        mask_data = source.read_bytes()
        if len(mask_data) != frame["bytes"] or digest(mask_data) != frame["sha256"]:
            raise ValueError(f"Released-mask SHA-256 mismatch: {frame['name']}")
        identity = image_identity(mask_data)
        if any(identity[k] != frame[k] for k in identity):
            raise ValueError(f"Released-mask image identity mismatch: {frame['name']}")
        reference = image_identity(images[frame["name"]].read_bytes(), rgb=True)
        if reference != frame["reference_rgb"] or reference["size"] != identity["size"]:
            raise ValueError(f"Held-out RGB identity mismatch: {frame['name']}")
        verified.append((frame["name"], mask_data))
    report = {"source": "released_bundle", "label": manifest["label"],
            "manifest_sha256": digest(data), "method": method, "scene": scene,
            "frames": len(frames), "reference_rgb_verified": True,
            "mask_sha256": {f["name"]: f["sha256"] for f in frames},
            "note": "Pinned released-mask identity verified; no SAM3 regeneration required."}
    return report, verified


def install_bundle(bundle, manifest_sha256, method, scene, workspace):
    """Validate every input before creating a fresh test_dynamic directory."""
    destination = Path(workspace) / "test_dynamic"
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    report, verified = verify_bundle(bundle, manifest_sha256, method, scene, workspace)
    destination.mkdir()
    for name, mask_data in verified:
        with (destination / name).open("xb") as handle:
            handle.write(mask_data)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-manifest", type=Path, required=True)
    parser.add_argument("--assets-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--archive", type=Path, help="Also write the bundle as one reproducible ZIP")
    args = parser.parse_args()
    report = build_bundle(args.runs_manifest, args.assets_root, args.output, args.label)
    if args.archive:
        report["archive"] = write_archive(args.output, args.archive)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
