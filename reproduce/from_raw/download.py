# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Download pinned inputs; extract observations without accepting learned scene priors."""
from __future__ import annotations

import argparse
import http.client
import json
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import time
import urllib.request
import zipfile

from stages import OBSERVATIONS, sha256, save

HERE = Path(__file__).resolve().parent
SCENES = ("apple", "block", "paper-windmill", "space-out", "spin", "teddy", "wheel")


def verify(path, spec):
    if path.stat().st_size != spec["bytes"] or sha256(path) != spec["sha256"]:
        raise ValueError(f"Asset size or SHA-256 mismatch: {path}")


def fetch_https(source, partial, expected_size, *, chunk_bytes=64 * 1024 * 1024,
                attempts=4):
    """Retry bounded transfers and resume partial inputs; final SHA-256 is mandatory."""
    offset = partial.stat().st_size if partial.exists() else 0
    if not 0 <= offset <= expected_size:
        raise ValueError(f"Partial download exceeds the pinned size: {partial}")
    if chunk_bytes < 1 or attempts < 1:
        raise ValueError("Download chunk size and attempt count must be positive")
    with partial.open("ab") as out:
        while offset < expected_size:
            end = min(offset + chunk_bytes, expected_size) - 1
            length = end - offset + 1
            request = urllib.request.Request(source, headers={"Range": f"bytes={offset}-{end}"})
            for attempt in range(attempts):
                try:
                    with urllib.request.urlopen(request, timeout=120) as response:
                        if response.status == 206:
                            wanted = f"bytes {offset}-{end}/{expected_size}"
                            if response.headers.get("Content-Range") != wanted:
                                raise ValueError("HTTP range does not match the pinned asset extent")
                        elif not (response.status == 200 and offset == 0 and length == expected_size):
                            raise ValueError("Server did not honor the requested asset range")
                        block = response.read(length + 1)
                        if len(block) != length:
                            raise OSError(f"Incomplete HTTP range: expected {length}, received {len(block)}")
                    break
                except (OSError, http.client.IncompleteRead) as exc:
                    if attempt + 1 == attempts:
                        raise
                    print(f"Retrying bytes {offset}-{end}: {type(exc).__name__}", flush=True)
                    time.sleep(min(2 ** attempt, 8))
            out.write(block)
            out.flush()
            offset += length
            print(f"Received {offset}/{expected_size} bytes for {partial.name}", flush=True)


def fetch(root, spec, *, profile=None, endpoint=None, url=None):
    target = root / spec["path"]
    if target.exists():
        verify(target, spec)
        return target
    source = url or spec.get("url")
    if not source:
        raise ValueError("WfM checkpoint hosting is pending. Supply --wfm-url or place the exact "
                         "checkpoint at models/wfm-step-12500.safetensors under --root.")
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    print(f"Downloading {spec['path']}", flush=True)
    if spec.get("hf_repo") and url is None:
        # The official client handles gated access and redirects without exposing
        # credentials. Users obtain SAM3 access through the publisher themselves.
        from huggingface_hub import hf_hub_download
        downloaded = Path(hf_hub_download(spec["hf_repo"], spec["hf_file"],
                                         revision=spec["revision"]))
        shutil.copyfile(downloaded, partial)
    elif source.startswith("s3://"):
        command = ["aws"]
        if profile:
            command += ["--profile", profile]
        if endpoint:
            command += ["--endpoint-url", endpoint]
        subprocess.run(command + ["s3", "cp", source, str(partial), "--no-progress"], check=True)
    elif source.startswith("https://"):
        fetch_https(source, partial, spec["bytes"])
    else:
        raise ValueError("Expected an HTTPS or S3 download URL")
    verify(partial, spec)
    partial.replace(target)
    return target


def extract_observations(archive, destination, scenes):
    """Use only the six explicit observation directories from the public ZIP."""
    if destination.exists():
        raise FileExistsError(f"Use a fresh observation directory: {destination}")
    selected = {}
    with zipfile.ZipFile(archive) as zf:
        for item in zf.infolist():
            parts = PurePosixPath(item.filename).parts
            if ".." in parts or PurePosixPath(item.filename).is_absolute():
                raise ValueError("Unsafe archive member")
            if item.is_dir() or len(parts) < 4:
                continue
            # Public WorldTree archive: released_data/dycheck/iphone/<scene>/<kind>/<file>.
            if parts[-4] != "iphone" or parts[-3] not in scenes or parts[-2] not in OBSERVATIONS:
                continue
            rel = Path(*parts[-3:])
            if rel in selected:
                raise ValueError(f"Duplicate observation: {rel}")
            if (item.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError(f"Observation cannot be a symlink: {rel}")
            selected[rel] = item
        for scene in scenes:
            for kind in OBSERVATIONS:
                if not any(p.parts[:2] == (scene, kind) for p in selected):
                    raise ValueError(f"Public archive lacks {scene}/{kind}")
        destination.mkdir(parents=True)
        for rel, item in selected.items():
            target = destination / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(item) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)
    save(destination / "source.json", {"archive_sha256": sha256(archive),
         "scenes": list(scenes), "observation_directories": OBSERVATIONS,
         "files_extracted": len(selected), "prepared_priors_imported": False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--group", choices=("all", "data", "models", "wfm"), default="all")
    parser.add_argument("--scenes", choices=SCENES, nargs="+", default=list(SCENES))
    parser.add_argument("--wfm-url", help="Authorized HTTPS/S3 location for the pinned WfM checkpoint")
    parser.add_argument("--profile", help="Optional AWS profile for an S3 checkpoint URL")
    parser.add_argument("--endpoint", help="Optional S3 endpoint for the checkpoint URL")
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    manifest = json.loads((HERE / "assets.lock.json").read_text())
    if args.group in ("all", "wfm"):
        checkpoint = root / manifest["wfm"]["path"]
        if checkpoint.exists():
            verify(checkpoint, manifest["wfm"])
        elif not (args.wfm_url or manifest["wfm"].get("url")):
            parser.error("WfM checkpoint hosting is pending. Supply --wfm-url or place the exact "
                         "checkpoint at models/wfm-step-12500.safetensors under --root. "
                         "No downloads have started.")
    for spec in manifest["artifacts"]:
        if args.group not in ("all", spec["group"]):
            continue
        path = fetch(root, spec)
        for member in spec.get("extract", []):
            target = root / member["path"]
            if target.exists():
                verify(target, member)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            partial = target.with_name(target.name + ".part")
            with zipfile.ZipFile(path) as zf, zf.open(member["member"]) as src, partial.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            verify(partial, member)
            partial.replace(target)
    if args.group in ("all", "wfm"):
        spec = manifest["wfm"]
        fetch(root, spec, profile=args.profile, endpoint=args.endpoint, url=args.wfm_url)
    if args.group in ("all", "data"):
        extract_observations(root / "downloads/released_data.zip", root / "observations", args.scenes)
    save(root / f"downloads-{args.group}.json", {"manifest_sha256": sha256(HERE / "assets.lock.json"),
         "group": args.group, "status": "verified", "scenes": args.scenes})


if __name__ == "__main__":
    main()
