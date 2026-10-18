# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Download pinned inputs; extract observations without accepting learned scene priors."""
from __future__ import annotations

import argparse
import http.client
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import time
import urllib.parse
import urllib.request
import zipfile

from stages import OBSERVATIONS, sha256, save

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SCENES = ("apple", "block", "paper-windmill", "space-out", "spin", "teddy", "wheel")
REVISION = re.compile(r"[0-9a-f]{40}")
# The null commit marks a Hugging Face entry whose upload is not published yet.
UNPUBLISHED = "0" * 40


def verify(path, spec):
    if path.stat().st_size != spec["bytes"] or sha256(path) != spec["sha256"]:
        raise ValueError(f"Asset size or SHA-256 mismatch: {path}")


def pinned(manifest):
    return [*manifest["artifacts"], manifest["wfm"], *([manifest["masks"]] if "masks" in manifest else [])]


def check_lock(manifest):
    """Pinned sources must name an immutable 40-hex commit, never a branch or tag."""
    for spec in pinned(manifest):
        if ("hf_repo" in spec or "revision" in spec) and not REVISION.fullmatch(str(spec.get("revision"))):
            raise ValueError(f"Expected a 40-hex commit revision for {spec['path']}")


def offline_from_environment():
    return os.environ.get("HF_HUB_OFFLINE", "").upper() in {"1", "ON", "YES", "TRUE"}


def source_of(root, spec, *, mirror=None, url=None):
    """Where fetch() would read an asset from; None when no published source exists."""
    if (root / spec["path"]).exists() or (spec.get("bundle") and (root / spec["bundle"]).exists()):
        return "existing"
    if mirror and (mirror / spec["path"]).exists():
        return "mirror"
    if url:
        return "url"
    if spec.get("hf_repo"):
        return None if spec.get("revision") == UNPUBLISHED else "huggingface"
    return "url" if spec.get("url") else None


def provenance(spec, source, *, mirror=None, url=None):
    record = dict(path=spec["path"], source=source)
    record.update({key: spec[key] for key in ("bytes", "sha256", "manifest_sha256", "files") if key in spec})
    if spec.get("hf_repo") and spec["revision"] != UNPUBLISHED:
        record.update(repo=f"{spec['hf_repo']}@{spec['revision']}",
                      repo_type=spec.get("repo_type", "model"))
    if source == "mirror":
        record["mirror"] = str(mirror)
    elif source == "url":
        # Drop query strings so signed URLs never land in the record.
        record["url"] = urllib.parse.urlsplit(url or spec["url"])._replace(query="", fragment="").geturl()
    return record


def check_published(spec):
    if not REVISION.fullmatch(str(spec.get("revision"))) or spec["revision"] == UNPUBLISHED:
        raise ValueError(f"Expected a published 40-hex commit revision for {spec['path']}")


def fetch_hub(spec, target):
    """Download into a staging directory beside the target, not via the shared HF cache."""
    check_published(spec)
    # The official client handles gated access and redirects without exposing
    # credentials. Users obtain SAM3 access through the publisher themselves.
    from huggingface_hub import hf_hub_download
    staging = target.with_name(f".{target.name}.hf")
    return staging, Path(hf_hub_download(spec["hf_repo"], spec["hf_file"],
                                         repo_type=spec.get("repo_type", "model"),
                                         revision=spec["revision"], local_dir=staging))


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


def fetch(root, spec, *, profile=None, endpoint=None, url=None, mirror=None, offline=False,
          records=None):
    target = root / spec["path"]
    source = source_of(root, spec, mirror=mirror, url=url)
    if source is None:
        raise ValueError(f"No published download for {spec['path']}. Supply --wfm-url or --mirror, "
                         f"or place the exact file at {target}.")
    if offline and source not in ("existing", "mirror"):
        raise ValueError(f"Offline: {spec['path']} is neither under --root nor in the mirror")
    if source == "existing":
        verify(target, spec)
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(target.name + ".part")
        print(f"Fetching {spec['path']} ({source})", flush=True)
        if source == "huggingface":
            staging, staged = fetch_hub(spec, target)
            try:
                verify(staged, spec)
            except ValueError:
                # A rejected file must not be reused as a completed local download.
                shutil.rmtree(staging)
                raise
            staged.replace(target)
            shutil.rmtree(staging)
        elif source == "mirror":
            shutil.copyfile(mirror / spec["path"], partial)
            try:
                verify(partial, spec)
            except ValueError:
                partial.unlink()
                raise
            partial.replace(target)
        else:
            location = url or spec["url"]
            if location.startswith("s3://"):
                command = ["aws"]
                if profile:
                    command += ["--profile", profile]
                if endpoint:
                    command += ["--endpoint-url", endpoint]
                subprocess.run(command + ["s3", "cp", location, str(partial), "--no-progress"], check=True)
            elif location.startswith("https://"):
                fetch_https(location, partial, spec["bytes"])
            else:
                raise ValueError("Expected an HTTPS or S3 download URL")
            verify(partial, spec)
            partial.replace(target)
    if records is not None:
        records.append(provenance(spec, source, mirror=mirror, url=url))
    return target


def fetch_masks(root, spec, *, mirror=None, offline=False, records=None):
    """Fetch the released mask archive, unpack it, and check every mask against its manifest."""
    sys.path.insert(0, str(REPO / "src"))
    from wfm.released_masks import extract_archive, verify_bundle_files

    def verify_masks(bundle):
        if verify_bundle_files(bundle, spec["manifest_sha256"])["files"] != spec["files"]:
            raise ValueError(f"Released-mask count differs from the lock: {spec['bundle']}")

    target = root / spec["bundle"]
    if target.exists():
        verify_masks(target)
        if records is not None:
            records.append(dict(path=spec["bundle"], source="existing",
                                manifest_sha256=spec["manifest_sha256"], files=spec["files"]))
        return target
    archive = fetch(root, spec, mirror=mirror, offline=offline, records=records)
    staging = target.with_name(f".{target.name}.staging")
    if staging.exists():
        shutil.rmtree(staging)
    try:
        staged = extract_archive(archive, staging)
        if staged.name != target.name:
            raise ValueError(f"Mask archive holds {staged.name}, expected {target.name}")
        verify_masks(staged)
        staged.replace(target)
    finally:
        # Never leave a partially verified bundle where a later run could reuse it.
        shutil.rmtree(staging, ignore_errors=True)
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
    parser.add_argument("--group", choices=("all", "data", "models", "sam3", "wfm", "masks"), default="all",
                        help="'all' covers data, models, sam3 and wfm; released evaluation masks "
                             "are the separate 'masks' group")
    parser.add_argument("--scenes", choices=SCENES, nargs="+", default=list(SCENES))
    parser.add_argument("--wfm-url", help="Alternative HTTPS/S3 location for the pinned WfM checkpoint")
    parser.add_argument("--profile", help="Optional AWS profile for an S3 checkpoint URL")
    parser.add_argument("--endpoint", help="Optional S3 endpoint for the checkpoint URL")
    parser.add_argument("--mirror", type=Path, default=os.environ.get("WFM_ASSET_MIRROR") or None,
                        help="Local copy laid out like --root, read before the network "
                             "(default: $WFM_ASSET_MIRROR); files are still SHA-256 verified")
    parser.add_argument("--offline", action="store_true", default=offline_from_environment(),
                        help="Use only files under --root or --mirror (default: on if HF_HUB_OFFLINE is set)")
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    mirror = args.mirror.expanduser().resolve() if args.mirror else None
    manifest = json.loads((HERE / "assets.lock.json").read_text())
    try:
        check_lock(manifest)
    except ValueError as exc:
        parser.error(str(exc))
    plan = [(spec, None) for spec in manifest["artifacts"] if args.group in ("all", spec["group"])]
    if args.group in ("all", "wfm"):
        plan.append((manifest["wfm"], args.wfm_url))
    if args.group == "masks":
        plan.append((manifest["masks"], None))
    sources = {spec["path"]: source_of(root, spec, mirror=mirror, url=url) for spec, url in plan}
    unavailable = [path for path, source in sources.items() if source is None]
    if unavailable:
        parser.error(f"No published download for {', '.join(unavailable)}: the Hugging Face upload "
                     "is not pinned yet. Use --mirror or place the exact files under --root "
                     "(the WfM checkpoint also accepts --wfm-url). No downloads have started.")
    if args.offline:
        remote = [path for path, source in sources.items() if source not in ("existing", "mirror")]
        if remote:
            parser.error(f"Offline, but missing under --root and --mirror: {', '.join(remote)}. "
                         "No downloads have started.")
    if args.group in ("all", "wfm") and sources[manifest["wfm"]["path"]] == "existing":
        verify(root / manifest["wfm"]["path"], manifest["wfm"])
    records = []
    for spec, url in plan:
        if spec is manifest.get("masks"):
            fetch_masks(root, spec, mirror=mirror, offline=args.offline, records=records)
            continue
        path = fetch(root, spec, profile=args.profile, endpoint=args.endpoint, url=url,
                     mirror=mirror, offline=args.offline, records=records)
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
    if args.group in ("all", "data"):
        extract_observations(root / "downloads/released_data.zip", root / "observations", args.scenes)
    save(root / f"downloads-{args.group}.json", {"manifest_sha256": sha256(HERE / "assets.lock.json"),
         "group": args.group, "status": "verified", "scenes": args.scenes, "offline": args.offline,
         "assets": records})


if __name__ == "__main__":
    main()
