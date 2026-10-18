import argparse
import json
import logging
import os
import os.path as osp
import re
import shutil
import sys
from collections import defaultdict

import imageio
import numpy as np

SCRIPT_DIR = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, osp.abspath(osp.join(SCRIPT_DIR, "..", "..", "..", "src")))
PROJECT_ROOT = osp.abspath(osp.join(SCRIPT_DIR, "..", "..", "..", "vendor", "mosca"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from lib_prior.sam3_wrapper import generate_sam3_epi, save_sam3_epi_artifacts


SCENE_PROMPTS = {
    "apple": ["apple", "arm"],
    "block": ["gray block", "block"],
    "paper-windmill": ["pinwheel"],
    "space-out": ["person", "human"],
    "spin": ["person", "human"],
    "teddy": ["teddy", "hand"],
    "wheel": ["wheel", "hand", "arm"],
}


def parse_args():
    parser = argparse.ArgumentParser(
        "Extract SAM3 dynamic masks for DyCheck test videos"
    )
    parser.add_argument("--rgb_transport", choices=("legacy", "ffv1"), default=None,
                        help="Override scene transport (paper-windmill defaults to lossless FFV1)")
    parser.add_argument(
        "--data_root",
        type=str,
        default="data_extracted/released_data/dycheck/iphone",
        help="DyCheck iPhone dataset root",
    )
    parser.add_argument(
        "--scene_list",
        nargs="*",
        default=sorted(SCENE_PROMPTS.keys()),
        help="Scenes to process",
    )
    parser.add_argument(
        "--output_dirname",
        type=str,
        default="test_sam3_dynamic_masks",
        help="Output directory name under each scene root",
    )
    parser.add_argument(
        "--prompt_override",
        type=str,
        default=None,
        help="Optional text prompt override used for every requested scene",
    )
    parser.add_argument(
        "--prompt_override_list",
        nargs="+",
        default=None,
        help="Optional ordered list of prompts tried and unioned for every scene",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default=None,
        help='GPU ids passed to SAM3, e.g. "0" or "0,1"',
    )
    parser.add_argument(
        "--sam3_checkpoint",
        type=str,
        default=None,
        help="Optional SAM3 checkpoint path",
    )
    parser.add_argument(
        "--sam3_bpe",
        type=str,
        default=None,
        help="Optional SAM3 BPE/tokenizer path",
    )
    parser.add_argument(
        "--sam3_lib_prior_root",
        type=str,
        default=None,
        help="Optional SoM lib_prior root containing the SAM3 package",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=10.0,
        help="Temporary RGB video fps for SAM3 inference",
    )
    parser.add_argument(
        "--track_prefix_sep",
        type=str,
        default="_",
        help='Filename separator for test track prefixes, default "_"',
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete any existing output directory for a scene before rerunning",
    )
    return parser.parse_args()


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def dedupe_keep_order(items):
    seen = set()
    deduped = []
    for item in items:
        if item not in seen:
            deduped.append(item)
            seen.add(item)
    return deduped


def resolve_prompt_list(scene_name, args):
    if args.prompt_override is not None and args.prompt_override_list is not None:
        raise ValueError("Use only one of --prompt_override and --prompt_override_list")

    prompt_list = None
    if args.prompt_override_list is not None:
        prompt_list = list(args.prompt_override_list)
    elif args.prompt_override is not None:
        prompt_list = [args.prompt_override]
    else:
        prompt_list = SCENE_PROMPTS.get(scene_name)

    if prompt_list is None:
        raise KeyError(
            f"No SAM3 prompt configured for scene {scene_name}. "
            f"Known scenes: {sorted(SCENE_PROMPTS.keys())}"
        )
    if isinstance(prompt_list, str):
        prompt_list = [prompt_list]

    prompt_list = dedupe_keep_order(
        [prompt.strip() for prompt in prompt_list if prompt and prompt.strip()]
    )
    if len(prompt_list) == 0:
        raise ValueError(f"Resolved empty prompt list for scene {scene_name}")
    return prompt_list


def prompt_slug(prompt):
    slug = re.sub(r"[^a-z0-9]+", "_", prompt.lower()).strip("_")
    return slug or "prompt"


def summarize_mask_list(mask_list):
    nonzero_counts = [int(np.count_nonzero(mask)) for mask in mask_list]
    return {
        "seed_nonzero": nonzero_counts[0] if nonzero_counts else 0,
        "nonzero_frames": int(sum(count > 0 for count in nonzero_counts)),
        "total_nonzero": int(sum(nonzero_counts)),
        "max_nonzero": int(max(nonzero_counts)) if nonzero_counts else 0,
    }


def build_zero_mask_list(img_list):
    return [np.zeros(np.asarray(img).shape[:2], dtype=np.uint8) for img in img_list]


def union_mask_lists(mask_lists, img_list):
    merged = build_zero_mask_list(img_list)
    for mask_list in mask_lists:
        if len(mask_list) != len(merged):
            raise ValueError(
                f"Mask list length mismatch: {len(mask_list)} vs {len(merged)}"
            )
        for frame_idx, mask in enumerate(mask_list):
            merged[frame_idx] = np.maximum(
                merged[frame_idx], np.asarray(mask).astype(np.uint8)
            )
    return merged


def load_track_images(test_img_dir, track_fns):
    img_list = []
    for fn in track_fns:
        img = imageio.imread(osp.join(test_img_dir, fn))
        img_list.append(np.asarray(img)[..., :3])
    return img_list


def group_test_tracks(test_img_dir, track_prefix_sep="_"):
    fn_list = sorted(
        [
            fn
            for fn in os.listdir(test_img_dir)
            if fn.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
    )
    grouped = defaultdict(list)
    for fn in fn_list:
        prefix = fn.split(track_prefix_sep, 1)[0]
        grouped[prefix].append(fn)
    return dict(sorted(grouped.items()))


def merge_track_outputs(track_dir, merged_mask_dir, merged_npy_dir):
    track_mask_dir = osp.join(track_dir, "masks")
    track_npy_dir = osp.join(track_dir, "epi", "error")
    if osp.isdir(track_mask_dir):
        for fn in sorted(os.listdir(track_mask_dir)):
            if fn.endswith(".png"):
                shutil.copy2(
                    osp.join(track_mask_dir, fn),
                    osp.join(merged_mask_dir, fn),
                )
    if osp.isdir(track_npy_dir):
        for fn in sorted(os.listdir(track_npy_dir)):
            if fn.endswith(".npy"):
                shutil.copy2(
                    osp.join(track_npy_dir, fn),
                    osp.join(merged_npy_dir, fn),
                )


def write_manifest(scene_output_dir, prompt_list, track_to_files, track_run_info):
    manifest_path = osp.join(scene_output_dir, "manifest.txt")
    with open(manifest_path, "w", encoding="utf-8") as handle:
        handle.write(f"prompt_list: {prompt_list}\n")
        handle.write(f"num_tracks: {len(track_to_files)}\n")
        handle.write("\n")
        for track_id, track_fns in track_to_files.items():
            handle.write(f"[track {track_id}] {len(track_fns)} frames\n")
            track_info = track_run_info.get(track_id, {})
            merged_stats = track_info.get("merged_stats")
            if merged_stats is not None:
                handle.write(
                    "merged_stats: "
                    f"seed_nonzero={merged_stats['seed_nonzero']} "
                    f"nonzero_frames={merged_stats['nonzero_frames']} "
                    f"total_nonzero={merged_stats['total_nonzero']} "
                    f"max_nonzero={merged_stats['max_nonzero']}\n"
                )
            for prompt_result in track_info.get("prompt_results", []):
                handle.write(
                    "prompt_result: "
                    f"prompt={prompt_result['prompt']!r} "
                    f"used_in_union={prompt_result['used_in_union']} "
                    f"seed_nonzero={prompt_result['seed_nonzero']} "
                    f"nonzero_frames={prompt_result['nonzero_frames']} "
                    f"total_nonzero={prompt_result['total_nonzero']} "
                    f"max_nonzero={prompt_result['max_nonzero']}\n"
                )
            handle.write("\n")
            for fn in track_fns:
                handle.write(f"{fn}\n")
            handle.write("\n")


def run_track_prompt_union(scene_name, track_id, track_dir, track_fns, img_list, prompt_list, args):
    transport = getattr(args, "rgb_transport", None)
    if transport is None:
        with open(osp.join(SCRIPT_DIR, "..", "configs", "mask_transport.json")) as handle:
            transport = json.load(handle).get(scene_name, "legacy")
    if transport not in ("legacy", "ffv1"):
        raise ValueError(f"Unknown RGB transport: {transport}")
    generate = generate_sam3_epi
    if transport == "ffv1":
        from wfm.dycheck_masks import generate_lossless_sam3_epi
        generate = generate_lossless_sam3_epi
    prompt_root = osp.join(track_dir, "prompt_runs")
    ensure_dir(prompt_root)
    successful_mask_lists = []
    prompt_results = []

    for prompt_idx, prompt in enumerate(prompt_list):
        prompt_dir = osp.join(
            prompt_root, f"{prompt_idx:02d}_{prompt_slug(prompt)}"
        )
        if args.force and osp.isdir(prompt_dir):
            shutil.rmtree(prompt_dir)
        ensure_dir(prompt_dir)
        logging.info(
            "Scene %s track %s: prompt %d/%d -> %r",
            scene_name,
            track_id,
            prompt_idx + 1,
            len(prompt_list),
            prompt,
        )
        mask_list = generate(
            save_dir=prompt_dir,
            img_name_list=track_fns,
            img_list=img_list,
            text_prompt=prompt,
            gpus=args.gpus,
            checkpoint_path=args.sam3_checkpoint,
            bpe_path=args.sam3_bpe,
            sam3_lib_prior_root=args.sam3_lib_prior_root,
            fps=args.fps,
            save_mask_png=True,
        )
        stats = summarize_mask_list(mask_list)
        used_in_union = stats["nonzero_frames"] > 0
        prompt_results.append(
            {
                "prompt": prompt,
                "used_in_union": used_in_union,
                **stats,
            }
        )
        if used_in_union:
            successful_mask_lists.append(mask_list)
        else:
            logging.warning(
                "Scene %s track %s: prompt %r produced empty masks; trying next prompt",
                scene_name,
                track_id,
                prompt,
            )

    if successful_mask_lists:
        merged_mask_list = union_mask_lists(successful_mask_lists, img_list)
    else:
        logging.warning(
            "Scene %s track %s: all prompts failed; writing all-zero merged mask",
            scene_name,
            track_id,
        )
        merged_mask_list = build_zero_mask_list(img_list)

    save_sam3_epi_artifacts(
        save_dir=track_dir,
        img_name_list=track_fns,
        img_list=img_list,
        mask_list=merged_mask_list,
        fps=args.fps,
        save_mask_png=True,
    )
    return prompt_results, summarize_mask_list(merged_mask_list)


def process_scene(scene_dir, scene_name, args):
    prompt_list = resolve_prompt_list(scene_name, args)
    test_img_dir = osp.join(scene_dir, "test_images")
    if not osp.isdir(test_img_dir):
        raise FileNotFoundError(f"Missing test_images directory: {test_img_dir}")

    scene_output_dir = osp.join(scene_dir, args.output_dirname)
    if args.force and osp.isdir(scene_output_dir):
        shutil.rmtree(scene_output_dir)
    ensure_dir(scene_output_dir)
    merged_mask_dir = osp.join(scene_output_dir, "masks")
    merged_npy_dir = osp.join(scene_output_dir, "epi", "error")
    track_root = osp.join(scene_output_dir, "tracks")
    ensure_dir(merged_mask_dir)
    ensure_dir(merged_npy_dir)
    ensure_dir(track_root)

    track_to_files = group_test_tracks(test_img_dir, args.track_prefix_sep)
    if len(track_to_files) == 0:
        raise RuntimeError(f"No test images found under {test_img_dir}")

    logging.info(
        "Scene %s: running SAM3 with prompts %s on %d tracks",
        scene_name,
        prompt_list,
        len(track_to_files),
    )
    track_run_info = {}
    for track_id, track_fns in track_to_files.items():
        track_dir = osp.join(track_root, f"track_{track_id}")
        if args.force and osp.isdir(track_dir):
            shutil.rmtree(track_dir)
        ensure_dir(track_dir)
        img_list = load_track_images(test_img_dir, track_fns)
        logging.info(
            "Scene %s track %s: %d frames",
            scene_name,
            track_id,
            len(track_fns),
        )
        prompt_results, merged_stats = run_track_prompt_union(
            scene_name=scene_name,
            track_id=track_id,
            track_dir=track_dir,
            track_fns=track_fns,
            img_list=img_list,
            prompt_list=prompt_list,
            args=args,
        )
        merge_track_outputs(track_dir, merged_mask_dir, merged_npy_dir)
        track_run_info[track_id] = {
            "prompt_results": prompt_results,
            "merged_stats": merged_stats,
        }
    write_manifest(scene_output_dir, prompt_list, track_to_files, track_run_info)
    logging.info("Finished scene %s -> %s", scene_name, scene_output_dir)


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    data_root = osp.abspath(args.data_root)
    if not osp.isdir(data_root):
        raise FileNotFoundError(f"DyCheck data root not found: {data_root}")

    for scene_name in args.scene_list:
        scene_dir = osp.join(data_root, scene_name)
        if not osp.isdir(scene_dir):
            raise FileNotFoundError(f"Scene directory not found: {scene_dir}")
        process_scene(scene_dir, scene_name, args)


if __name__ == "__main__":
    main()
