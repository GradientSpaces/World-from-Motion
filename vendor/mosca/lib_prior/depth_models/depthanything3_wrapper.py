import sys, os, os.path as osp
import json

THIS_DIR = osp.abspath(osp.dirname(__file__))
DA3_SRC_ROOT = osp.join(THIS_DIR, "Depth-Anything-3", "src")
for path in (THIS_DIR, DA3_SRC_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

import numpy as np
import torch
import cv2
from tqdm import tqdm
import imageio

from depth_utils import (
    viz_depth_list,
    save_depth_list,
    check_trajectory_quality,
    compute_max_baseline,
)
from depth_anything_3.api import DepthAnything3


def select_reference_camera(pose_file, all_camera_ids, frame_indices):
    """
    Select the camera with largest max baseline as reference for scale alignment.
    
    Args:
        pose_file: Path to camera_extrinsics.json
        all_camera_ids: List of all camera IDs (e.g., ["cam01", "cam02", ...])
        frame_indices: List of frame indices to load poses for
        
    Returns:
        tuple: (best_camera_id, best_max_baseline, all_baselines_dict)
            - best_camera_id: Camera ID with largest baseline
            - best_max_baseline: The max baseline value
            - all_baselines_dict: Dict mapping camera_id -> max_baseline
    """
    baselines = {}
    
    for cam_id in all_camera_ids:
        extrinsics = load_poses_from_json(pose_file, cam_id, frame_indices)
        if extrinsics is not None:
            baselines[cam_id] = compute_max_baseline(extrinsics)
        else:
            print(f"  Warning: Could not load poses for {cam_id}")
    
    if not baselines:
        return None, 0.0, {}
    
    best_cam = max(baselines, key=baselines.get)
    return best_cam, baselines[best_cam], baselines


def get_depthanything3_model(device="cuda", weights="depth-anything/DA3NESTED-GIANT-LARGE-1.1"):
    device = torch.device(device)
    model = DepthAnything3.from_pretrained(weights)
    model = model.to(device=device)
    return model


def depthanything3_process_folder(
    model,
    img_list,
    fn_list,
    dst,
    invalid_mask_list=None,
    extrinsics=None,
    intrinsics=None,
):
    """
    Process a folder of images using the DepthAnything3 model.
    
    Outputs in preproc-compatible format:
    - {dst}/depthanything3_disp/{img_name}.npy - disparity (1/depth)
    - {dst}/depthanything3_intrins.json - intrinsics {img_name: [fx, fy, cx, cy]}
    - {dst}/depthanything3_recon.npy - DROID-SLAM format dict with traj_c2w
    
    Args:
        model: The initialized wrapper model.
        img_list: List of numpy images (H, W, 3).
        fn_list: List of filenames corresponding to images.
        dst: Destination directory to save results.
        invalid_mask_list: Optional list of masks to apply to depth.
        extrinsics: Optional (N, 4, 4) c2w matrices for scale alignment.
        intrinsics: Optional (N, 3, 3) K matrices for scale alignment.
    """
    print("Running DepthAnythingV3...")
    
    img_list = np.asarray(img_list)
    T, H, W, C = img_list.shape

    # Reshape to 14-divisible size for ViT
    new_H = H // 14 * 14
    new_W = W // 14 * 14
    working_img_list = []
    for img in img_list:
        img = cv2.resize(img.copy(), (new_W, new_H))
        working_img_list.append(img)
    
    # Scale intrinsics to working resolution if provided
    working_intrinsics = None
    if intrinsics is not None:
        scale_x = new_W / W
        scale_y = new_H / H
        working_intrinsics = intrinsics.copy()
        working_intrinsics[:, 0, 0] *= scale_x  # fx
        working_intrinsics[:, 1, 1] *= scale_y  # fy
        working_intrinsics[:, 0, 2] *= scale_x  # cx
        working_intrinsics[:, 1, 2] *= scale_y  # cy
    
    # Check if we should align to input poses
    align_to_input = False
    use_extrinsics = None  # Extrinsics to pass to model (None if alignment not possible)
    if extrinsics is not None and intrinsics is not None:
        # Check trajectory quality before deciding to align
        quality = check_trajectory_quality(extrinsics)
        print(f"\nTrajectory quality check: {quality['reason']}")
        if quality['stats']:
            print(f"  - Effective rank: {quality['stats']['effective_rank']}")
            print(f"  - Max baseline: {quality['stats']['max_baseline']:.4f}")
            print(f"  - Baseline ratio: {quality['stats']['baseline_ratio']:.4f}")
        
        if quality['can_align']:
            align_to_input = True
            use_extrinsics = extrinsics  # Only pass extrinsics when alignment is possible
            print("-> Enabling scale alignment to input poses")
        else:
            # Don't pass extrinsics to avoid alignment crash on degenerate trajectories
            use_extrinsics = None
            print("-> Disabling scale alignment (using DA3's own scale)")
    
    prediction = model.inference(
        image=working_img_list,
        extrinsics=use_extrinsics,
        intrinsics=working_intrinsics if use_extrinsics is not None else None,
        align_to_input_ext_scale=align_to_input,  # Use DA3's aligned predictions for consistent visualization
        # export_dir="./debug",
        # export_format="mini_npz-glb"
    )
    
    # Resize depth back to original resolution
    dep_list = []
    for dep in prediction.depth:
        dep = cv2.resize(dep, (W, H), interpolation=cv2.INTER_NEAREST_EXACT)
        dep_list.append(dep)
    dep_list = np.asarray(dep_list)

    # === Output paths (preproc-compatible) ===
    depth_folder = osp.join(dst, "depthanything3_depth")
    disp_folder = osp.join(dst, "depthanything3_disp")
    intrins_path = osp.join(dst, "depthanything3_intrins.json")
    recon_path = osp.join(dst, "depthanything3_recon.npy")
    depth_viz_path = osp.join(dst, "depth_anything3.mp4")

    # Save metric depth in the layout consumed by MoSca's dep_mode=depthanything3.
    print(f"Saving depth to {depth_folder}...")
    save_depth_list(dep_list.copy(), fn_list, depth_folder, invalid_mask_list)
    
    # === Save disparity as .npy files (preproc format) ===
    os.makedirs(disp_folder, exist_ok=True)
    print(f"Saving disparity to {disp_folder}...")
    for i, dep in enumerate(dep_list):
        if invalid_mask_list is not None:
            dep = dep.copy()
            dep[invalid_mask_list[i] > 0] = 0
        # Convert depth to disparity
        disp = 1.0 / np.clip(dep, a_min=1e-6, a_max=1e6)
        # Get image name without extension
        img_name = osp.splitext(fn_list[i])[0]
        np.save(osp.join(disp_folder, f"{img_name}.npy"), disp)
    
    # Save depth visualization
    viz_depth_list(dep_list, depth_viz_path)

    # === Save intrinsics as JSON (preproc format) ===
    intrins_dict = {}
    # Use input intrinsics if provided, otherwise DA3's predicted intrinsics
    if intrinsics is not None:
        # Input intrinsics are already at original resolution
        for i, fn in enumerate(fn_list):
            img_name = osp.splitext(fn)[0]
            K = intrinsics[i]
            intrins_dict[img_name] = [float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])]
        print(f"Saving input intrinsics to {intrins_path}...")
        with open(intrins_path, "w") as f:
            json.dump(intrins_dict, f, indent=1)
    elif prediction.intrinsics is not None:
        pred_intrinsics = prediction.intrinsics
        # Scale intrinsics to original resolution
        if len(prediction.depth.shape) == 3:
            pred_H, pred_W = prediction.depth.shape[1], prediction.depth.shape[2]
        else:
            pred_H, pred_W = prediction.depth.shape[-2], prediction.depth.shape[-1]
            
        scale_x = W / pred_W
        scale_y = H / pred_H
        
        for i, fn in enumerate(fn_list):
            img_name = osp.splitext(fn)[0]
            K = pred_intrinsics[i]
            fx = float(K[0, 0] * scale_x)
            fy = float(K[1, 1] * scale_y)
            cx = float(K[0, 2] * scale_x)
            cy = float(K[1, 2] * scale_y)
            intrins_dict[img_name] = [fx, fy, cx, cy]
        
        print(f"Saving DA3 intrinsics to {intrins_path}...")
        with open(intrins_path, "w") as f:
            json.dump(intrins_dict, f, indent=1)

    # === Save poses in DROID-SLAM format ===
    # Prefer GT extrinsics if provided, otherwise use DA3's predicted extrinsics
    if extrinsics is not None:
        # GT extrinsics are W2C - convert to C2W
        c2w = np.linalg.inv(extrinsics).astype(np.float32)
        print("Using GT extrinsics for droid_recon.npy")
    elif prediction.extrinsics is not None:
        # DepthAnything3 outputs W2C (OpenCV/COLMAP convention)
        w2c = prediction.extrinsics  # (N, 3, 4) or (N, 4, 4)
        
        # Ensure 4x4 format
        N = w2c.shape[0]
        if w2c.shape[-2] == 3:
            w2c_4x4 = np.zeros((N, 4, 4), dtype=np.float32)
            w2c_4x4[:, :3, :] = w2c
            w2c_4x4[:, 3, 3] = 1.0
        else:
            w2c_4x4 = w2c.astype(np.float32)
        
        # Invert W2C to get C2W (what DROID-SLAM outputs as traj_c2w)
        c2w = np.linalg.inv(w2c_4x4)
        print("Using DA3 predicted extrinsics (no GT provided)")
    else:
        c2w = None
    
    if c2w is not None:
        # Use input intrinsics if provided (already at original resolution)
        # Otherwise use DA3's predicted intrinsics scaled to original resolution
        if intrinsics is not None:
            K_avg = intrinsics.mean(axis=0)
            fx = float(K_avg[0, 0])
            fy = float(K_avg[1, 1])
            cx = float(K_avg[0, 2])
            cy = float(K_avg[1, 2])
            intrinsics_arr = np.array([fx, fy, cx, cy], dtype=np.float32)
        elif prediction.intrinsics is not None:
            # Scale DA3's intrinsics from prediction resolution to original resolution
            if len(prediction.depth.shape) == 3:
                pred_H, pred_W = prediction.depth.shape[1], prediction.depth.shape[2]
            else:
                pred_H, pred_W = prediction.depth.shape[-2], prediction.depth.shape[-1]
            scale_x = W / pred_W
            scale_y = H / pred_H
            K_avg = prediction.intrinsics.mean(axis=0)
            fx = float(K_avg[0, 0] * scale_x)
            fy = float(K_avg[1, 1] * scale_y)
            cx = float(K_avg[0, 2] * scale_x)
            cy = float(K_avg[1, 2] * scale_y)
            intrinsics_arr = np.array([fx, fy, cx, cy], dtype=np.float32)
        else:
            intrinsics_arr = np.array([W, H, W/2, H/2], dtype=np.float32)
        
        # Create DROID-SLAM compatible dict
        recon_dict = {
            "tstamps": np.arange(T, dtype=np.float32),
            "traj_c2w": c2w.astype(np.float32),
            "intrinsics": intrinsics_arr,
            "img_shape": np.array([H, W]),
        }
        
        print(f"Saving poses to {recon_path}...")
        np.save(recon_path, np.array(recon_dict))

    return


def depthanything3_process_with_cross_camera_alignment(
    model,
    img_list,
    fn_list,
    dst,
    extrinsics,
    intrinsics,
    pose_file,
    all_camera_ids,
    input_camera_id,
    video_dir,
    sample_stride=4,
    invalid_mask_list=None,
):
    """
    Process images with cross-camera scale alignment.
    
    When the input camera has insufficient motion for reliable scale alignment,
    this function uses a reference camera with better motion to establish scale.
    
    Args:
        model: The initialized DA3 model.
        img_list: List of numpy images (H, W, 3) for the input camera.
        fn_list: List of filenames corresponding to images.
        dst: Destination directory to save results.
        extrinsics: (N, 4, 4) w2c matrices for the input camera.
        intrinsics: (N, 3, 3) K matrices for the input camera.
        pose_file: Path to camera_extrinsics.json containing all camera poses.
        all_camera_ids: List of all camera IDs in the scene.
        input_camera_id: Camera ID of the input camera.
        video_dir: Directory containing all camera videos.
        sample_stride: Frame sampling stride for joint inference (default: 4).
        invalid_mask_list: Optional list of masks to apply to depth.
    """
    print("\n" + "="*60)
    print("Cross-Camera Scale Alignment")
    print("="*60)
    
    img_list = np.asarray(img_list)
    T, H, W, C = img_list.shape
    
    # Get frame indices from filenames
    frame_indices = []
    for fn in fn_list:
        base = osp.splitext(fn)[0]
        try:
            frame_indices.append(int(base))
        except ValueError:
            frame_indices.append(len(frame_indices))
    
    # Step 1: Compute max baseline for all cameras and select reference
    print(f"\nComputing baselines for {len(all_camera_ids)} cameras...")
    best_cam, best_baseline, all_baselines = select_reference_camera(
        pose_file, all_camera_ids, frame_indices
    )
    
    if best_cam is None:
        print("ERROR: Could not compute baselines for any camera. Falling back to regular processing.")
        depthanything3_process_folder(
            model, img_list, fn_list, dst,
            invalid_mask_list=invalid_mask_list,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
        )
        return
    
    # Print baseline info
    print("\nCamera baselines:")
    for cam_id, baseline in sorted(all_baselines.items(), key=lambda x: -x[1]):
        marker = " <-- INPUT" if cam_id == input_camera_id else ""
        marker += " <-- BEST" if cam_id == best_cam else ""
        print(f"  {cam_id}: {baseline:.4f}{marker}")
    
    input_baseline = all_baselines.get(input_camera_id, 0.0)
    
    # Step 2: Check if input camera is the best (or close enough)
    if input_camera_id == best_cam:
        print(f"\nInput camera {input_camera_id} has the best baseline. Using regular alignment.")
        depthanything3_process_folder(
            model, img_list, fn_list, dst,
            invalid_mask_list=invalid_mask_list,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
        )
        return
    
    print(f"\nInput camera {input_camera_id} baseline ({input_baseline:.4f}) < best camera {best_cam} ({best_baseline:.4f})")
    print("Performing cross-camera scale alignment...")
    
    # Step 3: Load reference camera video and poses
    ref_video_path = osp.join(video_dir, f"{best_cam}.mp4")
    if not osp.exists(ref_video_path):
        print(f"ERROR: Reference camera video not found: {ref_video_path}")
        print("Falling back to regular processing (unaligned scale).")
        depthanything3_process_folder(
            model, img_list, fn_list, dst,
            invalid_mask_list=invalid_mask_list,
            extrinsics=None,  # Skip alignment
            intrinsics=intrinsics,
        )
        return
    
    print(f"\nLoading reference camera video: {ref_video_path}")
    ref_images, ref_frame_indices = load_video_frames(ref_video_path, stride=1, max_frames=len(frame_indices))
    
    # Ensure same number of frames
    if len(ref_images) != len(img_list):
        min_frames = min(len(ref_images), len(img_list))
        print(f"Warning: Frame count mismatch. Using first {min_frames} frames.")
        ref_images = ref_images[:min_frames]
        ref_frame_indices = ref_frame_indices[:min_frames]
        img_list = img_list[:min_frames]
        fn_list = fn_list[:min_frames]
        frame_indices = frame_indices[:min_frames]
        if extrinsics is not None:
            extrinsics = extrinsics[:min_frames]
        if intrinsics is not None:
            intrinsics = intrinsics[:min_frames]
    
    # Load reference camera poses
    ref_extrinsics = load_poses_from_json(pose_file, best_cam, frame_indices)
    if ref_extrinsics is None:
        print("ERROR: Could not load reference camera poses. Falling back to regular processing.")
        depthanything3_process_folder(
            model, img_list, fn_list, dst,
            invalid_mask_list=invalid_mask_list,
            extrinsics=None,
            intrinsics=intrinsics,
        )
        return
    
    # Step 4: Sample frames for joint inference
    sampled_indices = list(range(0, len(img_list), sample_stride))
    n_sampled = len(sampled_indices)
    print(f"\nSampling every {sample_stride}th frame: {n_sampled} frames from each camera")
    
    # Prepare sampled data
    input_sampled_imgs = img_list[sampled_indices]
    input_sampled_ext = extrinsics[sampled_indices] if extrinsics is not None else None
    input_sampled_int = intrinsics[sampled_indices] if intrinsics is not None else None
    
    ref_sampled_imgs = ref_images[sampled_indices]
    ref_sampled_ext = ref_extrinsics[sampled_indices]
    ref_sampled_int = intrinsics[sampled_indices] if intrinsics is not None else None  # Same intrinsics
    
    # Combine for joint inference: [input_sampled, ref_sampled]
    joint_imgs = np.concatenate([input_sampled_imgs, ref_sampled_imgs], axis=0)
    joint_ext = np.concatenate([input_sampled_ext, ref_sampled_ext], axis=0) if input_sampled_ext is not None else None
    joint_int = np.concatenate([input_sampled_int, ref_sampled_int], axis=0) if input_sampled_int is not None else None
    
    print(f"Joint inference: {len(joint_imgs)} total frames ({n_sampled} input + {n_sampled} reference)")
    
    # Reshape to 14-divisible size for ViT
    new_H = H // 14 * 14
    new_W = W // 14 * 14
    
    # Prepare joint working images
    joint_working_imgs = []
    for img in joint_imgs:
        img_resized = cv2.resize(img.copy(), (new_W, new_H))
        joint_working_imgs.append(img_resized)
    
    # Prepare reference-only working images
    ref_working_imgs = []
    for img in ref_sampled_imgs:
        img_resized = cv2.resize(img.copy(), (new_W, new_H))
        ref_working_imgs.append(img_resized)
    
    # Scale intrinsics for reference camera
    ref_working_int = None
    if ref_sampled_int is not None:
        scale_x = new_W / W
        scale_y = new_H / H
        ref_working_int = ref_sampled_int.copy()
        ref_working_int[:, 0, 0] *= scale_x
        ref_working_int[:, 1, 1] *= scale_y
        ref_working_int[:, 0, 2] *= scale_x
        ref_working_int[:, 1, 2] *= scale_y
    
    # Check reference trajectory quality on FULL trajectory (not sampled)
    # We already selected it as the best camera, so this is just a sanity check
    ref_quality = check_trajectory_quality(ref_extrinsics)
    print(f"Reference full trajectory quality: {ref_quality['reason']}")
    
    if not ref_quality['can_align']:
        # This shouldn't happen since we selected the camera with best baseline
        # But log a warning and continue anyway - the user requested no fallback
        print("WARNING: Reference camera trajectory has quality issues, but proceeding anyway.")
    
    # Step 5: Run JOINT inference WITHOUT alignment
    # This ensures input and reference have consistent scale
    # DA3 will predict both depths AND poses for all images
    print("\n--- Joint Inference (input + reference, no alignment) ---")
    
    joint_prediction = model.inference(
        image=joint_working_imgs,
        extrinsics=None,  # No GT poses - let DA3 predict
        intrinsics=None,
        align_to_input_ext_scale=False,
    )
    
    # Extract depths from joint prediction
    # First n_sampled frames are input, next n_sampled are reference
    joint_input_depth = []
    for i in range(n_sampled):
        dep = cv2.resize(joint_prediction.depth[i], (W, H), interpolation=cv2.INTER_NEAREST_EXACT)
        joint_input_depth.append(dep)
    joint_input_depth = np.stack(joint_input_depth)
    
    joint_ref_depth = []
    for i in range(n_sampled, 2 * n_sampled):
        dep = cv2.resize(joint_prediction.depth[i], (W, H), interpolation=cv2.INTER_NEAREST_EXACT)
        joint_ref_depth.append(dep)
    joint_ref_depth = np.stack(joint_ref_depth)
    
    print(f"Joint inference complete. Got {n_sampled} input + {n_sampled} reference depth maps (consistent scale).")
    
    # Step 6: Align DA3's predicted poses (for reference camera) to GT poses using Umeyama
    # This gives us the scale factor to convert from DA3 scale to GT scale
    print("\n--- Pose Alignment (DA3 predicted -> GT) ---")
    
    # Import the alignment function from DA3
    from depth_anything_3.utils.pose_align import align_poses_umeyama
    
    # Extract DA3's predicted poses for reference camera (last n_sampled poses)
    if joint_prediction.extrinsics is not None:
        pred_ref_ext = joint_prediction.extrinsics[n_sampled:2*n_sampled]  # (n_sampled, 3, 4) or (n_sampled, 4, 4)
        
        # Ensure 4x4 format
        if pred_ref_ext.shape[1] == 3:
            pred_ref_ext_4x4 = np.zeros((n_sampled, 4, 4), dtype=np.float32)
            pred_ref_ext_4x4[:, :3, :] = pred_ref_ext
            pred_ref_ext_4x4[:, 3, 3] = 1.0
            pred_ref_ext = pred_ref_ext_4x4
        
        print(f"DA3 predicted {len(pred_ref_ext)} poses for reference camera")
        print(f"GT reference poses shape: {ref_sampled_ext.shape}")
        
        try:
            # Align predicted reference poses to GT reference poses
            # Returns rotation, translation, scale
            _, _, scale = align_poses_umeyama(
                ref_sampled_ext,  # GT poses (reference)
                pred_ref_ext,     # DA3 predicted poses (to be aligned)
                return_aligned=False,
                ransac=len(pred_ref_ext) >= 10,
                random_state=42,
            )
            print(f"Umeyama alignment scale: {scale:.4f}")
        except Exception as e:
            print(f"WARNING: Umeyama alignment failed: {e}")
            print("Using scale=1.0")
            scale = 1.0
    else:
        print("WARNING: DA3 did not return predicted poses. Using scale=1.0")
        scale = 1.0
    
    # Step 7: Apply scale to joint depths
    # Umeyama: aligned = R @ (s * predicted) + t
    # So s scales predicted translations UP to GT scale when s > 1
    # For depth: GT_depth = s * DA3_depth
    joint_input_depth_scaled = joint_input_depth * scale
    joint_ref_depth_scaled = joint_ref_depth * scale
    print(f"Applied scale {scale:.4f} to {n_sampled} sampled frames from each camera.")
    
    # Step 8: Run full input camera inference WITHOUT alignment
    print("\n--- Full Input Camera Inference ---")
    
    single_working_imgs = []
    for img in img_list:
        img_resized = cv2.resize(img.copy(), (new_W, new_H))
        single_working_imgs.append(img_resized)
    
    single_prediction = model.inference(
        image=single_working_imgs,
        extrinsics=None,
        intrinsics=None,
        align_to_input_ext_scale=False,
    )
    
    # Resize to original resolution
    single_depth = []
    for dep in single_prediction.depth:
        dep = cv2.resize(dep, (W, H), interpolation=cv2.INTER_NEAREST_EXACT)
        single_depth.append(dep)
    single_depth = np.stack(single_depth)
    print(f"Full input inference complete. Got {len(single_depth)} depth maps.")
    
    # Step 9: Scale transfer from joint to full using overlapping sampled frames
    print("\n--- Scale Transfer (joint -> full) ---")
    
    # Compare joint_input_depth_scaled (GT scale) vs single_depth at sampled indices
    single_sampled_depth = single_depth[sampled_indices]
    
    transfer_scale_factors = []
    for i in range(n_sampled):
        scaled_dep = joint_input_depth_scaled[i]
        single_dep = single_sampled_depth[i]
        
        valid_mask = (scaled_dep > 1e-3) & (single_dep > 1e-3)
        if valid_mask.sum() > 100:
            s = np.median(scaled_dep[valid_mask] / single_dep[valid_mask])
            transfer_scale_factors.append(s)
    
    if len(transfer_scale_factors) == 0:
        print("WARNING: Could not compute transfer scale. Using scale=1.0")
        transfer_scale = 1.0
    else:
        transfer_scale = np.median(transfer_scale_factors)
        print(f"Transfer scale factors: min={min(transfer_scale_factors):.4f}, max={max(transfer_scale_factors):.4f}, median={transfer_scale:.4f}")
    
    # Apply transfer scale to full input depths
    single_depth = single_depth * transfer_scale
    print(f"Applied transfer scale {transfer_scale:.4f} to all {len(single_depth)} input frames.")
    
    # single_depth is now in GT scale
    aligned_depth = single_depth
    
    # Step 10: Save results (same format as depthanything3_process_folder)
    print("\n--- Saving Results ---")
    
    disp_folder = osp.join(dst, "depthanything3_disp")
    intrins_path = osp.join(dst, "depthanything3_intrins.json")
    recon_path = osp.join(dst, "depthanything3_recon.npy")
    depth_viz_path = osp.join(dst, "depth_anything3.mp4")
    
    # Save disparity
    os.makedirs(disp_folder, exist_ok=True)
    print(f"Saving disparity to {disp_folder}...")
    for i, dep in enumerate(aligned_depth):
        if invalid_mask_list is not None and i < len(invalid_mask_list):
            dep = dep.copy()
            dep[invalid_mask_list[i] > 0] = 0
        disp = 1.0 / np.clip(dep, a_min=1e-6, a_max=1e6)
        img_name = osp.splitext(fn_list[i])[0]
        np.save(osp.join(disp_folder, f"{img_name}.npy"), disp)
    
    # Save depth visualization
    viz_depth_list(list(aligned_depth), depth_viz_path)
    
    # Save intrinsics
    if intrinsics is not None:
        intrins_dict = {}
        for i, fn in enumerate(fn_list):
            img_name = osp.splitext(fn)[0]
            K = intrinsics[i]
            intrins_dict[img_name] = [float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])]
        print(f"Saving intrinsics to {intrins_path}...")
        with open(intrins_path, "w") as f:
            json.dump(intrins_dict, f, indent=1)
    
    # Save poses (use input camera's GT poses since depth is now aligned to GT scale)
    if extrinsics is not None:
        # Convert w2c to c2w
        c2w = np.linalg.inv(extrinsics).astype(np.float32)
        
        if intrinsics is not None:
            K_avg = intrinsics.mean(axis=0)
            intrinsics_arr = np.array([
                float(K_avg[0, 0]), float(K_avg[1, 1]),
                float(K_avg[0, 2]), float(K_avg[1, 2])
            ], dtype=np.float32)
        else:
            intrinsics_arr = np.array([W, H, W/2, H/2], dtype=np.float32)
        
        recon_dict = {
            "tstamps": np.arange(len(fn_list), dtype=np.float32),
            "traj_c2w": c2w,
            "intrinsics": intrinsics_arr,
            "img_shape": np.array([H, W]),
        }
        print(f"Saving poses to {recon_path}...")
        np.save(recon_path, np.array(recon_dict))
    
    # Save cross-camera alignment info
    alignment_info = {
        "input_camera": input_camera_id,
        "reference_camera": best_cam,
        "input_baseline": float(input_baseline),
        "reference_baseline": float(best_baseline),
        "umeyama_pose_scale": float(scale),  # Scale from aligning DA3 predicted poses to GT poses
        "transfer_scale": float(transfer_scale),  # Scale from sampled to full inference
        "sample_stride": sample_stride,
        "n_sampled_frames": n_sampled,
        "all_baselines": {k: float(v) for k, v in all_baselines.items()},
    }
    alignment_info_path = osp.join(dst, "cross_camera_alignment.json")
    with open(alignment_info_path, "w") as f:
        json.dump(alignment_info, f, indent=2)
    print(f"Saved alignment info to {alignment_info_path}")
    
    print("\nCross-camera scale alignment complete!")
    return


def load_poses_from_json(pose_file, camera_id, frame_indices):
    """
    Load camera poses from a JSON file.
    
    Args:
        pose_file: Path to camera_extrinsics.json
        camera_id: Camera ID (e.g., "cam05")
        frame_indices: List of frame indices to load
        
    Returns:
        extrinsics: (N, 4, 4) w2c (world-to-camera) matrices for DA3, or None if file doesn't exist
    """
    if not osp.exists(pose_file):
        print(f"Pose file not found: {pose_file}")
        return None
    
    with open(pose_file, "r") as f:
        pose_data = json.load(f)
    
    def parse_matrix_string(mat_str):
        """Parse a matrix string like '[r00 r01 r02 0] [r10 r11 r12 0] ...' into 4x4 array."""
        rows = []
        for row_match in mat_str.strip().split(']'):
            row_match = row_match.strip()
            if row_match.startswith('['):
                row_match = row_match[1:]
            if row_match:
                values = [float(x) for x in row_match.split()]
                if len(values) == 4:
                    rows.append(values)
        return np.array(rows, dtype=np.float32)
    
    traj_unreal = []
    for idx in frame_indices:
        frame_key = f"frame{idx}"
        if frame_key not in pose_data:
            print(f"Warning: {frame_key} not found in pose file")
            return None
        if camera_id not in pose_data[frame_key]:
            print(f"Warning: {camera_id} not found in {frame_key}")
            return None
        
        mat_str = pose_data[frame_key][camera_id]
        mat = parse_matrix_string(mat_str)
        
        traj_unreal.append(mat)
    
    # convert from unreal to opencv
    # The Unreal matrix after transpose is c2w (camera-to-world)
    # DA3 expects w2c (world-to-camera) extrinsics
    # 
    # Unreal: X=Forward, Y=Right, Z=Up (camera axes)
    # OpenCV: X=Right, Y=Down, Z=Forward (camera axes)
    # 
    # We need to convert both camera axes (columns) AND world axes (rows+translation)
    traj = np.stack(traj_unreal).transpose(0, 2, 1)
    
    c2ws = []
    for mat in traj:
        # Camera axis reordering (columns): Unreal [Fwd,Right,Up] -> OpenCV [Right,Down,Fwd]
        mat = mat[:, [1, 2, 0, 3]]
        mat[:3, 1] *= -1.  # Flip Y (Up -> Down)
        
        # World axis reordering (rows): same transform for consistency
        mat = mat[[1, 2, 0, 3], :]
        mat[1, :] *= -1.  # Flip Y row
        
        mat[:3, 3] /= 100  # cm to m
        c2ws.append(mat)
    c2ws = np.stack(c2ws)
    
    # Convert to w2c (don't normalize - DA3 does its own internal normalization)
    w2cs = np.linalg.inv(c2ws).astype(np.float32)

    print(f"Loaded {len(w2cs)} poses (w2c) for {camera_id}")
    return w2cs


def load_video_frames(video_file, stride=1, max_frames=None):
    """Load frames from a video file."""
    cap = cv2.VideoCapture(video_file)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if max_frames is not None:
        total_frames = min(total_frames, max_frames)
    
    images, frame_indices = [], []
    for i in range(0, total_frames, stride):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            break
        images.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frame_indices.append(i)
    
    cap.release()
    return np.stack(images, axis=0), frame_indices


def depthanything3_process_video(
    model,
    video_file,
    dst,
    stride=1,
    max_frames=None,
    pose_file=None,
    intrinsics_str=None,
    camera_id=None,
):
    """
    Process a video with DA3. DA3 predicts depth and poses.
    
    Args:
        model: The initialized DA3 model.
        video_file: Path to video file
        dst: Destination directory to save results.
        stride: Frame stride (default: 1)
        max_frames: Maximum number of frames to load
        pose_file: Optional path to camera_extrinsics.json for pose alignment
        intrinsics_str: Optional intrinsics as "fx,fy,cx,cy" string
        camera_id: Optional camera ID for pose lookup (inferred from video filename if not provided)
    """
    # Extract camera_id from video filename if not provided (e.g., "cam05.mp4" -> "cam05")
    if camera_id is None:
        camera_id = osp.splitext(osp.basename(video_file))[0]
    
    # Load poses first if pose_file is provided
    extrinsics = None
    if pose_file is not None:
        # First, determine frame indices we'll load
        cap = cv2.VideoCapture(video_file)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        if max_frames is not None:
            total_frames = min(total_frames, max_frames)
        frame_indices = list(range(0, total_frames, stride))
        
        extrinsics = load_poses_from_json(pose_file, camera_id, frame_indices)
    
    print(f"Loading video: {video_file}")
    images, frame_indices = load_video_frames(video_file, stride, max_frames)
    fn_list = [f"{idx:05d}.png" for idx in frame_indices]
    
    # Parse intrinsics string to (N, 3, 3) K matrices
    intrinsics = None
    if intrinsics_str is not None:
        fx, fy, cx, cy = [float(x) for x in intrinsics_str.split(",")]
        K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float32)
        intrinsics = np.tile(K[None], (len(images), 1, 1))
        print(f"Using intrinsics: fx={fx}, fy={fy}, cx={cx}, cy={cy}")
    
    depthanything3_process_folder(
        model=model,
        img_list=images,
        fn_list=fn_list,
        dst=dst,
        extrinsics=extrinsics,
        intrinsics=intrinsics,
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="video", choices=["folder", "video"])
    # Folder mode args
    parser.add_argument("--src", type=str, default="./data/iphone/paper-windmill/images")
    parser.add_argument("--dst", type=str, default="./debug")
    # Video mode args
    parser.add_argument("--video_file", type=str, default="data/multisync/scene2/videos/cam05.mp4")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max_frames", type=int, default=None)
    # Pose alignment args (optional)
    parser.add_argument("--pose_file", type=str, default="data/multisync/scene2/cameras/camera_extrinsics.json",
                        help="Path to camera_extrinsics.json for pose alignment")
    parser.add_argument("--intrinsics", type=str, default="969.70, 969.70, 640, 640",
                        help="Camera intrinsics as 'fx,fy,cx,cy'")
    parser.add_argument("--camera_id", type=str, default="cam05",
                        help="Camera ID for pose lookup (e.g., 'cam05')")
    
    args = parser.parse_args()
    
    device = "cuda"
    model = get_depthanything3_model(device=device)
    
    if args.mode == "video":
        if args.video_file is None:
            raise ValueError("--video_file required for video mode")
        depthanything3_process_video(
            model=model,
            video_file=args.video_file,
            dst=args.dst,
            stride=args.stride,
            max_frames=args.max_frames,
            pose_file=args.pose_file,
            intrinsics_str=args.intrinsics,
            camera_id=args.camera_id,
        )
    else:
        src = osp.expanduser(args.src)
        fns = sorted(os.listdir(src))
        img_list, fn_list = [], []
        for fn in fns:
            if fn.endswith(".jpg") or fn.endswith(".png"):
                img = imageio.imread(osp.join(src, fn))
                img_list.append(img)
                fn_list.append(fn)
        depthanything3_process_folder(model, img_list, fn_list, dst=args.dst)