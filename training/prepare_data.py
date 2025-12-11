"""
Prepare data script: project LiDAR points into image space and associate them with segmentation masks.
"""
import argparse
import os
import sys
from glob import glob
import json
import numpy as np
import pandas as pd
from pathlib import Path
import cv2
from utils.utils import (polar_to_cartesian, project_points_homography, build_label_map, associate_points_to_mask, save_final_dataset)

def infer_angle_type(angle_arr):
    return "deg" if np.nanmax(np.abs(angle_arr)) > 2*np.pi + 1e-3 else "rad"

def process_dataset(lidar_csv_path: str, masks_root_dir: str, out_json: str, H: np.ndarray = None):
    """
    Prepare dataset JSON mapping LiDAR point ids to segmented image regions.
    """
    if H is None:
        # default virtual homography from notebook (calibration)
        scale_x = -30
        scale_y = -30
        center_u = 400
        center_v = 580
        H = np.array([[scale_x, 0, center_u],
                      [0, scale_y, center_v],
                      [0, 0, 1]], dtype=np.float32)

    lidar_df = pd.read_csv(lidar_csv_path)
    # images with masks
    mask_files = glob(os.path.join(masks_root_dir, "*", "*", "*.png")) + glob(os.path.join(masks_root_dir, "*", "*.png"))
    # Group masks by the original image path
    image_to_masks = {}
    for f in mask_files:
        basename = os.path.basename(f)
        try:
            image_key = None
            if "_mask_" in basename:
                image_name = basename.split("_mask_")[0] + "_.png"
                image_key = f"images/front/{image_name}"
            else:
                parent = Path(f).parent
                image_key = None
            if image_key is None:
                image_path_candidates = glob(os.path.join("images", "front", "image_*.png"))
                if image_path_candidates:
                    image_key = image_path_candidates[0]
                else:
                    image_key = "images/front/image_0_.png"
        except Exception as e:
            print(f"Could not parse filename for {f}; error: {e}")
            continue
        if image_key not in image_to_masks:
            image_to_masks[image_key] = []
        image_to_masks[image_key].append(f)

    final_dataset = []

    if len(image_to_masks) == 0:
        print("No masks found in", masks_root_dir)
    else:
        # infer image size from first mask
        first_mask = cv2.imread(list(image_to_masks.values())[0][0], cv2.IMREAD_GRAYSCALE)
        if first_mask is None:
            raise RuntimeError("Could not read mask to infer image shape.")
        img_height, img_width = first_mask.shape

    for img_path, mask_paths in image_to_masks.items():
        # filter LiDAR points that belong to this image
        img_lidar_df = lidar_df[lidar_df['img_front'] == img_path].copy()
        if img_lidar_df.empty:
            print(f"No LiDAR points for {img_path} -> skipping")
            continue
        print(f"Processing: {img_path} -> {len(img_lidar_df)} LiDAR points, {len(mask_paths)} masks")
        # Convert polar to cartesian
        distances = np.abs(img_lidar_df['distance'].values)
        angles = img_lidar_df['angle'].values
        # convert to radians if needed
        if infer_angle_type(angles) == "deg":
            angles = np.deg2rad(angles)
        x = distances * np.cos(angles)
        y = distances * np.sin(angles)
        uv = project_points_homography(x, y, H)
        lidar_ids = img_lidar_df['id'].values
        # read masks (binary)
        masks = [cv2.imread(p, cv2.IMREAD_GRAYSCALE) > 0 for p in mask_paths]
        # associate points
        regions_info = associate_points_to_mask(uv, lidar_ids, masks, mask_paths)
        scene_dict = {
            "scene_id": Path(img_path).stem.replace(".png", ""),
            "image_file_path": img_path,
            "lidar_file_path": str(lidar_csv_path),
            "regions": []
        }
        for r in regions_info:
            scene_dict["regions"].append({
                "region_id": r["region_id"],
                "mask_path": r["mask_path"],
                "lidar_cluster_indices": r["lidar_cluster_indices"]
            })
        final_dataset.append(scene_dict)

    save_final_dataset(final_dataset, out_json)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Prepare dataset: project LiDAR points and match with masks")
    parser.add_argument("--lidar-csv", type=str, default="dataset/lidar_data.csv")
    parser.add_argument("--masks-dir", type=str, default="masked_regions")
    parser.add_argument("--out-json", type=str, default="final_processed_dataset.json")
    args = parser.parse_args()
    process_dataset(args.lidar_csv, args.masks_dir, args.out_json)
