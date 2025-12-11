"""
Utility helpers for LIDAR and image processing used by the project.
"""
import os
from typing import List, Tuple, Dict, Optional
import numpy as np
import pandas as pd
import cv2
from sklearn.cluster import KMeans
from PIL import Image
from pathlib import Path
import json

# --- small utils ---
def to_numpy(x):
    """Convert object (torch tensor, list) to 1D numpy array"""
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy().reshape(-1)
    except Exception:
        pass
    if isinstance(x, (list, tuple)):
        return np.asarray(x).reshape(-1)
    return np.asarray(x).reshape(-1)

def l2_normalize(mat, axis=1, eps=1e-12):
    m = np.asarray(mat, dtype=np.float32)
    if m.ndim == 1:
        return m / (np.linalg.norm(m) + eps)
    n = np.linalg.norm(m, axis=axis, keepdims=True) + eps
    return m / n

# --- lidar helpers ---
def infer_angle_type(angle_arr: np.ndarray) -> str:
    """Return 'deg' if angles appear degrees, else 'rad'."""
    if np.nanmax(np.abs(angle_arr)) > 2*np.pi + 1e-3:
        return "deg"
    return "rad"

def polar_to_cartesian(distances: np.ndarray, angles: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return x,y arrays. angles may be degrees or radians (auto-detected)."""
    angle_unit = infer_angle_type(angles)
    if angle_unit == "deg":
        angles = np.deg2rad(angles)
    # Using standard robot coords: x = r * cos(theta), y = r * sin(theta) (forward is x)
    x = distances * np.cos(angles)
    y = distances * np.sin(angles)
    return x, y

def project_points_homography(x: np.ndarray, y: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Project 2D points (x,y) using 3x3 homography H to pixel coordinates (u,v)."""
    pts = np.vstack((x, y, np.ones_like(x))).T
    proj = (H @ pts.T).T
    w = proj[:, 2:3]
    uv = proj[:, :2] / w
    return uv.astype(int)

# --- kmeans centroids per-scan ---
def compute_kmeans_centroids_per_scan(lidar_df: pd.DataFrame, k: int = 10, ids: Optional[List[int]] = None) -> pd.DataFrame:
    """
    Given lidar_df with fields ['id', 'angle', 'distance', 'img_front'], returns centroid DataFrame:
    columns: ['ids', 'img_front', 'centroid_x', 'centroid_y']
    """
    df_res = pd.DataFrame(columns=["ids", "img_front", "centroid_x", "centroid_y"])
    if ids is None:
        ids = sorted(lidar_df["id"].unique().tolist())
    for frame_id in ids:
        ssc = lidar_df[lidar_df["id"]==frame_id]
        if ssc.empty:
            continue
        # filter front angles (optional heuristic)
        angle_arr = ssc["angle"].values
        if infer_angle_type(angle_arr) == "deg":
            angles_rad = np.deg2rad(angle_arr)
        else:
            angles_rad = angle_arr
        # mask in front field of view (~ +/- 80 deg)
        front_mask = (angles_rad <= 4*np.pi/9) | (angles_rad >= 2*np.pi - 4*np.pi/9)
        filtered_distances = np.abs(ssc["distance"].values[front_mask])
        filtered_angles = angles_rad[front_mask]
        if len(filtered_distances) < k:
            continue
        x = filtered_distances * np.cos(filtered_angles)
        y = filtered_distances * np.sin(filtered_angles)
        X = np.column_stack([x, y])
        try:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
            kmeans.fit(X)
            centers = kmeans.cluster_centers_
            for cx, cy in centers:
                df_res = pd.concat([df_res, pd.DataFrame({"ids":[frame_id], "img_front":[ssc["img_front"].values[0]], "centroid_x":[cx], "centroid_y":[cy]})], ignore_index=True)
        except Exception as e:
            print(f"KMeans error on frame {frame_id}: {e}")
            continue
    return df_res

# --- mask helpers (SAM) optional ---
def get_masks_for_image(image_rgb: np.ndarray, sam_model=None, mask_generator=None):
    """
    Returns a list of boolean masks for given image.
    - If a mask_generator is provided (from segment-anything), it will be used.
    - Else fallback to an empty list or alternative.
    """
    if mask_generator is not None:
        sam_result = mask_generator.generate(image_rgb)
        masks = [m["segmentation"] for m in sorted(sam_result, key=lambda x: x["area"], reverse=True)]
        return masks
    else:
        print("WARNING: No SAM generator provided. Returning empty list.")
        return []

# --- matching points to masks ---
def associate_points_to_mask(projected_uv: np.ndarray, image_to_lidar_ids: np.ndarray, masks: List[np.ndarray], mask_paths: List[str]=None):
    """
    Args:
    - projected_uv: (N,2) ints -> x=col (u), y=row (v)
    - image_to_lidar_ids: original lidar ids for each point (N,)
    - masks: list of boolean 2D arrays where True indicates the region
    - mask_paths: optional list of filenames (same order as masks)

    Returns:
    - region_to_lidar_ids: list of dicts {'region_id': id, 'mask_path':..., 'lidar_indices': [ids]}
    """
    H, W = masks[0].shape if masks else (None, None)
    out = []
    for i, mask in enumerate(masks):
        if mask_paths:
            mask_path = mask_paths[i]
            region_id = Path(mask_path).stem
        else:
            mask_path = None
            region_id = str(i)
        lidar_ids_in_region = []
        for idx, (u,v) in enumerate(projected_uv):
            u = int(u); v = int(v)
            if u<0 or v<0 or mask is None or v>=mask.shape[0] or u>=mask.shape[1]:
                continue
            if mask[v, u] > 0:
                lidar_ids_in_region.append(int(image_to_lidar_ids[idx]))
        out.append({"region_id": region_id, "mask_path": mask_path, "lidar_cluster_indices": lidar_ids_in_region})
    return out

# --- build label map ---
def build_label_map(region_df: pd.DataFrame, background_id: int=255):
    """
    region_df expected to contain columns: ['region' (mask path), 'pred_class', 'pred_score']
    Returns:
    - label_map (H, W) with integer class ids
    - class2id dict
    - palette dict (RGB tuples)
    """
    assert len(region_df) > 0
    m0 = cv2.imread(region_df.iloc[0]["region"], cv2.IMREAD_GRAYSCALE)
    H, W = m0.shape
    label_map = np.full((H, W), background_id, dtype=np.int32)
    score_map = np.full((H, W), -np.inf, dtype=np.float32)

    # build palette and class->id mapping
    classes = sorted(set(region_df["pred_class"].tolist()))
    class2id = {c: i for i, c in enumerate(classes)}
    base_palette = {
        "wall": (139, 69, 19),
        "person": (255, 255, 0),
        "car": (255, 20, 147),
        "tree": (34, 139, 34),
        "sky": (135, 206, 235),
        "object": (128, 0, 128),
    }
    palette = {}
    import numpy as np
    for c in class2id:
        if c in base_palette:
            palette[c] = base_palette[c]
        else:
            rng = np.random.RandomState(abs(hash(c)) % (2**32))
            palette[c] = tuple(int(x) for x in rng.randint(0,256,3))

    for _, row in region_df.iterrows():
        mask = cv2.imread(row["region"], cv2.IMREAD_GRAYSCALE) > 0
        cid = class2id[row["pred_class"]]
        sc = float(row["pred_score"]) if "pred_score" in row else 1.0
        upd = mask & (sc > score_map)
        label_map[upd] = cid
        score_map[upd] = sc

    return label_map, class2id, palette

# --- json helpers ---
def save_final_dataset(final_dataset, out_path):
    with open(out_path, "w") as f:
        json.dump(final_dataset, f, indent=2)
    print(f"Final dataset saved to {out_path}")

