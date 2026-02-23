from __future__ import annotations
import json
import re
from pathlib import Path
import numpy as np


# Camera intrinsics
def parse_cameras(cameras_txt: Path) -> dict[int, dict]:
    cameras = {}
    with open(cameras_txt) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            cam_id = int(parts[0])
            model = parts[1]
            width, height = int(parts[2]), int(parts[3])
            params = [float(x) for x in parts[4:]]
            cameras[cam_id] = {
                "model": model,
                "width": width,
                "height": height,
                "params": params,
            }
    return cameras

# This function extracts fx, fy, cx, cy from a COLMAP camera dict based on its model type.
def intrinsics_from_camera(cam: dict) -> dict:
    model = cam["model"]
    p = cam["params"]
    if model == "SIMPLE_PINHOLE":
        fx = fy = p[0]; cx = p[1]; cy = p[2]
    elif model == "PINHOLE":
        fx, fy, cx, cy = p[0], p[1], p[2], p[3]
    elif model == "SIMPLE_RADIAL":
        fx = fy = p[0]; cx = p[1]; cy = p[2]  # p[3] is radial distortion k1
    elif model == "RADIAL":
        fx = fy = p[0]; cx = p[1]; cy = p[2]
    elif model in ("OPENCV", "FULL_OPENCV"):
        fx, fy, cx, cy = p[0], p[1], p[2], p[3]
    else:
        # Fallback: assume first param is focal, second and third are cx/cy
        fx = fy = p[0]; cx = p[1] if len(p) > 1 else cam["width"] / 2
        cy = p[2] if len(p) > 2 else cam["height"] / 2
    return {"fx": fx, "fy": fy, "cx": cx, "cy": cy,
            "width": cam["width"], "height": cam["height"]}


# Image Poses
def quat_to_rotation_matrix(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    q = np.array([qw, qx, qy, qz], dtype=np.float64)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    R = np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y)],
        [    2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x)],
        [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y)],
    ], dtype=np.float64)
    return R

# This function parses the images.txt file from COLMAP output and returns a list of pose dicts with extrinsics and intrinsics.
def parse_images(images_txt: Path) -> list[dict]:
    poses = []
    with open(images_txt) as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]

    # images.txt has pairs of lines per image: pose line, then point2D line
    i = 0
    while i < len(lines):
        pose_line = lines[i]
        i += 2 

        parts = pose_line.split()
        image_id = int(parts[0])
        qw, qx, qy, qz = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        tx, ty, tz = float(parts[5]), float(parts[6]), float(parts[7])
        camera_id = int(parts[8])
        image_name = parts[9]

        R = quat_to_rotation_matrix(qw, qx, qy, qz)
        t = np.array([tx, ty, tz])
        position = (-R.T @ t).tolist()
        poses.append({
            "image_id": image_id,
            "image_name": image_name,
            "camera_id": camera_id,
            "qw": qw, "qx": qx, "qy": qy, "qz": qz,
            "tx": tx, "ty": ty, "tz": tz,
            "position": position,
        })

    # Sort by image name so trajectory order matches frame order
    poses.sort(key=lambda p: p["image_name"])
    return poses


# Main Export
def colmap_to_trajectory(sparse_dir: Path,output_json: Path) -> dict:
    cameras_txt = sparse_dir / "cameras.txt"
    images_txt = sparse_dir / "images.txt"

    if not cameras_txt.exists() or not images_txt.exists():
        raise FileNotFoundError(
            f"Expected cameras.txt and images.txt in {sparse_dir}\n"
            "Make sure COLMAP ran model_converter with --output_type TXT"
        )

    cameras = parse_cameras(cameras_txt)
    poses = parse_images(images_txt)

    # Attach intrinsics to each pose
    for pose in poses:
        cam = cameras.get(pose["camera_id"], {})
        pose["intrinsics"] = intrinsics_from_camera(cam) if cam else {}

    # Build summary
    positions = np.array([p["position"] for p in poses])
    if len(positions) > 1:
        span = float(np.linalg.norm(positions.max(axis=0) - positions.min(axis=0)))
        total_path = float(sum(
            np.linalg.norm(np.array(poses[i+1]["position"]) - np.array(poses[i]["position"]))
            for i in range(len(poses) - 1)
        ))
    else:
        span = 0.0
        total_path = 0.0

    trajectory = {
        "num_registered": len(poses),
        "num_cameras": len(cameras),
        "scene_span_m": round(span, 4),
        "total_path_m": round(total_path, 4),
        "poses": poses,
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(trajectory, indent=2))
    print(f"[pose_estimator] Registered {len(poses)} images → {output_json}")
    return trajectory
