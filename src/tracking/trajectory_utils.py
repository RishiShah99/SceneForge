from __future__ import annotations
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Stats based on camera positions in the trajectory, and jump detection for tracking instability.
def trajectory_stats(poses: list[dict]) -> dict:
    if not poses:
        return {}
    positions = np.array([p["position"] for p in poses])
    diffs = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    return {
        "num_poses": len(poses),
        "total_path_m": float(diffs.sum()),
        "mean_step_m": float(diffs.mean()) if len(diffs) else 0.0,
        "max_step_m": float(diffs.max()) if len(diffs) else 0.0,
        "bbox_min": positions.min(axis=0).tolist(),
        "bbox_max": positions.max(axis=0).tolist(),
        "scene_span_m": float(np.linalg.norm(positions.max(axis=0) - positions.min(axis=0))),
    }

# This function detects "jumps" in the trajectory where the step distance exceeds a threshold based on the median step.
def detect_jumps(poses: list[dict], threshold_multiplier: float = 5.0) -> list[int]:
    if len(poses) < 2:
        return []
    positions = np.array([p["position"] for p in poses])
    diffs = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    median = np.median(diffs)
    threshold = median * threshold_multiplier
    return [i + 1 for i, d in enumerate(diffs) if d > threshold]

# Plotting
def plot_trajectory_3d(poses: list[dict], out_file: Path, title: str = "Camera Trajectory") -> None:
    if not poses:
        raise ValueError("No poses to plot")

    positions = np.array([p["position"] for p in poses])
    xs, ys, zs = positions[:, 0], positions[:, 1], positions[:, 2]
    jumps = detect_jumps(poses)

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    # Main trajectory line
    ax.plot(xs, ys, zs, color="#3366cc", linewidth=1.5, alpha=0.8, label="path")
    ax.scatter(xs[0], ys[0], zs[0], color="green", s=60, zorder=5, label="start")
    ax.scatter(xs[-1], ys[-1], zs[-1], color="red", s=60, zorder=5, label="end")

    # Mark jumps
    if jumps:
        jx = positions[jumps, 0]
        jy = positions[jumps, 1]
        jz = positions[jumps, 2]
        ax.scatter(jx, jy, jz, color="orange", s=80, marker="x", zorder=6,
                   label=f"jumps ({len(jumps)})")

    ax.set_title(title)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, dpi=180)
    plt.close()
    print(f"[trajectory_utils] Saved 3D plot → {out_file}")

# This function saves a 2D top-down view of the camera path by plotting X vs Z positions and marking the start, end, and jump points.
def plot_trajectory_topdown(poses: list[dict], out_file: Path, title: str = "Camera Path (Top-Down)") -> None:
    """Save a 2D top-down (X/Z plane) view of the camera path."""
    positions = np.array([p["position"] for p in poses])
    xs, zs = positions[:, 0], positions[:, 2]

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(xs, zs, color="#3366cc", linewidth=1.5, alpha=0.8)
    ax.scatter(xs[0], zs[0], color="green", s=80, zorder=5, label="start")
    ax.scatter(xs[-1], zs[-1], color="red", s=80, zorder=5, label="end")

    # Draw frame index every N frames
    step = max(1, len(poses) // 20)
    for i in range(0, len(poses), step):
        ax.annotate(str(i), (xs[i], zs[i]), fontsize=6, color="gray")

    ax.set_title(title)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Z (m)")
    ax.set_aspect("equal")
    ax.legend()
    plt.tight_layout()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, dpi=180)
    plt.close()
    print(f"[trajectory_utils] Saved top-down plot → {out_file}")

def load_trajectory(json_path: Path) -> list[dict]:
    data = json.loads(json_path.read_text())
    return data.get("poses", data) if isinstance(data, dict) else data
