from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.extract_frames import extract_frames
from scripts.run_colmap import find_colmap, run_colmap_pipeline
from src.tracking.pose_estimator import colmap_to_trajectory
from src.tracking.trajectory_utils import (
    trajectory_stats,
    detect_jumps,
    plot_trajectory_3d,
    plot_trajectory_topdown,
)


BANNER = """
╔══════════════════════════════════════════╗
║         SceneForge — Stage 1             ║
║   Camera Trajectory from Video           ║
╚══════════════════════════════════════════╝
"""

# This function loads the config of the YAML file
def load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text())

# This function runs the diagnostics on the frames and returns a summary dict
def run_diagnostics(frames_dir: Path, out_dir: Path) -> dict:
    """Run frame_diagnostics inline and return the summary dict."""
    import cv2
    import pandas as pd

    rows = []
    frame_paths = sorted(frames_dir.glob("*.jpg")) + sorted(frames_dir.glob("*.png"))
    for fp in frame_paths:
        img = cv2.imread(str(fp))
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        orb = cv2.ORB_create(nfeatures=1000)
        kps = orb.detect(gray, None)
        features = len(kps) if kps else 0
        rows.append({"frame": fp.name, "blur_score": blur, "orb_features": features})

    df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "frame_metrics.csv", index=False)

    summary = {
        "num_frames": len(df),
        "blur_mean": float(df["blur_score"].mean()) if len(df) else 0,
        "blur_min": float(df["blur_score"].min()) if len(df) else 0,
        "orb_mean": float(df["orb_features"].mean()) if len(df) else 0,
        "orb_min": int(df["orb_features"].min()) if len(df) else 0,
    }
    (out_dir / "diagnostics_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[diagnostics] {summary['num_frames']} frames — "
          f"blur mean={summary['blur_mean']:.1f}, ORB mean={summary['orb_mean']:.0f}")
    return summary

# This function prints warnings based on the diagnostics summary and config thresholds
def print_quality_warnings(diag: dict, cfg: dict) -> None:
    warn = cfg["quality"]
    print("\n--- Frame Quality Warnings ---")
    if diag["blur_mean"] < warn["blur_warn_threshold"]:
        print(f"  ⚠  Low average blur score ({diag['blur_mean']:.1f} < {warn['blur_warn_threshold']}). "
              "Frames are blurry — COLMAP may struggle.")
    if diag["orb_mean"] < warn["texture_warn_threshold"]:
        print(f"  ⚠  Low average feature count ({diag['orb_mean']:.0f} < {warn['texture_warn_threshold']}). "
              "Low-texture scene — COLMAP may fail on some frames.")
    if diag["blur_mean"] >= warn["blur_warn_threshold"] and diag["orb_mean"] >= warn["texture_warn_threshold"]:
        print("  ✓  Frame quality looks acceptable for COLMAP.")

def main() -> None:
    parser = argparse.ArgumentParser(description="SceneForge Stage 1 — Camera Trajectory Pipeline")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--video", type=Path, help="Input video file (.mp4 etc.)")
    group.add_argument("--frames", type=Path, help="Pre-extracted frames folder (skip extraction)")
    parser.add_argument("--scene", type=str, required=True,
                        help="Scene name (used for output folder naming, e.g. 'south_building')")
    parser.add_argument("--fps", type=float, default=None,
                        help="Frames per second to extract (overrides config)")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "default.yaml")
    parser.add_argument("--skip-colmap", action="store_true",
                        help="Skip COLMAP if sparse/0/ already exists (re-use previous run)")
    args = parser.parse_args()

    print(BANNER)
    cfg = load_config(args.config)
    scene = args.scene

    frames_dir = ROOT / cfg["paths"]["frames"] / scene
    colmap_out = ROOT / cfg["paths"]["colmap_projects"] / scene
    outputs_dir = ROOT / cfg["paths"]["outputs"] / scene
    diag_dir = outputs_dir / "diagnostics"
    plots_dir = outputs_dir / "plots"

    # Extract frames
    print(f"\n[1/5] Frame extraction")
    if args.video:
        target_fps = args.fps or cfg["video"]["target_fps"]
        extract_frames(args.video, frames_dir, target_fps)
    else:
        frames_dir = args.frames
        n = len(list(frames_dir.glob("*.jpg"))) + len(list(frames_dir.glob("*.png")))
        print(f"      Using existing frames dir: {frames_dir} ({n} frames)")

    frame_count = len(list(frames_dir.glob("*.jpg"))) + len(list(frames_dir.glob("*.png")))
    if frame_count < 10:
        print(f"[ERROR] Only {frame_count} frames found — need at least 10 for COLMAP. "
              "Try a lower --fps or a longer video.")
        sys.exit(1)

    # Frame Diagnostics
    print(f"\n[2/5] Frame diagnostics")
    diag = run_diagnostics(frames_dir, diag_dir)
    print_quality_warnings(diag, cfg)

    # COLMAP
    print(f"\n[3/5] COLMAP SfM")
    sparse_txt_dir = colmap_out / "sparse" / "0"

    if args.skip_colmap and (sparse_txt_dir / "images.txt").exists():
        print(f"      Skipping — using existing model at {sparse_txt_dir}")
    else:
        colmap_exe = find_colmap(cfg)
        run_colmap_pipeline(
            frames_dir=frames_dir,
            out_dir=colmap_out,
            colmap_exe=colmap_exe,
            camera_model=cfg["colmap"]["camera_model"],
            use_gpu=cfg["colmap"]["use_gpu"],
        )

    # Check COLMAP output exists
    if not (sparse_txt_dir / "images.txt").exists():
        print("\n[ERROR] COLMAP did not produce images.txt. Reconstruction may have failed.")
        print("  Common causes:")
        print("  - Too few features (blurry / low-texture frames)")
        print("  - Frames not overlapping enough")
        print("  - Video too short or extracted at too low FPS")
        sys.exit(1)

    # Parse poses to trajectory
    print(f"\n[4/5] Parsing COLMAP output → trajectory.json")
    traj_json = outputs_dir / "trajectory.json"
    trajectory = colmap_to_trajectory(sparse_txt_dir, traj_json)
    poses = trajectory["poses"]

    # Plot the trajectory
    print(f"\n[5/5] Plotting trajectory")
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_trajectory_3d(poses, plots_dir / "trajectory_3d.png", title=f"Camera Trajectory — {scene}")
    plot_trajectory_topdown(poses, plots_dir / "trajectory_topdown.png", title=f"Top-Down Path — {scene}")

    stats = trajectory_stats(poses)
    jumps = detect_jumps(poses)

    print(f"""
╔══════════════════════════════════════════╗
║            Stage 1 Complete              ║
╚══════════════════════════════════════════╝

Scene:              {scene}
Frames extracted:   {frame_count}
Frames registered:  {stats.get('num_poses', 0)} / {frame_count} \
({100 * stats.get('num_poses', 0) / max(frame_count, 1):.1f}%)
Total path length:  {stats.get('total_path_m', 0):.3f} m
Scene span:         {stats.get('scene_span_m', 0):.3f} m
Tracking jumps:     {len(jumps)} detected

Output files:
  trajectory.json   → {traj_json}
  3D plot           → {plots_dir / 'trajectory_3d.png'}
  Top-down plot     → {plots_dir / 'trajectory_topdown.png'}
  Frame metrics     → {diag_dir / 'frame_metrics.csv'}
""")

    if len(jumps) > 0:
        print(f"Jump frames (tracking instability): {jumps[:10]}{'...' if len(jumps) > 10 else ''}")
    registered_ratio = stats.get("num_poses", 0) / max(frame_count, 1)
    if registered_ratio < 0.5:
        print(f"Only {registered_ratio*100:.0f}% of frames registered. "
              "Reconstruction is partial — check diagnostics.")
    elif registered_ratio < 0.8:
        print(f"{registered_ratio*100:.0f}% registration. Decent but not complete coverage.")
    else:
        print(f"{registered_ratio*100:.0f}% registration — solid reconstruction.")


if __name__ == "__main__":
    main()
