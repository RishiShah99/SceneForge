import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def run(poses_file: Path, out_file: Path):
    data = json.loads(poses_file.read_text())
    poses = data.get("poses", [])
    if not poses:
        raise RuntimeError("No poses found in trajectory file")

    xs = [p["position"][0] for p in poses]
    ys = [p["position"][1] for p in poses]
    zs = [p["position"][2] for p in poses]

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(xs, ys, zs, marker="o", markersize=2)
    ax.set_title("Camera Trajectory")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.tight_layout()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, dpi=180)
    print(f"Saved trajectory plot: {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--poses", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.poses, args.out)
