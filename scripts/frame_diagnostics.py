import argparse
import json
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import pandas as pd

# This function computes the variance of the Laplacian of the image, which is a common method for estimating blur.  
def laplacian_variance(gray):
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

# This function uses ORB (Oriented FAST and Rotated BRIEF) to detect keypoints in the image
def orb_feature_count(gray):
    orb = cv2.ORB_create(nfeatures=1000)
    kps = orb.detect(gray, None)
    return 0 if kps is None else len(kps)

# This function runs diagnostics on a directory of frames, computing blur and texture metrics, saving results to CSV and JSON
def run(frames_dir: Path, out_dir: Path):
    # Create output directory if it doesn't exist
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    # Find all image files in the frames directory
    frame_paths = sorted(frames_dir.glob("*.jpg")) + sorted(frames_dir.glob("*.png"))
    if not frame_paths:
        raise RuntimeError(f"No frames found in {frames_dir}")

    # Process each frame: read the image, convert to grayscale, compute blur and ORB features, and store results in rows
    for fp in frame_paths:
        img = cv2.imread(str(fp))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rows.append({
            "frame": fp.name,
            "width": int(img.shape[1]),
            "height": int(img.shape[0]),
            "blur_score": laplacian_variance(gray),
            "orb_features": orb_feature_count(gray),
        })

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "frame_metrics.csv", index=False)

    summary = {
        "num_frames": int(len(df)),
        "blur_mean": float(df["blur_score"].mean()),
        "blur_min": float(df["blur_score"].min()),
        "blur_max": float(df["blur_score"].max()),
        "orb_mean": float(df["orb_features"].mean()),
        "orb_min": int(df["orb_features"].min()),
        "orb_max": int(df["orb_features"].max()),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    plt.figure(figsize=(10, 4))
    plt.plot(df.index, df["blur_score"])
    plt.title("Blur Score (Laplacian Variance) per Frame")
    plt.xlabel("Frame Index")
    plt.ylabel("Blur Score")
    plt.tight_layout()
    plt.savefig(out_dir / "blur_timeline.png", dpi=150)
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(df.index, df["orb_features"])
    plt.title("ORB Feature Count per Frame")
    plt.xlabel("Frame Index")
    plt.ylabel("ORB Features")
    plt.tight_layout()
    plt.savefig(out_dir / "texture_timeline.png", dpi=150)
    plt.close()

    print(f"Saved diagnostics to {out_dir}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.frames, args.out)
