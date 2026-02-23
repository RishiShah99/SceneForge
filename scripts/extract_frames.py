import argparse
from pathlib import Path
import cv2


def extract_frames(video_path: Path, out_dir: Path, target_fps: float) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(int(round(src_fps / target_fps)), 1)

    saved = 0
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0:
            out_path = out_dir / f"frame_{saved:05d}.jpg"
            cv2.imwrite(str(out_path), frame)
            saved += 1
        idx += 1

    cap.release()
    print(f"Video FPS: {src_fps:.2f}")
    print(f"Input frames: {frame_count}")
    print(f"Saved frames: {saved}")
    print(f"Output dir: {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=10.0)
    args = parser.parse_args()
    extract_frames(args.video, args.out, args.fps)
