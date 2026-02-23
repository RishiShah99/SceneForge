from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parent.parent

# This function loads the configuration from a YAML file
def load_config(config_path: Path) -> dict:
    return yaml.safe_load(config_path.read_text())

# This function checks if the COLMAP executable exists at the specified path in the config
def find_colmap(config: dict) -> Path:
    exe = ROOT / config["colmap"]["exe"]
    if not exe.exists():
        raise FileNotFoundError(
            f"COLMAP not found at {exe}\n"
            "Check configs/default.yaml → colmap.exe path."
        )
    return exe

# This function runs a subprocess command and prints the label for the step.
def run(cmd: list[str], label: str) -> None:
    print(f"\n{'='*60}")
    print(f"[COLMAP] {label}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"\n[ERROR] COLMAP step '{label}' failed (exit code {result.returncode})")
        sys.exit(result.returncode)

# This function parses the cameras.txt file from COLMAP output and returns a dict of camera parameters
def run_colmap_pipeline(frames_dir: Path, out_dir: Path, colmap_exe: Path, camera_model: str = "SIMPLE_RADIAL", use_gpu: bool = False,) -> Path:
    db_path = out_dir / "database.db"
    
    # COLMAP outputs a "sparse/0" folder with the model files; we will convert them to text format in-place
    sparse_dir = out_dir / "sparse"
    text_dir = out_dir / "sparse" / "0"

    out_dir.mkdir(parents=True, exist_ok=True)
    sparse_dir.mkdir(parents=True, exist_ok=True)

    gpu_index = "0" if use_gpu else "-1"
    sift_gpu = "1" if use_gpu else "0"

    colmap = str(colmap_exe)

    # Feature extraction
    run([
        colmap, "feature_extractor",
        "--database_path", str(db_path),
        "--image_path", str(frames_dir),
        "--ImageReader.camera_model", camera_model,
        "--ImageReader.single_camera", "1",   # assume one camera for all frames
        "--SiftExtraction.use_gpu", sift_gpu,
        "--SiftExtraction.gpu_index", gpu_index,
    ], "Feature extraction")

    # Feature matching (sequential for video; exhaustive for unordered images)
    run([
        colmap, "sequential_matcher",
        "--database_path", str(db_path),
        "--SiftMatching.use_gpu", sift_gpu,
        "--SiftMatching.gpu_index", gpu_index,
        "--SequentialMatching.overlap", "10",   # match each frame to 10 neighbours
        "--SequentialMatching.loop_detection", "0",
    ], "Sequential matching")

    # Sparse Reconstruction (mapper)
    run([
        colmap, "mapper",
        "--database_path", str(db_path),
        "--image_path", str(frames_dir),
        "--output_path", str(sparse_dir),
    ], "Sparse mapper (SfM)")

    # Convert binary format to text format for easier parsing in Python
    binary_model = sparse_dir / "0"
    text_model = sparse_dir / "0"  

    run([
        colmap, "model_converter",
        "--input_path", str(binary_model),
        "--output_path", str(text_model),
        "--output_type", "TXT",
    ], "Model converter (binary → TXT)")

    print(f"\n[run_colmap] Done. Text model at: {text_model}")
    return text_model


# Working with CLI
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run COLMAP SfM on extracted frames.")
    parser.add_argument("--frames", type=Path, required=True,
                        help="Folder containing extracted .jpg frames")
    parser.add_argument("--out", type=Path, required=True,
                        help="Output folder for COLMAP project files")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    colmap_exe = find_colmap(cfg)

    run_colmap_pipeline(
        frames_dir=args.frames,
        out_dir=args.out,
        colmap_exe=colmap_exe,
        camera_model=cfg["colmap"]["camera_model"],
        use_gpu=cfg["colmap"]["use_gpu"],
    )
