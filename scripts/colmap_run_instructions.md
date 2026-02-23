# COLMAP Run Instructions (Phase 1)

SceneForge uses COLMAP as the **pose estimation / SfM backend** for the offline v1 pipeline.

## Install
- Download the prebuilt COLMAP binary for your OS from the official releases.
- Add COLMAP to PATH or note the executable path.

## Basic flow (GUI or CLI)
1. Extract frames to `data/frames/<scene_name>/`
2. Run COLMAP feature extraction + matching + mapping
3. Export sparse model (`cameras`, `images`, `points3D`) to text format
4. Parse exported poses into `trajectory.json`

## Suggested output layout
`data/outputs/<scene_name>/colmap/`
- database.db
- sparse/0/ (binary or text model)
- sparse_txt/ (exported text model)

## Notes
- COLMAP may not register all frames. That's okay.
- Keep the first pass simple: just get poses and a sparse cloud.
