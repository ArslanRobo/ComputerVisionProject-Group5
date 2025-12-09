# 3D Scene Reconstruction and Virtual Tour (CS436)

Modular Python code for checkpoints 1–3 of the SfM pipeline plus the Phase 3 export for the tour viewer. The notebooks remain for reporting, but runnable, well-structured `.py` equivalents now live in the repo. All scripts consume pre-extracted frames from `data/frames` (no video decoding is required).

## Setup
1) Python 3.10+ recommended.  
2) Install dependencies:
```
pip install opencv-python-headless numpy scipy tqdm pillow piexif matplotlib open3d plotly
```
3) Place your frames in `data/frames/` (default) or pass `--image-dir` to the scripts. Images are downscaled to `max_dim` (default 1600) and intrinsics are scaled accordingly.

## Code Structure
- `checkpoint1.py`: feature matching demo (SIFT + Lowe ratio) across consecutive frames, with optional grids/visualizations.
- `checkpoint2.py`: two-view reconstruction (EXIF intrinsics, essential matrix, triangulation, outlier removal, optional PLY and projection plots).
- `checkpoint3.py`: full SfM + bundle adjustment + cleaning, visualization, and PLY outputs (faithful port of the checkpoint3 notebook logic).
- `sfm/pipeline.py`: library implementation of loading, calibration, matching, two-view init, incremental PnP/triangulation, BA, and JSON export.
- `Scripts/run_checkpoint1.py`: thin CLI wrapper around `SfMPipeline` (legacy runner kept for reference).
- `Tour/`: static Three.js viewer (index.html) plus `cameras.json` and `list.txt`; drop your sparse PLY and matching `cameras.json` here to preview the tour.

## Usage (from repo root)
```
# Checkpoint 1: feature matching
python checkpoint1.py --image-dir data/frames --pairs 5 --scale 0.5 --output-dir outputs/checkpoint1

# Checkpoint 2: two-view reconstruction
python checkpoint2.py --image-dir data/frames --idx0 0 --idx1 1 --max-dim 1600 --save-ply --save-viz --simple-K  # optional simple intrinsics

# Checkpoint 3: full SfM + export (with PLYs and visualizations)
python checkpoint3.py --image-dir data/frames --max-dim 1600 --no-plotly --save-projections
```

Outputs:
- Checkpoint 1: match visualizations saved under `outputs/checkpoint1/`.
- Checkpoint 2: optional PLY (`outputs/checkpoint2/`) and 2D projection plot (`outputs/checkpoint2/point_cloud_2d_projections.png`).
- Checkpoint 3: sparse clouds saved as `week3_sparse_before_BA.ply`, `week3_sparse_after_BA.ply`, `week3_sparse_final.ply` (plus Plotly/matplotlib views).
- Tour: Three.js viewer driven by `Tour/index.html`, `Tour/cameras.json`, `Tour/list.txt`, and a PLY you place alongside (defaults to `complete_room_web_old.ply` in the HTML).

## Visualization
- Use Open3D for quick inspection (`open3d.visualization.draw_geometries([pcd])`) or the saved projection plots from checkpoint 2.
- The `camera_poses.json` plus the original images can be fed directly into the Three.js tour viewer in the repo.

## Notes / Tuning
- `max_dim` controls downscaling; lower if memory is tight.
- `sfm/pipeline.py` exposes thresholds (min PnP matches, triangulation filtering) that can be tweaked per dataset.
- Intrinsics fall back to a default focal length when EXIF is absent; sensor size defaults to 9.8 mm (Pixel 8 Pro main sensor).
- The reusable `sfm/pipeline.py` still contains the camera JSON exporter used by the tour viewer if you need that output.

## Three.js Tour (front-end)
The `Tour/` folder is a self-contained viewer for the sparse cloud and camera poses.

1) Prepare assets  
   - Copy your sparse PLY (e.g., `week3_sparse_final.ply`) into `Tour/` and either rename it to `complete_room_web_old.ply` or update the filename in `Tour/index.html` (see `loadPointCloud()` call).  
   - Supply `cameras.json` with camera centers/rotations. A sample is provided in `Tour/cameras.json`. You can also export your own from `sfm/pipeline.py` and align the schema if you prefer.  
   - Keep `list.txt` in `Tour/` (frame order reference; not required by the current JS logic but kept for compatibility).

2) Run locally (static server is required for browser fetch):  
```
cd Tour
python -m http.server 8000
```
Then open http://localhost:8000 in your browser. (If hosting elsewhere, serve the `Tour/` folder as static files.)

3) Controls  
   - Next/Previous buttons or arrow keys to step through camera poses.  
   - “Start Auto Tour” cycles every ~3s.  
   - Checkboxes toggle camera helpers and point cloud visibility.  
   - Mouse: left-drag rotate, right-drag pan, scroll zoom (OrbitControls).

4) Customizing  
   - Point size/material: edit `PointsMaterial` in `loadPointCloud()` within `Tour/index.html`.  
   - Camera helper size/color: adjust sphere/axes creation in `loadCameras()`.  
   - If your PLY or JSON names differ, change the paths at the top of `loadPointCloud()`/`loadCameras()`.

## Repository structure (upload guidance)
- Root: notebooks + `checkpoint1/2/3.py`, `sfm/`, `README.md`.
- Data: place your frames under `data/frames/` (these are consumed by the Python scripts).
- Tour: ship `Tour/` alongside the repo so viewers can load `cameras.json` and the PLY via a static server.

## Reproducing end-to-end
1) Run `checkpoint3.py` on `data/frames` to generate sparse clouds (and optionally a camera JSON via `sfm/pipeline.py` if needed).  
2) Copy the chosen PLY into `Tour/` (renaming or updating the filename in `index.html`).  
3) Serve `Tour/` with a static server (`python -m http.server`) and open in a browser to navigate the reconstruction.  
4) Use the projection plots/PLYs from checkpoints 2 and 3 for reporting and debugging.

## For Phase 4 (Tour)
- Use the `camera_poses.json` from checkpoint 3 and place images under `images/` for the viewer.
