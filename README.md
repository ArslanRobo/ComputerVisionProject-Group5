# 3D Scene Reconstruction and Virtual Tour (CS436)

Structured Python code and a Three.js viewer for a full SfM pipeline. The notebooks remain for reporting, but all runnable logic is in `.py` files. Scripts expect pre-extracted frames in `data/frames/`.

## Requirements
- Python 3.10+
- Recommended packages:
```
pip install opencv-python-headless numpy scipy tqdm pillow piexif matplotlib open3d plotly
```

## Repository Layout
- `checkpoint1.py` – feature matching demo (SIFT + Lowe ratio) across consecutive frames; optional grid and saved match visuals.
- `checkpoint2.py` – two-view reconstruction (EXIF intrinsics or simple K, essential matrix, triangulation, outlier filtering, optional PLY and projection plots).
- `checkpoint3.py` – full SfM: EXIF intrinsics, SIFT/FLANN, two-view init, incremental PnP + triangulation, bundle adjustment, cloud cleaning, PLY exports.
- `sfm/pipeline.py` – reusable library (load/resize images, intrinsics, matching, two-view init, incremental SfM, BA, optional camera JSON export).
- `Scripts/run_checkpoint1.py` – legacy CLI wrapper around `SfMPipeline`.
- `Tour/` – Three.js viewer (`index.html`) plus example `cameras.json` and `list.txt`; drop a sparse PLY here to preview a tour.
- Not tracked by git: `data/` (frames), `CameraJson/`, `IMAGES/`, `WillSeeLater/`, `ImagesForComputerVision/`.

## Quickstart (CLI)
From repo root:
```
# Checkpoint 1: feature matching
python checkpoint1.py --image-dir data/frames --pairs 5 --scale 0.5 --output-dir outputs/checkpoint1

# Checkpoint 2: two-view reconstruction
python checkpoint2.py --image-dir data/frames --idx0 0 --idx1 1 --max-dim 1600 --save-ply --save-viz --simple-K  # optional simple K

# Checkpoint 3: full SfM
python checkpoint3.py --image-dir data/frames --max-dim 1600 --no-plotly --save-projections
```

### Outputs
- Checkpoint 1: match visualizations in `outputs/checkpoint1/`.
- Checkpoint 2: PLY (if `--save-ply`) and projection plot `outputs/checkpoint2/point_cloud_2d_projections.png`.
- Checkpoint 3: sparse clouds `week3_sparse_before_BA.ply`, `week3_sparse_after_BA.ply`, `week3_sparse_final.ply` (plus optional Plotly/matplotlib views).

## Data Preparation
Place your frames under `data/frames/` (ignored by git). Scripts load JPG/PNG, downscale to `--max-dim` (default 1600), and scale intrinsics accordingly. No video extraction is performed in these scripts.

## Three.js Tour (front-end)
The `Tour/` folder lets you browse the reconstruction in a browser.

1) Assets  
   - Copy your sparse PLY into `Tour/` and either rename it to `complete_room_web_old.ply` or change the filename in `Tour/index.html` (`loadPointCloud()` call).  
   - Provide `cameras.json` with camera centers/rotations. A sample is included; you can export your own from `sfm/pipeline.py` (camera JSON exporter) and adapt the schema if needed.  
   - `list.txt` remains as a frame-order reference (not required by the current JS logic).

2) Serve locally (required for fetch):  
```
cd Tour
python -m http.server 8000
```
Open http://localhost:8000 in a browser. Host the folder as static files if deploying elsewhere.

3) Controls  
   - Next/Previous buttons or arrow keys step through cameras.  
   - “Start Auto Tour” cycles every ~3 s.  
   - Checkboxes toggle camera helpers and point cloud visibility.  
   - Mouse: left-drag rotate, right-drag pan, scroll zoom.

4) Customizing  
   - Point size/material: edit `PointsMaterial` in `loadPointCloud()` inside `Tour/index.html`.  
   - Helper visuals: adjust sphere/axes creation in `loadCameras()`.  
   - File names/paths: update the URLs passed to `loadPointCloud()` / `loadCameras()`.

## Reproduce End-to-End
1) Run `checkpoint3.py` on `data/frames` to produce sparse clouds (and camera JSON if you enable export in `sfm/pipeline.py`).  
2) Copy the chosen PLY into `Tour/` (rename or update the path in `index.html`).  
3) Serve `Tour/` with `python -m http.server` and open it in a browser to navigate the reconstruction.  
4) Use checkpoint 2/3 plots and PLYs for reporting and debugging.

## Notes and Tuning
- `max_dim` controls downscaling; lower it if you hit memory limits.
- Thresholds for PnP/triangulation/BA can be adjusted in `sfm/pipeline.py`.
- Intrinsics default to EXIF `FocalLength`; fallback is a 9.8 mm sensor width and default focal length scaled to image size.
- Data and other ignored folders listed in `.gitignore` will not be committed; ensure you include frames and tour assets when sharing outside git.
