# Project Overview
- `checkpoint3.py`: Step-wise script for multi-view structure-from-motion with incremental PnP, map expansion, and bundle adjustment.
- `checkpoint3.ipynb`: Notebook version of `checkpoint3.py`, split into cells by each step header for interactive runs.
- `checkpoint1.ipynb`, `checkpoint2.ipynb`: Earlier checkpoints for the project (kept for reference).
- `ai_usage_report.md`: Notes on AI assistance usage.

# Code Structure (checkpoint3)
- Configuration: Image path, resize settings, and core thresholds/constants.
- Step 1: Load and resize images.
- Step 2: Estimate camera intrinsics (EXIF fallback to default focal length).
- Step 3: SIFT feature extraction per frame.
- Step 4: FLANN feature matching with Lowe ratio test.
- Step 5: Two-view initialization (essential matrix, pose recovery, initial triangulation).
- Step 6: Initialize SfM data structures (camera poses, point cloud, observations).
- Step 7: Incremental PnP registration for additional images.
- Step 8: Map expansion via triangulation of new matches between registered views.
- Step 9: Bundle adjustment (least squares) over cameras and points.
- Step 10: Outlier removal with z-score filtering.
- Step 11: 3D visualization (Open3D export + Plotly scatter).
- Step 12: Final summary of counts and output path.

# Usage Notes
- Run the notebook for interactive exploration or execute `python checkpoint3.py` for the scripted pipeline.
- Dependencies: OpenCV (with SIFT), NumPy, Matplotlib, Open3D, SciPy, Plotly, tqdm, glob, and standard library modules.
