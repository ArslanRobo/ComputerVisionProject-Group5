"""
Checkpoint 3: full SfM pipeline with refinement and exports.

This script is a faithful Python translation of the checkpoint3 notebook. It:
- loads frames from a folder (default: data/frames)
- estimates intrinsics from EXIF (with sane defaults)
- extracts SIFT features and matches with FLANN + ratio test
- runs two-view initialization, incremental PnP registration, triangulation
- performs bundle adjustment
- cleans the sparse cloud and saves PLYs before/after cleaning
- offers simple visualizations (matplotlib projections and Plotly 3D)

Run:
    python checkpoint3.py --image-dir data/frames --max-dim 1600
    python checkpoint3.py --image-dir data/frames --max-dim 1600 --no-plotly --save-projections
"""

from __future__ import annotations

import argparse
import os
import warnings
from dataclasses import dataclass
from glob import glob
from pathlib import Path
from typing import Dict, List, Set, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import plotly.graph_objects as go
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from tqdm import tqdm

warnings.filterwarnings("ignore")


# -----------------------------------------------------------------------------
# Data structures
# -----------------------------------------------------------------------------

@dataclass
class CameraPose:
    R: np.ndarray  # 3x3 rotation matrix
    t: np.ndarray  # 3x1 translation vector
    image_idx: int

    def to_projection_matrix(self, K: np.ndarray) -> np.ndarray:
        return K @ np.hstack((self.R, self.t))

    def to_params(self) -> np.ndarray:
        rvec, _ = cv2.Rodrigues(self.R)
        return np.concatenate([rvec.ravel(), self.t.ravel()])

    @staticmethod
    def from_params(params: np.ndarray, image_idx: int):
        rvec = params[:3].reshape(3, 1)
        tvec = params[3:6].reshape(3, 1)
        R, _ = cv2.Rodrigues(rvec)
        return CameraPose(R, tvec, image_idx)


@dataclass
class Observation:
    image_idx: int
    point_3d_idx: int
    kp_idx: int
    point_2d: np.ndarray


@dataclass
class ReconstructionState:
    points_3d: np.ndarray
    camera_poses: Dict[int, CameraPose]
    observations: List[Observation]
    registered: Set[int]


# -----------------------------------------------------------------------------
# Step 1: Data loading & preprocessing
# -----------------------------------------------------------------------------

class ImageLoader:
    """Handle image loading and resizing."""

    def __init__(self, image_folder: str, max_dim: int = 2000):
        self.image_folder = image_folder
        self.max_dim = max_dim

    def load_and_resize(self) -> Tuple[List[np.ndarray], List[str]]:
        paths = sorted(
            glob(os.path.join(self.image_folder, "*.jpg"))
            + glob(os.path.join(self.image_folder, "*.png"))
        )
        if not paths:
            raise ValueError(f"No images found in {self.image_folder}")

        images = []
        for path in tqdm(paths, desc="Loading images"):
            img = cv2.imread(path)
            if img is None:
                continue
            h, w = img.shape[:2]
            scale = min(1.0, self.max_dim / max(h, w))
            new_w, new_h = int(w * scale), int(h * scale)
            img_resized = cv2.resize(img, (new_w, new_h))
            images.append(img_resized)

        print(f"Loaded {len(images)} images, size: {images[0].shape}")
        return images, paths


# -----------------------------------------------------------------------------
# Step 2: Camera calibration
# -----------------------------------------------------------------------------

class CameraCalibrator:
    """Extract camera intrinsics from EXIF or use defaults."""

    @staticmethod
    def get_intrinsics(
        image_path: str,
        image_shape: Tuple[int, int],
        sensor_width_mm: float = 9.8,
        sensor_height_mm: float | None = None,
        default_fl_mm: float = 6.9,
    ) -> np.ndarray:
        import piexif

        h, w = image_shape
        if sensor_height_mm is None:
            sensor_height_mm = sensor_width_mm

        try:
            exif = piexif.load(image_path)
            fl = exif["Exif"][piexif.ExifIFD.FocalLength]
            fl_mm = fl[0] / fl[1]
            print(f"EXIF focal length: {fl_mm:.2f} mm")
        except Exception:
            fl_mm = default_fl_mm
            print(f"Using default focal length: {fl_mm:.2f} mm")

        fx = (fl_mm / sensor_width_mm) * w
        fy = (fl_mm / sensor_height_mm) * h
        cx, cy = w / 2, h / 2
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)


# -----------------------------------------------------------------------------
# Step 3: Feature extraction & matching
# -----------------------------------------------------------------------------

class FeatureExtractor:
    def __init__(self, n_features: int = 2000):
        self.sift = cv2.SIFT_create(nfeatures=n_features)

    def extract_all(self, images: List[np.ndarray]) -> Tuple[List, List]:
        kps, descs = [], []
        for img in tqdm(images, desc="Extracting SIFT"):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            kp, des = self.sift.detectAndCompute(gray, None)
            kps.append(kp)
            descs.append(des)
        print(f"Extracted features for {len(images)} images")
        return kps, descs


class FeatureMatcher:
    def __init__(self):
        index_params = dict(algorithm=1, trees=5)
        search_params = dict(checks=50)
        self.matcher = cv2.FlannBasedMatcher(index_params, search_params)

    def match_pair(self, des1: np.ndarray, des2: np.ndarray, ratio_thresh: float = 0.75) -> List:
        if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
            return []
        matches = self.matcher.knnMatch(des1, des2, k=2)
        good = []
        for match in matches:
            if len(match) == 2:
                m, n = match
                if m.distance < ratio_thresh * n.distance:
                    good.append(m)
        return good


# -----------------------------------------------------------------------------
# Step 4: Initial two-view reconstruction
# -----------------------------------------------------------------------------

class TwoViewReconstructor:
    def __init__(self, K: np.ndarray):
        self.K = K

    def find_initial_pair(
        self,
        kps: List,
        descs: List,
        matcher: FeatureMatcher,
        min_matches: int = 100,
    ) -> Tuple[int, int, List]:
        print("Finding initial pair...")
        best_pair, best_matches, best_score = None, [], 0
        for i in range(min(20, len(kps) - 1)):
            for j in range(i + 1, min(i + 5, len(kps))):
                m = matcher.match_pair(descs[i], descs[j])
                if len(m) > min_matches:
                    score = len(m) * (j - i)
                    if score > best_score:
                        best_score = score
                        best_pair = (i, j)
                        best_matches = m
        if best_pair is None:
            raise ValueError("Could not find suitable initial pair")
        print(f"Initial pair: {best_pair[0]} and {best_pair[1]} with {len(best_matches)} matches")
        return best_pair[0], best_pair[1], best_matches

    def reconstruct(
        self, kps: List, matches: List, i0: int, i1: int
    ) -> Tuple[CameraPose, CameraPose, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        pts1 = np.float32([kps[i0][m.queryIdx].pt for m in matches])
        pts2 = np.float32([kps[i1][m.trainIdx].pt for m in matches])

        E, mask = cv2.findEssentialMat(pts1, pts2, self.K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        _, R, t, pose_mask = cv2.recoverPose(E, pts1, pts2, self.K, mask=mask)

        inliers = pose_mask.ravel().astype(bool)
        match_indices = np.where(inliers)[0]
        pts1_inl = pts1[inliers]
        pts2_inl = pts2[inliers]

        P1 = self.K @ np.hstack((np.eye(3), np.zeros((3, 1))))
        P2 = self.K @ np.hstack((R, t))
        points_4d = cv2.triangulatePoints(P1, P2, pts1_inl.T, pts2_inl.T)
        points_3d = (points_4d[:3] / points_4d[3]).T

        valid = points_3d[:, 2] > 0
        points_3d = points_3d[valid]
        pts1_inl = pts1_inl[valid]
        pts2_inl = pts2_inl[valid]
        match_indices = match_indices[valid]

        pose0 = CameraPose(np.eye(3), np.zeros((3, 1)), i0)
        pose1 = CameraPose(R, t, i1)
        print(f"Initial reconstruction: {len(points_3d)} 3D points")
        return pose0, pose1, points_3d, match_indices, pts1_inl, pts2_inl


# -----------------------------------------------------------------------------
# Step 5: Incremental reconstruction with PnP
# -----------------------------------------------------------------------------

class IncrementalReconstructor:
    def __init__(self, K: np.ndarray, min_2d3d_matches: int = 10):
        self.K = K
        self.min_2d3d_matches = min_2d3d_matches

    def find_2d3d_correspondences(
        self,
        new_idx: int,
        registered_indices: Set[int],
        kps: List,
        matcher: FeatureMatcher,
        descs: List,
        observations: List[Observation],
        points_3d: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        pts3d, pts2d, point_indices = [], [], []
        obs_map = {(obs.image_idx, obs.kp_idx): obs.point_3d_idx for obs in observations}

        for reg_idx in registered_indices:
            matches = matcher.match_pair(descs[reg_idx], descs[new_idx])
            for m in matches:
                key = (reg_idx, m.queryIdx)
                if key in obs_map and obs_map[key] < len(points_3d):
                    pts3d.append(points_3d[obs_map[key]])
                    pts2d.append(kps[new_idx][m.trainIdx].pt)
                    point_indices.append(obs_map[key])

        if len(pts3d) < self.min_2d3d_matches:
            return np.array([]), np.array([]), np.array([])
        return np.array(pts3d), np.array(pts2d), np.array(point_indices)

    def register_image(self, image_idx: int, pts3d: np.ndarray, pts2d: np.ndarray) -> Tuple[bool, CameraPose | None]:
        if len(pts3d) < self.min_2d3d_matches:
            return False, None
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            pts3d,
            pts2d,
            self.K,
            None,
            reprojectionError=4.0,
            confidence=0.999,
            iterationsCount=1000,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not success or inliers is None or len(inliers) < self.min_2d3d_matches:
            return False, None
        R, _ = cv2.Rodrigues(rvec)
        t = tvec.reshape((3, 1))
        return True, CameraPose(R, t, image_idx)


# -----------------------------------------------------------------------------
# Step 6: Triangulation with quality checks
# -----------------------------------------------------------------------------

class Triangulator:
    @staticmethod
    def triangulate_points(pose1: CameraPose, pose2: CameraPose, pts1: np.ndarray, pts2: np.ndarray, K: np.ndarray) -> np.ndarray:
        P1 = pose1.to_projection_matrix(K)
        P2 = pose2.to_projection_matrix(K)
        points_4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
        return (points_4d[:3] / points_4d[3]).T

    @staticmethod
    def filter_triangulated_points(
        points_3d: np.ndarray,
        pose1: CameraPose,
        pose2: CameraPose,
        pts1: np.ndarray,
        pts2: np.ndarray,
        K: np.ndarray,
        max_reproj_error: float = 2.0,
        min_depth: float = 0.1,
        max_depth: float = 500.0,
    ) -> np.ndarray:
        valid_mask = np.ones(len(points_3d), dtype=bool)
        depths1 = pose1.R @ points_3d.T + pose1.t
        depths2 = pose2.R @ points_3d.T + pose2.t
        valid_mask &= (depths1[2] > min_depth) & (depths1[2] < max_depth)
        valid_mask &= (depths2[2] > min_depth) & (depths2[2] < max_depth)

        P1 = pose1.to_projection_matrix(K)
        P2 = pose2.to_projection_matrix(K)
        pts1_proj = (P1 @ np.hstack([points_3d, np.ones((len(points_3d), 1))]).T)
        pts1_proj = (pts1_proj[:2] / pts1_proj[2]).T
        pts2_proj = (P2 @ np.hstack([points_3d, np.ones((len(points_3d), 1))]).T)
        pts2_proj = (pts2_proj[:2] / pts2_proj[2]).T
        error1 = np.linalg.norm(pts1 - pts1_proj, axis=1)
        error2 = np.linalg.norm(pts2 - pts2_proj, axis=1)
        valid_mask &= (error1 < max_reproj_error) & (error2 < max_reproj_error)
        return valid_mask


# -----------------------------------------------------------------------------
# Step 7: Bundle adjustment
# -----------------------------------------------------------------------------

class BundleAdjuster:
    def __init__(self, K: np.ndarray):
        self.K = K

    def prepare_parameters(self, camera_poses: Dict[int, CameraPose], points_3d: np.ndarray) -> np.ndarray:
        n_cameras = len(camera_poses)
        params = np.zeros(n_cameras * 6 + len(points_3d) * 3)
        for i, (img_idx, pose) in enumerate(sorted(camera_poses.items())):
            params[i * 6 : (i + 1) * 6] = pose.to_params()
        params[n_cameras * 6 :] = points_3d.ravel()
        return params

    def unpack_parameters(self, params: np.ndarray, camera_indices: List[int], n_points: int) -> Tuple[Dict[int, CameraPose], np.ndarray]:
        n_cameras = len(camera_indices)
        camera_poses = {}
        for i, img_idx in enumerate(camera_indices):
            camera_params = params[i * 6 : (i + 1) * 6]
            camera_poses[img_idx] = CameraPose.from_params(camera_params, img_idx)
        points_3d = params[n_cameras * 6 :].reshape((n_points, 3))
        return camera_poses, points_3d

    def compute_residuals(
        self,
        params: np.ndarray,
        observations: List[Observation],
        camera_indices: List[int],
        n_points: int,
    ) -> np.ndarray:
        camera_poses, points_3d = self.unpack_parameters(params, camera_indices, n_points)
        residuals = []
        for obs in observations:
            if obs.image_idx not in camera_poses:
                continue
            if obs.point_3d_idx >= len(points_3d):
                continue
            pose = camera_poses[obs.image_idx]
            point_3d = points_3d[obs.point_3d_idx]
            point_cam = pose.R @ point_3d + pose.t.ravel()
            point_2d_proj = self.K @ point_cam
            point_2d_proj = point_2d_proj[:2] / point_2d_proj[2]
            residual = obs.point_2d - point_2d_proj
            residuals.extend(residual)
        return np.array(residuals)

    def optimize(
        self,
        camera_poses: Dict[int, CameraPose],
        points_3d: np.ndarray,
        observations: List[Observation],
        max_iterations: int = 50,
    ) -> Tuple[Dict[int, CameraPose], np.ndarray]:
        print("\n" + "=" * 70)
        print("BUNDLE ADJUSTMENT OPTIMIZATION")
        print("=" * 70)

        camera_indices = sorted(camera_poses.keys())
        params_init = self.prepare_parameters(camera_poses, points_3d)
        residuals_init = self.compute_residuals(params_init, observations, camera_indices, len(points_3d))
        error_init = np.sqrt(np.mean(residuals_init ** 2))
        print(f"Initial RMSE: {error_init:.3f} pixels")

        print("\nOptimizing...")
        result = least_squares(
            self.compute_residuals,
            params_init,
            args=(observations, camera_indices, len(points_3d)),
            method="trf",
            max_nfev=max_iterations,
            verbose=2,
            ftol=1e-4,
            xtol=1e-4,
        )

        camera_poses_opt, points_3d_opt = self.unpack_parameters(result.x, camera_indices, len(points_3d))
        residuals_final = self.compute_residuals(result.x, observations, camera_indices, len(points_3d))
        error_final = np.sqrt(np.mean(residuals_final ** 2))
        print(f"Final RMSE: {error_final:.3f} pixels (improvement {error_init - error_final:.3f})")
        print("=" * 70 + "\n")
        return camera_poses_opt, points_3d_opt


# -----------------------------------------------------------------------------
# Step 8: Outlier removal & point cloud cleaning
# -----------------------------------------------------------------------------

class PointCloudCleaner:
    @staticmethod
    def remove_statistical_outliers(points: np.ndarray, nb_neighbors: int = 20, std_ratio: float = 2.0) -> np.ndarray:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd_clean, _ = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
        points_clean = np.asarray(pcd_clean.points)
        print(f"Statistical outlier removal: {len(points)} -> {len(points_clean)} points")
        return points_clean

    @staticmethod
    def remove_extreme_points(points: np.ndarray, threshold: float = 1000.0) -> np.ndarray:
        mask = (np.abs(points) < threshold).all(axis=1)
        points_clean = points[mask]
        print(f"Extreme point removal: {len(points)} -> {len(points_clean)} points")
        return points_clean

    @staticmethod
    def remove_duplicate_points(points: np.ndarray, tolerance: float = 0.01) -> np.ndarray:
        if len(points) == 0:
            return points
        tree = cKDTree(points)
        pairs = tree.query_pairs(r=tolerance)
        remove_indices = {j for _, j in pairs}
        keep_indices = [i for i in range(len(points)) if i not in remove_indices]
        points_clean = points[keep_indices]
        print(f"Duplicate removal: {len(points)} -> {len(points_clean)} points")
        return points_clean


# -----------------------------------------------------------------------------
# Step 9: Visualization utilities
# -----------------------------------------------------------------------------

class Visualizer:
    @staticmethod
    def plot_point_cloud_plotly(points: np.ndarray, title: str = "Point Cloud"):
        fig = go.Figure(
            data=[
                go.Scatter3d(
                    x=points[:, 0],
                    y=points[:, 1],
                    z=points[:, 2],
                    mode="markers",
                    marker=dict(size=2, color=points[:, 2], colorscale="Viridis", opacity=0.8),
                )
            ]
        )
        fig.update_layout(
            title=title,
            scene=dict(aspectmode="data", xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
            width=1000,
            height=700,
        )
        fig.show()

    @staticmethod
    def save_point_cloud(points: np.ndarray, filename: str, color=None):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.paint_uniform_color(color if color is not None else [0.2, 0.6, 1.0])
        o3d.io.write_point_cloud(filename, pcd)
        print(f"Saved point cloud: {filename}")

    @staticmethod
    def compare_point_clouds(points_before: np.ndarray, points_after: np.ndarray):
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        axes[0].scatter(points_before[:, 0], points_before[:, 2], s=1, alpha=0.5, c=points_before[:, 2], cmap="viridis")
        axes[0].set_title(f"Before Refinement ({len(points_before)} points)")
        axes[0].set_xlabel("X")
        axes[0].set_ylabel("Z")
        axes[0].axis("equal")

        axes[1].scatter(points_after[:, 0], points_after[:, 2], s=1, alpha=0.5, c=points_after[:, 2], cmap="viridis")
        axes[1].set_title(f"After Bundle Adjustment ({len(points_after)} points)")
        axes[1].set_xlabel("X")
        axes[1].set_ylabel("Z")
        axes[1].axis("equal")

        plt.tight_layout()
        plt.show()


def plot_point_cloud_projections(file_path: Path):
    """Plot XZ, XY, YZ projections from a saved PLY."""
    print(f"Loading point cloud from: {file_path}")
    pcd = o3d.io.read_point_cloud(str(file_path))
    points = np.asarray(pcd.points)
    if points.shape[0] == 0:
        print("The point cloud is empty, cannot create 2D images.")
        return

    fig_xz = plt.figure(figsize=(8, 6))
    plt.scatter(points[:, 0], points[:, 2], s=5, alpha=0.6)
    plt.title("Point Cloud Projection: X-Z Plane")
    plt.xlabel("X")
    plt.ylabel("Z")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.show()

    fig_xy = plt.figure(figsize=(8, 6))
    plt.scatter(points[:, 0], points[:, 1], s=5, alpha=0.6)
    plt.title("Point Cloud Projection: X-Y Plane")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.show()

    fig_yz = plt.figure(figsize=(8, 6))
    plt.scatter(points[:, 1], points[:, 2], s=5, alpha=0.6)
    plt.title("Point Cloud Projection: Y-Z Plane")
    plt.xlabel("Y")
    plt.ylabel("Z")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.show()


# -----------------------------------------------------------------------------
# Main pipeline
# -----------------------------------------------------------------------------

class SfMPipeline:
    def __init__(self, image_folder: str, max_dim: int = 640):
        self.image_folder = image_folder
        self.max_dim = max_dim
        self.state: ReconstructionState | None = None

    def run(self, run_plotly: bool = True, compare: bool = True) -> ReconstructionState:
        print("\n" + "=" * 70)
        print("STRUCTURE FROM MOTION PIPELINE")
        print("=" * 70 + "\n")

        print("STEP 1: Loading Images")
        loader = ImageLoader(self.image_folder, self.max_dim)
        images, image_paths = loader.load_and_resize()

        print("\nSTEP 2: Camera Calibration")
        h, w = images[0].shape[:2]
        K = CameraCalibrator.get_intrinsics(image_paths[0], (h, w))
        print(f"Camera intrinsics K:\n{K}\n")

        print("STEP 3: Feature Extraction")
        extractor = FeatureExtractor(n_features=2000)
        kps, descs = extractor.extract_all(images)

        print("\nSTEP 4: Feature Matching")
        matcher = FeatureMatcher()

        print("\nSTEP 5: Initial Two-View Reconstruction")
        two_view = TwoViewReconstructor(K)
        i0, i1, matches01 = two_view.find_initial_pair(kps, descs, matcher)
        pose0, pose1, pts3d_init, kept_idx, pts1_filt, pts2_filt = two_view.reconstruct(kps, matches01, i0, i1)

        camera_poses = {i0: pose0, i1: pose1}
        registered = {i0, i1}
        points_3d = pts3d_init.copy()
        observations: List[Observation] = []
        for idx, match_idx in enumerate(kept_idx):
            if idx < len(pts3d_init):
                m = matches01[match_idx]
                observations.append(Observation(i0, idx, m.queryIdx, pts1_filt[idx]))
                observations.append(Observation(i1, idx, m.trainIdx, pts2_filt[idx]))
        print(f"Initial observations: {len(observations)}")

        print("\nSTEP 6: Incremental PnP Registration")
        incremental = IncrementalReconstructor(K, min_2d3d_matches=15)
        triangulator = Triangulator()
        unregistered = [i for i in range(len(images)) if i not in registered]
        iteration = 0
        max_iterations = 10

        while unregistered and iteration < max_iterations:
            iteration += 1
            print(f"\nIteration {iteration}: {len(unregistered)} images remaining")
            newly_registered = []
            for j in tqdm(unregistered, desc="Registering images"):
                pts3d, pts2d, _ = incremental.find_2d3d_correspondences(
                    j, registered, kps, matcher, descs, observations, points_3d
                )
                if len(pts3d) < 15:
                    continue
                success, pose_j = incremental.register_image(j, pts3d, pts2d)
                if success and pose_j:
                    newly_registered.append((j, pose_j))

            if not newly_registered:
                print("No more images can be registered. Stopping.")
                break

            for j, pose_j in newly_registered:
                registered.add(j)
                camera_poses[j] = pose_j
                best_ref = i0  # keep the first view as reference, as in notebook
                matches_new = matcher.match_pair(descs[best_ref], descs[j])
                if len(matches_new) < 20:
                    continue
                pts_ref = np.float32([kps[best_ref][m.queryIdx].pt for m in matches_new])
                pts_new = np.float32([kps[j][m.trainIdx].pt for m in matches_new])
                points_new = triangulator.triangulate_points(camera_poses[best_ref], pose_j, pts_ref, pts_new, K)
                valid_mask = triangulator.filter_triangulated_points(
                    points_new, camera_poses[best_ref], pose_j, pts_ref, pts_new, K
                )
                points_new_f = points_new[valid_mask]
                pts_ref_f = pts_ref[valid_mask]
                pts_new_f = pts_new[valid_mask]
                keep_ids = np.nonzero(valid_mask)[0]
                start_idx = len(points_3d)
                points_3d = np.vstack([points_3d, points_new_f])
                for local_idx, match_idx in enumerate(keep_ids):
                    p_idx = start_idx + local_idx
                    m = matches_new[match_idx]
                    observations.append(Observation(best_ref, p_idx, m.queryIdx, pts_ref_f[local_idx]))
                    observations.append(Observation(j, p_idx, m.trainIdx, pts_new_f[local_idx]))

            print(f"Registered {len(newly_registered)} images. Total: {len(registered)}/{len(images)}")
            print(f"Total 3D points: {len(points_3d)}")
            unregistered = [i for i in range(len(images)) if i not in registered]

        print("\n" + "=" * 70)
        print("INCREMENTAL RECONSTRUCTION COMPLETE")
        print("=" * 70)
        print(f"Registered images: {len(registered)}/{len(images)}")
        print(f"Total 3D points: {len(points_3d)}")
        print(f"Total observations: {len(observations)}\n")

        points_3d_before = points_3d.copy()
        Visualizer.save_point_cloud(points_3d, "week3_sparse_before_BA.ply")

        print("STEP 7: Bundle Adjustment (Refinement)")
        ba = BundleAdjuster(K)
        camera_poses_refined, points_3d_refined = ba.optimize(camera_poses, points_3d, observations, max_iterations=50)

        print("STEP 8: Point Cloud Cleaning")
        cleaner = PointCloudCleaner()
        points_3d_clean = cleaner.remove_extreme_points(points_3d_refined, threshold=1000)
        points_3d_clean = cleaner.remove_statistical_outliers(points_3d_clean, nb_neighbors=20, std_ratio=2.0)
        points_3d_clean = cleaner.remove_duplicate_points(points_3d_clean, tolerance=0.01)

        print("\n" + "=" * 70)
        print("FINAL POINT CLOUD")
        print("=" * 70)
        print(f"Before BA: {len(points_3d_before)} points")
        print(f"After BA: {len(points_3d_refined)} points")
        print(f"After cleaning: {len(points_3d_clean)} points\n")

        print("STEP 9: Saving Results")
        Visualizer.save_point_cloud(points_3d_refined, "week3_sparse_after_BA.ply")
        Visualizer.save_point_cloud(points_3d_clean, "week3_sparse_final.ply")

        if compare and len(points_3d_before) < 100000:
            Visualizer.compare_point_clouds(points_3d_before, points_3d_clean)
        if run_plotly and len(points_3d_clean) < 50000:
            Visualizer.plot_point_cloud_plotly(points_3d_clean, "Week 3: Final Refined Point Cloud")

        self.state = ReconstructionState(
            points_3d=points_3d_refined,
            camera_poses=camera_poses_refined,
            observations=observations,
            registered=registered,
        )
        print("\n" + "=" * 70)
        print("SfM PIPELINE COMPLETE")
        print("=" * 70)
        print("Output files:")
        print("  - week3_sparse_before_BA.ply")
        print("  - week3_sparse_after_BA.ply")
        print("  - week3_sparse_final.ply")
        print(f"Registered images: {len(registered)}/{len(images)}")
        print(f"Final point cloud: {len(points_3d_clean)} points")
        print(f"Camera poses: {len(camera_poses_refined)}")
        return self.state


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Checkpoint 3: full SfM reconstruction")
    parser.add_argument("--image-dir", type=Path, default=Path("data/frames"), help="Folder with frames (jpg/png).")
    parser.add_argument("--max-dim", type=int, default=1600, help="Resize longer side to this value.")
    parser.add_argument("--no-plotly", action="store_true", help="Disable interactive Plotly visualization.")
    parser.add_argument("--no-compare", action="store_true", help="Skip before/after matplotlib comparison.")
    parser.add_argument("--save-projections", action="store_true", help="Plot XY/XZ/YZ projections of the final PLY.")
    return parser.parse_args()


def main():
    args = parse_args()
    pipeline = SfMPipeline(str(args.image_dir), max_dim=args.max_dim)
    state = pipeline.run(run_plotly=not args.no_plotly, compare=not args.no_compare)

    if args.save_projections:
        plot_point_cloud_projections(Path("week3_sparse_after_BA.ply"))

    print(f"Point cloud shape: {state.points_3d.shape}")
    print(f"Registered cameras: {len(state.camera_poses)}")


if __name__ == "__main__":
    main()
