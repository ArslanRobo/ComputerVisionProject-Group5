"""
Checkpoint 2: two-view reconstruction.

This script translates the week-2 notebook into a reusable Python module:
- loads two frames from a folder (default: data/frames)
- computes camera intrinsics (EXIF fallback)
- detects/matches SIFT features
- estimates the essential matrix and relative pose
- triangulates a sparse point cloud, filters it, and saves/plots results

Run:
    python checkpoint2.py --image-dir data/frames --idx0 0 --idx1 1 --max-dim 1600
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from scipy import stats


def resize_image(img: np.ndarray, max_dim: int) -> Tuple[np.ndarray, float]:
    """Resize image to max_dim while keeping aspect ratio; returns resized image and scale."""
    if max_dim is None:
        return img, 1.0
    h, w = img.shape[:2]
    scale = min(1.0, max_dim / max(h, w))
    if scale != 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img, scale


def load_image_pair(
    image_dir: Path, idx0: int, idx1: int, max_dim: int
) -> Tuple[np.ndarray, np.ndarray, str, str, float, Tuple[int, int]]:
    """
    Load two images from the folder by index, resize, and return:
    (img1_resized, img2_resized, path1, path2, scale, original_shape).
    """
    paths = sorted(
        [p for p in image_dir.glob("*.jpg")] + [p for p in image_dir.glob("*.png")]
    )
    if len(paths) < max(idx0, idx1) + 1:
        raise ValueError(f"Not enough frames in {image_dir}; found {len(paths)}.")

    img1 = cv2.imread(str(paths[idx0]))
    img2 = cv2.imread(str(paths[idx1]))
    if img1 is None or img2 is None:
        raise FileNotFoundError("Failed to load one or both images.")

    orig_shape = img1.shape[:2]

    img1, scale1 = resize_image(img1, max_dim)
    img2, scale2 = resize_image(img2, max_dim)
    if abs(scale1 - scale2) > 1e-6:
        raise ValueError("Images were resized with different scales; check inputs.")

    return img1, img2, str(paths[idx0]), str(paths[idx1]), scale1, orig_shape


def intrinsics_from_exif(
    image_path: str,
    image_shape: Tuple[int, int],
    sensor_width_mm: float = 9.8,
    default_fl_mm: float = 6.9,
) -> np.ndarray:
    """
    Compute camera intrinsics using EXIF focal length when available.
    Falls back to default focal length otherwise.
    """
    h, w = image_shape
    fl_mm = default_fl_mm
    try:
        import piexif

        exif = piexif.load(image_path)
        fl = exif["Exif"][piexif.ExifIFD.FocalLength]
        fl_mm = fl[0] / fl[1]
        print(f"EXIF focal length: {fl_mm:.2f} mm")
    except Exception:
        print(f"Using default focal length: {fl_mm:.2f} mm (EXIF missing)")

    fx = (fl_mm / sensor_width_mm) * w
    fy = fx
    cx, cy = w / 2, h / 2
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)


def detect_and_match_sift(img1_gray: np.ndarray, img2_gray: np.ndarray, ratio: float = 0.75):
    """Detect SIFT keypoints/descriptors and match with Lowe's ratio test."""
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1_gray, None)
    kp2, des2 = sift.detectAndCompute(img2_gray, None)
    if des1 is None or des2 is None or len(des1) == 0 or len(des2) == 0:
        return kp1 or [], kp2 or [], []
    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    matches_knn = matcher.knnMatch(des1, des2, k=2)
    good = [m for m, n in matches_knn if m.distance < ratio * n.distance]
    return kp1, kp2, good


def estimate_pose(pts1: np.ndarray, pts2: np.ndarray, K: np.ndarray):
    """Estimate essential matrix and recover relative camera pose."""
    E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    _, R, t, pose_mask = cv2.recoverPose(E, pts1, pts2, K, mask=mask)
    return R, t, pose_mask


def triangulate_points(P1: np.ndarray, P2: np.ndarray, pts1: np.ndarray, pts2: np.ndarray) -> np.ndarray:
    """Triangulate 2D correspondences into 3D points."""
    points_4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
    pts3d = (points_4d[:3] / points_4d[3]).T
    return pts3d


def cheirality_mask(P: np.ndarray, points_3d: np.ndarray) -> np.ndarray:
    """Return mask of points with positive depth in the camera frame."""
    pts_h = np.hstack([points_3d, np.ones((points_3d.shape[0], 1))])
    cam_pts = (P @ pts_h.T).T
    return cam_pts[:, 2] > 0


def remove_outliers(points: np.ndarray, z_thresh: float = 3.0) -> np.ndarray:
    """Remove extreme outliers using z-score thresholding."""
    if points.size == 0:
        return points
    z_scores = np.abs(stats.zscore(points, axis=0))
    mask = (z_scores < z_thresh).all(axis=1)
    return points[mask]


def plot_projections(points: np.ndarray, save_path: Path | None = None):
    """Plot XY/XZ/YZ projections of the point cloud."""
    if len(points) == 0:
        print("No points to plot.")
        return
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].scatter(points[:, 0], points[:, 1], s=1, c="blue", alpha=0.5)
    axes[0].set_title("XY Projection")
    axes[0].axis("equal")
    axes[1].scatter(points[:, 0], points[:, 2], s=1, c="green", alpha=0.5)
    axes[1].set_title("XZ Projection")
    axes[1].axis("equal")
    axes[2].scatter(points[:, 1], points[:, 2], s=1, c="red", alpha=0.5)
    axes[2].set_title("YZ Projection")
    axes[2].axis("equal")
    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def save_point_cloud(points: np.ndarray, path: Path):
    """Save a point cloud to PLY."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(path), pcd)


@dataclass
class ReconstructionResult:
    R: np.ndarray
    t: np.ndarray
    K: np.ndarray
    points: np.ndarray
    inlier_mask: np.ndarray


def run_two_view_reconstruction(
    image_dir: Path,
    idx0: int,
    idx1: int,
    max_dim: int,
    ratio: float,
    z_thresh: float,
    save_viz: bool,
    use_simple_intrinsics: bool,
) -> ReconstructionResult:
    """End-to-end two-view reconstruction pipeline."""
    img1, img2, p1, p2, scale, orig_shape = load_image_pair(image_dir, idx0, idx1, max_dim)
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    h, w = orig_shape
    if use_simple_intrinsics:
        K = np.array([[w, 0, w / 2], [0, w, h / 2], [0, 0, 1.0]], dtype=np.float64)
        print("Using simple intrinsic matrix (fx=fy=width, cx=cy=center).")
    else:
        K = intrinsics_from_exif(p1, orig_shape, default_fl_mm=6.9)
    if scale != 1.0:
        K[:2] *= scale

    kp1, kp2, good = detect_and_match_sift(gray1, gray2, ratio)
    print(f"Matched {len(good)} features between frames {idx0} and {idx1}")

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good])

    R, t, pose_mask = estimate_pose(pts1, pts2, K)
    inliers = pose_mask.ravel().astype(bool)
    pts1_inl = pts1[inliers]
    pts2_inl = pts2[inliers]

    P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = K @ np.hstack([R, t])
    points_3d = triangulate_points(P1, P2, pts1_inl, pts2_inl)

    # Keep points in front of both cameras
    mask1 = cheirality_mask(P1, points_3d)
    mask2 = cheirality_mask(P2, points_3d)
    valid = mask1 & mask2
    points_3d = points_3d[valid]

    # Remove extreme outliers
    points_clean = remove_outliers(points_3d, z_thresh=z_thresh)
    print(f"Points: {len(points_3d)} after cheirality, {len(points_clean)} after outlier removal")

    if save_viz:
        plot_projections(points_clean, save_path=Path("outputs/checkpoint2/point_cloud_2d_projections.png"))

    return ReconstructionResult(R=R, t=t, K=K, points=points_clean, inlier_mask=inliers)


def parse_args():
    parser = argparse.ArgumentParser(description="Checkpoint 2: two-view reconstruction")
    parser.add_argument("--image-dir", type=Path, default=Path("data/frames"), help="Folder with frames.")
    parser.add_argument("--idx0", type=int, default=0, help="Index of first frame.")
    parser.add_argument("--idx1", type=int, default=1, help="Index of second frame.")
    parser.add_argument("--max-dim", type=int, default=1600, help="Resize longer side to this value.")
    parser.add_argument("--ratio", type=float, default=0.75, help="Lowe ratio threshold.")
    parser.add_argument("--z-thresh", type=float, default=3.0, help="Z-score threshold for outlier removal.")
    parser.add_argument("--save-ply", action="store_true", help="Save the reconstructed point cloud to PLY.")
    parser.add_argument("--save-viz", action="store_true", help="Save 2D projection visualizations.")
    parser.add_argument("--simple-K", action="store_true", help="Use simple intrinsics (fx=fy=width, cx=cy=center).")
    return parser.parse_args()


def main():
    args = parse_args()
    result = run_two_view_reconstruction(
        args.image_dir,
        args.idx0,
        args.idx1,
        args.max_dim,
        args.ratio,
        args.z_thresh,
        args.save_viz,
        use_simple_intrinsics=args.simple_K,
    )

    print("\nReconstruction summary")
    print("---------------------")
    print(f"Rotation:\n{result.R}")
    print(f"Translation (normalized): {result.t.ravel()}")
    print(f"Points reconstructed: {len(result.points)}")

    if args.save_ply:
        ply_path = Path("outputs/checkpoint2/two_view_reconstruction.ply")
        save_point_cloud(result.points, ply_path)
        print(f"Saved point cloud to {ply_path}")


if __name__ == "__main__":
    main()
