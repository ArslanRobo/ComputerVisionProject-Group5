"""
Checkpoint 1: feature detection and matching on frame pairs.

This script mirrors the demonstration notebook but runs as a reusable CLI:
- loads frames from a folder (default: data/frames)
- optionally shows a random grid of frames
- performs SIFT matching across consecutive pairs and saves/prints results

Run:
    python checkpoint1.py --image-dir data/frames --pairs 5 --scale 0.5
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np


def discover_frames(image_dir: Path) -> List[Path]:
    """Return sorted list of frame paths (jpg/png)."""
    paths = sorted(
        [p for p in image_dir.glob("*.jpg")] + [p for p in image_dir.glob("*.png")]
    )
    if not paths:
        raise FileNotFoundError(f"No frames found in {image_dir}")
    return paths


def preprocess_image(img_path: Path, scale: float = 0.5) -> np.ndarray:
    """Load a frame as grayscale and optionally downscale."""
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {img_path}")
    if scale != 1.0:
        h, w = img.shape[:2]
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return img


def create_sift():
    """Create a SIFT detector (opencv-contrib is required)."""
    if hasattr(cv2, "SIFT_create"):
        return cv2.SIFT_create()
    if hasattr(cv2, "xfeatures2d") and hasattr(cv2.xfeatures2d, "SIFT_create"):
        return cv2.xfeatures2d.SIFT_create()  # type: ignore
    raise RuntimeError("SIFT is unavailable. Install opencv-contrib-python.")


def match_features(
    img1_gray: np.ndarray,
    img2_gray: np.ndarray,
    ratio_thresh: float = 0.75,
) -> Tuple[List[cv2.KeyPoint], List[cv2.KeyPoint], List[cv2.DMatch]]:
    """Detect SIFT features and match them with Lowe's ratio test."""
    sift = create_sift()
    kp1, des1 = sift.detectAndCompute(img1_gray, None)
    kp2, des2 = sift.detectAndCompute(img2_gray, None)
    if des1 is None or des2 is None or len(des1) == 0 or len(des2) == 0:
        return kp1 or [], kp2 or [], []

    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    knn = matcher.knnMatch(des1, des2, k=2)

    good: List[cv2.DMatch] = []
    for pair in knn:
        if len(pair) == 2:
            m, n = pair
            if m.distance < ratio_thresh * n.distance:
                good.append(m)
    return kp1, kp2, good


def visualize_matches(
    img1_gray: np.ndarray,
    img2_gray: np.ndarray,
    kp1: Sequence[cv2.KeyPoint],
    kp2: Sequence[cv2.KeyPoint],
    matches: Sequence[cv2.DMatch],
    draw_limit: int = 50,
    out_path: Path | None = None,
):
    """Create a side-by-side match visualization and optionally save it."""
    vis = cv2.drawMatches(
        cv2.cvtColor(img1_gray, cv2.COLOR_GRAY2BGR),
        kp1,
        cv2.cvtColor(img2_gray, cv2.COLOR_GRAY2BGR),
        kp2,
        list(matches)[:draw_limit],
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), vis)
    return vis


def plot_sample_grid(image_paths: Sequence[Path], sample_size: int = 10):
    """Plot a random grid of frames (for quick sanity checks)."""
    if not image_paths:
        raise ValueError("No images provided for plotting.")
    sample_paths = random.sample(list(image_paths), min(sample_size, len(image_paths)))
    cols = 5
    rows = max(1, int(np.ceil(len(sample_paths) / cols)))
    plt.figure(figsize=(15, 3 * rows))
    for i, p in enumerate(sample_paths):
        img_bgr = cv2.imread(str(p))
        if img_bgr is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        plt.subplot(rows, cols, i + 1)
        plt.imshow(img_rgb)
        plt.title(p.name)
        plt.axis("off")
    plt.tight_layout()
    plt.show()


def match_consecutive_pairs(
    image_paths: Sequence[Path],
    pairs: int = 5,
    scale: float = 0.5,
    ratio_thresh: float = 0.75,
    draw_limit: int = 50,
    output_dir: Path | None = None,
) -> List[Tuple[int, int, int]]:
    """
    Match consecutive frame pairs.

    Returns:
        A list of tuples (idx1, idx2, num_good_matches).
    """
    results = []
    for i in range(min(pairs, len(image_paths) - 1)):
        j = i + 1
        img1 = preprocess_image(image_paths[i], scale=scale)
        img2 = preprocess_image(image_paths[j], scale=scale)
        kp1, kp2, good = match_features(img1, img2, ratio_thresh)
        results.append((i, j, len(good)))
        print(f"Pair {i}-{j}: {len(good)} good matches")

        vis_path = output_dir / f"matches_{i:03d}_{j:03d}.jpg" if output_dir else None
        vis = visualize_matches(img1, img2, kp1, kp2, good, draw_limit, vis_path)

        # Show inline for quick inspection
        plt.figure(figsize=(12, 6))
        plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        plt.title(f"Feature Matches: Frame {i} vs Frame {j}")
        plt.axis("off")
        plt.show()
    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Checkpoint 1: feature matching demo")
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=Path("data/frames"),
        help="Folder containing extracted frames (jpg/png).",
    )
    parser.add_argument("--pairs", type=int, default=5, help="Number of consecutive pairs to match.")
    parser.add_argument("--scale", type=float, default=0.5, help="Downscale factor for matching.")
    parser.add_argument("--ratio", type=float, default=0.75, help="Lowe ratio threshold.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/checkpoint1"),
        help="Where to save match visualizations.",
    )
    parser.add_argument(
        "--skip-grid",
        action="store_true",
        help="Skip plotting a random frame grid before matching.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    image_paths = discover_frames(args.image_dir)
    print(f"Found {len(image_paths)} frames in {args.image_dir}")

    if not args.skip_grid:
        plot_sample_grid(image_paths)

    match_consecutive_pairs(
        image_paths,
        pairs=args.pairs,
        scale=args.scale,
        ratio_thresh=args.ratio,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
