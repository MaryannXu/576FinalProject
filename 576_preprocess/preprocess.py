#!/usr/bin/env python3

import os
import sys
import math
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

# Target normalized piece size (square)
TARGET_SIZE = 128

# Number of pixels from the border to use when extracting edge features
EDGE_STRIP_WIDTH = 4

# Number of histogram bins per color channel for features
COLOR_HIST_BINS = 8

def read_planar_rgb(path):
    """
    Read a planar RGB file:
        [R plane][G plane][B plane]

    Returns:
        img (H, W, 3) uint8 numpy array
    """
    data = np.fromfile(path, dtype=np.uint8)
    if data.size % 3 != 0:
        raise ValueError("File size is not divisible by 3, not a planar RGB image?")

    num_pixels = data.size // 3
    side = int(round(math.sqrt(num_pixels)))
    if side * side != num_pixels:
        raise ValueError(
            f"Image is not square: {num_pixels} pixels cannot form side×side."
        )

    H = W = side
    R = data[0 : H * W].reshape((H, W))
    G = data[H * W : 2 * H * W].reshape((H, W))
    B = data[2 * H * W : 3 * H * W].reshape((H, W))

    img = np.stack([R, G, B], axis=2)  # (H, W, 3)
    return img

def connected_components(mask):
    """
    Simple 4-connected component labeling

    mask: (H, W) bool array, True = foreground
    Returns:
        labels: (H, W) int32 array, 0 = background, 1..num_labels = components
        num_labels: number of components
    """
    H, W = mask.shape
    labels = np.zeros((H, W), dtype=np.int32)
    current_label = 0

    # 4-neighborhood (up, down, left, right)
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for y in range(H):
        for x in range(W):
            if not mask[y, x]:
                continue
            if labels[y, x] != 0:
                continue

            current_label += 1
            stack = [(y, x)]
            labels[y, x] = current_label

            while stack:
                cy, cx = stack.pop()
                for dy, dx in neighbors:
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < H and 0 <= nx < W:
                        if mask[ny, nx] and labels[ny, nx] == 0:
                            labels[ny, nx] = current_label
                            stack.append((ny, nx))

    return labels, current_label


def find_piece_bboxes(img):
    """
    Given an RGB image with black background and puzzle pieces,
    return bounding boxes for each connected piece.

    Each bbox is (x0, y0, x1, y1), inclusive coordinates.
    """
    bg = (img[:, :, 0] == 0) & (img[:, :, 1] == 0) & (img[:, :, 2] == 0)
    mask = ~bg  # foreground: any non-black pixel

    # Use our own connected components instead of scipy.ndimage.label
    labels, num = connected_components(mask)

    bboxes = []
    for label_id in range(1, num + 1):
        ys, xs = np.where(labels == label_id)
        if ys.size == 0:
            continue

        x0, x1 = xs.min(), xs.max()
        y0, y1 = ys.min(), ys.max()
        bboxes.append((x0, y0, x1, y1))

    # optional: sort for deterministic order
    bboxes.sort(key=lambda b: (b[1], b[0]))
    return bboxes


def crop_piece_with_mask(img, bbox):
    """
    Crop a piece from the full image and build a foreground mask.

    Returns:
        piece_rgb: (H, W, 3) uint8
        mask:      (H, W) bool
    """
    x0, y0, x1, y1 = bbox
    piece_rgb = img[y0 : y1 + 1, x0 : x1 + 1, :]
    mask = np.any(piece_rgb != 0, axis=2)  # True where not black
    return piece_rgb, mask


def make_rgba(piece_rgb, mask):
    """
    Convert RGB + mask to RGBA, with alpha 255 on foreground, 0 on background.
    """
    alpha = (mask * 255).astype(np.uint8)
    rgba = np.dstack([piece_rgb, alpha])
    return rgba

def normalize_orientation(rgba):
    """
    Rotate the piece so that its main axis is horizontal.
    Uses PCA on foreground pixels to estimate orientation.
    Returns:
        rotated_rgba: (H2, W2, 4)
        rotated_mask: (H2, W2) bool
    """
    # Mask is alpha > 0
    alpha = rgba[:, :, 3]
    mask = alpha > 0
    ys, xs = np.where(mask)

    # If somehow empty, just return as is
    if ys.size == 0:
        return rgba, mask

    # Centered coordinates of foreground
    xs_centered = xs - xs.mean()
    ys_centered = ys - ys.mean()
    coords = np.stack([xs_centered, ys_centered], axis=0)  # shape (2, N)

    # Covariance and eigenvectors
    cov = np.cov(coords)
    eigvals, eigvecs = np.linalg.eigh(cov)
    principal = eigvecs[:, np.argmax(eigvals)]
    angle_rad = math.atan2(principal[1], principal[0])
    angle_deg = math.degrees(angle_rad)

    # Rotate so principal axis is horizontal
    pil_img = Image.fromarray(rgba, mode="RGBA")
    rotated = pil_img.rotate(-angle_deg, expand=True, fillcolor=(0, 0, 0, 0))
    rotated_rgba = np.array(rotated)
    rotated_mask = rotated_rgba[:, :, 3] > 0
    return rotated_rgba, rotated_mask

def tight_crop_rgba(rgba):
    """
    Crop RGBA image to bounding box of non-zero alpha.
    """
    alpha = rgba[:, :, 3]
    mask = alpha > 0
    ys, xs = np.where(mask)
    if ys.size == 0:
        return rgba  # nothing to crop
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()
    return rgba[y0 : y1 + 1, x0 : x1 + 1, :]


def resize_rgba_to_square(rgba, target_size=TARGET_SIZE):
    """
    Resize RGBA to a square (target_size × target_size).
    Keeps aspect ratio by simple resize (pieces are roughly square anyway).
    """
    pil_img = Image.fromarray(rgba, mode="RGBA")
    resized = pil_img.resize((target_size, target_size), Image.BILINEAR)
    return np.array(resized)

def edge_strip(mask, width=EDGE_STRIP_WIDTH):
    """
    Given a foreground mask (H, W), returns boolean masks
    for strips along top, right, bottom, left edges.
    """
    H, W = mask.shape
    w = min(width, H // 2, W // 2)  # be safe on tiny pieces

    top = np.zeros_like(mask)
    top[0:w, :] = True

    bottom = np.zeros_like(mask)
    bottom[H - w : H, :] = True

    left = np.zeros_like(mask)
    left[:, 0:w] = True

    right = np.zeros_like(mask)
    right[:, W - w : W] = True

    return {
        "top": top & mask,
        "right": right & mask,
        "bottom": bottom & mask,
        "left": left & mask,
    }


def color_hist_features(rgb_pixels):
    """
    Compute simple per-channel histograms for a set of RGB pixels.
    Returns a 3*COLOR_HIST_BINS-dim vector (numpy array).
    """
    if rgb_pixels.size == 0:
        return np.zeros(3 * COLOR_HIST_BINS, dtype=np.float32)

    feats = []
    for c in range(3):
        channel = rgb_pixels[:, c]
        hist, _ = np.histogram(
            channel, bins=COLOR_HIST_BINS, range=(0, 256), density=True
        )
        feats.append(hist.astype(np.float32))
    return np.concatenate(feats, axis=0)


def extract_edge_features(rgba):
    """
    Given a normalized RGBA piece, compute per-edge color histogram features.
    Returns:
        features: dict mapping edge name -> list[float]
    """
    rgb = rgba[:, :, :3]
    mask = rgba[:, :, 3] > 0

    strips = edge_strip(mask)
    edge_features = {}

    for edge_name, strip_mask in strips.items():
        ys, xs = np.where(strip_mask)
        pixels = rgb[ys, xs, :] if ys.size > 0 else np.empty((0, 3), dtype=np.uint8)
        feats = color_hist_features(pixels)
        edge_features[edge_name] = feats.tolist()

    return edge_features


def save_debug_image(img, bboxes, out_path):
    """
    Save a debug PNG with red rectangles drawn around each detected piece.
    """
    debug_img = Image.fromarray(img.copy())
    draw = ImageDraw.Draw(debug_img)

    for (x0, y0, x1, y1) in bboxes:
        draw.rectangle([x0, y0, x1, y1], outline=(255, 0, 0), width=3)

    debug_img.save(out_path)


def main():
    if len(sys.argv) != 2:
        print("Usage: python preprocess.py input.rgb")
        sys.exit(1)

    input_path = sys.argv[1]
    if not os.path.isfile(input_path):
        print(f"File not found: {input_path}")
        sys.exit(1)

    base_name = os.path.splitext(os.path.basename(input_path))[0]

    # 1. Read the planar RGB image
    img = read_planar_rgb(input_path)

    # 2. Find bounding boxes for each piece
    bboxes = find_piece_bboxes(img)
    print(f"Found {len(bboxes)} puzzle pieces.")

    # Where to save outputs
    pieces_dir = Path(f"{base_name}_pieces")
    pieces_dir.mkdir(parents=True, exist_ok=True)

    feature_records = []

    # 3. For each piece: crop, normalize orientation + size, extract features, save
    for idx, bbox in enumerate(bboxes):
        piece_rgb, mask = crop_piece_with_mask(img, bbox)
        rgba = make_rgba(piece_rgb, mask)

        # Normalize orientation
        rgba_rot, _ = normalize_orientation(rgba)

        # Tight crop and resize to standard size
        rgba_rot_crop = tight_crop_rgba(rgba_rot)
        rgba_norm = resize_rgba_to_square(rgba_rot_crop, TARGET_SIZE)

        # Extract per-edge features
        edge_feats = extract_edge_features(rgba_norm)

        # Save image
        piece_filename = pieces_dir / f"{base_name}_piece_{idx:02d}.png"
        Image.fromarray(rgba_norm, mode="RGBA").save(piece_filename)

        feature_records.append(
            {
                "index": idx,
                "image": str(piece_filename),
                "bbox": {
                    "x0": int(bbox[0]),
                    "y0": int(bbox[1]),
                    "x1": int(bbox[2]),
                    "y1": int(bbox[3]),
                },
                "edge_features": edge_feats,
            }
        )

    # 4. Save a debug image with rectangles around each piece
    debug_path = f"{base_name}_debug.png"
    save_debug_image(img, bboxes, debug_path)
    print(f"Saved debug image with bounding boxes: {debug_path}")

    # 5. Save all features to JSON (used later for matching)
    features_path = f"{base_name}_features.json"
    with open(features_path, "w") as f:
        json.dump(
            {
                "input_image": input_path,
                "target_size": TARGET_SIZE,
                "edge_strip_width": EDGE_STRIP_WIDTH,
                "color_hist_bins": COLOR_HIST_BINS,
                "pieces": feature_records,
            },
            f,
            indent=2,
        )

    print(f"Saved per-piece edge features to: {features_path}")
    print(f"Normalized pieces saved in: {pieces_dir}/")

if __name__ == "__main__":
    main()
