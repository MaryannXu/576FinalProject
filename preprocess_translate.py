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
                # if the pixel is already labeled, skip it
                continue

            current_label += 1
            stack = [(y, x)]  # to run dfs
            labels[y, x] = current_label

            while stack:
                cy, cx = stack.pop()
                for dy, dx in neighbors:
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < H and 0 <= nx < W:
                        if mask[ny, nx] and labels[ny, nx] == 0:
                            labels[ny, nx] = current_label
                            stack.append((ny, nx))

    return labels, current_label  # labels is a 2d array and current_label is the number of components (should be 16 for starry night)


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

import cv2

def normalize_orientation(rgba):
    """
    Rotate the piece so that its bounding box is axis-aligned.
    Uses cv2.minAreaRect on the mask.
    Returns:
        rotated_rgba: (H2, W2, 4)
        rotated_mask: (H2, W2) bool
        angle_deg: float (degrees rotated to get to upright)
    """
    # Mask is alpha > 0
    alpha = rgba[:, :, 3]
    mask = (alpha > 0).astype(np.uint8) * 255
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return rgba, alpha > 0, 0.0
        
    # Find largest contour
    cnt = max(contours, key=cv2.contourArea)
    
    # Get min area rect
    rect = cv2.minAreaRect(cnt)
    (center, (w, h), angle) = rect
    
    # cv2.minAreaRect returns angle in [-90, 0).
    # We want to rotate the image so the sides are axis aligned.
    # The angle is the rotation of the rectangle.
    # If w < h, it might be "standing up" but rotated.
    # We want to rotate by 'angle' to align it?
    # Or maybe 'angle + 90'?
    
    # Let's just rotate by 'angle'.
    # If the piece ends up 90 degrees off, the solver handles it (0, 90, 180, 270).
    # We just want it axis-aligned.
    
    # However, minAreaRect angle definition varies by OpenCV version.
    # In 4.5+, it's in [0, 90]? Or [-90, 0]?
    # Let's assume standard behavior: it gives the angle of the first side.
    
    # If we rotate by 'angle', the rectangle becomes axis aligned.
    # But we need to rotate the IMAGE by 'angle'.
    # If rect angle is -30, it means the rect is rotated -30 (CW 30).
    # So we need to rotate the image by +30 (CCW 30) to fix it?
    # Or rotate by -30?
    
    # Let's try rotating by 'angle'.
    # If angle is positive (CW?), we rotate by -angle (CCW) to undo?
    # OpenCV rotation: positive is CCW.
    # minAreaRect angle: usually clockwise is positive? No, standard math is CCW positive.
    # But image y-axis is down.
    
    # Let's stick to: rotate by `angle`.
    # And if the resulting bbox is larger, try `angle + 90`.
    # Actually, minAreaRect gives the angle that the rect is rotated.
    # So we should rotate by `-angle` to align it with axes?
    # Let's try `angle`.
    
    # Wait, `minAreaRect` returns angle of the rectangle.
    # If we rotate the image by `angle`, does it align?
    # Usually `angle` is the angle of the width side with the horizontal.
    # So if we rotate by `-angle`, the width side becomes horizontal.
    
    angle_deg = angle
    
    # Handle the fact that we might want the longer side horizontal or vertical?
    # Puzzle pieces are square-ish. It doesn't matter.
    # The solver handles 90 degree increments.
    # We just need it to be axis aligned.
    
    # Rotate
    pil_img = Image.fromarray(rgba, mode="RGBA")
    # PIL rotate is CCW.
    # If angle_deg is the angle of the rect (CW?), then we might need to be careful.
    # In OpenCV 4.x, angle is in [0, 90].
    # Let's just use the angle.
    
    # If we rotate by `angle_deg`, we might align it.
    rotated = pil_img.rotate(angle_deg, expand=True, fillcolor=(0, 0, 0, 0))
    
    # Check if we improved the bbox area?
    # No, minAreaRect is the best fit.
    # So rotating by -angle (or angle) should align it.
    
    rotated_rgba = np.array(rotated)
    rotated_mask = rotated_rgba[:, :, 3] > 0
    
    # We return -angle_deg because the caller (animation) expects 'angle_deg' 
    # to be the value such that `original = normalized.rotate(angle_deg)`.
    # Here `normalized = original.rotate(angle_deg)`.
    # So `original = normalized.rotate(-angle_deg)`.
    # So we should return `-angle_deg` as the "initial_rotation"?
    # Wait.
    # `preprocess.py` (old): `rotated = pil_img.rotate(-angle_deg)`
    # And returned `angle_deg`.
    # So `normalized = original.rotate(-angle_deg)`.
    # `original = normalized.rotate(angle_deg)`.
    # So `start_angle` was `angle_deg`.
    
    # Here: `rotated = pil_img.rotate(angle_deg)`.
    # So `normalized = original.rotate(angle_deg)`.
    # `original = normalized.rotate(-angle_deg)`.
    # So we should return `-angle_deg`.
    
    return rotated_rgba, rotated_mask, -angle_deg

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
        print("Usage: python preprocess.py <filename.rgb>")
        sys.exit(1)

    input_path = sys.argv[1]
    if not os.path.exists(input_path):
        print(f"Error: File {input_path} not found.")
        sys.exit(1)

    base_name = os.path.splitext(os.path.basename(input_path))[0]
    
    # Setup output directories
    output_root = Path("output") / base_name
    pieces_dir = output_root / "pieces"
    
    # Create directories
    pieces_dir.mkdir(parents=True, exist_ok=True)

    print(f"Processing {input_path}...")
    print(f"Output directory: {output_root}")

    # 1. Read the planar RGB image
    img = read_planar_rgb(input_path)

    # 2. Find bounding boxes for each piece
    bboxes = find_piece_bboxes(img)
    print(f"Found {len(bboxes)} puzzle pieces.")

    feature_records = []

    # 3. For each piece: crop, normalize orientation + size, extract features, save
    for idx, bbox in enumerate(bboxes):
        piece_rgb, mask = crop_piece_with_mask(img, bbox)
        rgba = make_rgba(piece_rgb, mask)

        # For translation only, we assume pieces are already upright.
        # Just crop.
        cropped_rgba = tight_crop_rgba(rgba)
        angle_deg = 0.0
        
        # Save debug image for the first few
        if idx < 5:
            debug_path = pieces_dir / f"debug_piece_{idx:02d}.png"
            Image.fromarray(cropped_rgba).save(debug_path)

        rgba_norm = resize_rgba_to_square(cropped_rgba, TARGET_SIZE)

        # Extract per-edge features
        edge_feats = extract_edge_features(rgba_norm)

        # Save image
        piece_filename = pieces_dir / f"piece_{idx:02d}.png"
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
                "initial_rotation": angle_deg,
                "edge_features": edge_feats,
            }
        )

    # 4. Save a debug image with rectangles around each piece
    debug_path = output_root / "bboxes_debug.png"
    save_debug_image(img, bboxes, debug_path)
    print(f"Saved debug image: {debug_path}")

    # 5. Save all features to JSON (used later for matching)
    features_path = output_root / "features.json"
    with open(features_path, "w") as f:
        json.dump(
            {
                "input_image": str(input_path),
                "target_size": TARGET_SIZE,
                "edge_strip_width": EDGE_STRIP_WIDTH,
                "color_hist_bins": COLOR_HIST_BINS,
                "pieces": feature_records,
            },
            f,
            indent=2,
        )

    print(f"Saved features to: {features_path}")
    print("Preprocessing complete.")

if __name__ == "__main__":
    main()
