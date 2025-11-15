# features.py
from typing import List
import numpy as np
import cv2

from models import Piece

# Edge index convention: 0=top, 1=right, 2=bottom, 3=left

def extract_edge_strip(image: np.ndarray, edge_index: int, strip_width: int = 5) -> np.ndarray:
    """
    Extract a thin strip along the specified edge of the piece image.
    Returns a (H x W x 3) (or W x strip_width x 3) subimage depending on design.
    """
    h, w, _ = image.shape

    if edge_index == 0:  # top
        strip = image[0:strip_width, :, :]
    elif edge_index == 1:  # right
        strip = image[:, w-strip_width:w, :]
    elif edge_index == 2:  # bottom
        strip = image[h-strip_width:h, :, :]
    else:  # left
        strip = image[:, 0:strip_width, :]

    return strip

def build_edge_feature(strip: np.ndarray, num_samples: int = 64) -> np.ndarray:
    """
    Build a 1D feature vector from an edge strip.
    Example: downsample along the long direction and average over the strip thickness.
    Returns a 1D numpy array.
    """
    # TODO: implement:
    # 1. Resize strip to (num_samples x strip_width)
    # 2. Average across strip_width dimension
    # 3. Flatten RGB into a 1D vector
    feature = np.zeros(num_samples * 3, dtype=np.float32)
    return feature

def compute_piece_edge_features(piece: Piece, num_samples: int = 64, strip_width: int = 5) -> None:
    """
    Compute and store edge feature vectors for all 4 edges of a piece.
    Modifies piece.edge_features in-place.
    """
    for edge_index in range(4):
        strip = extract_edge_strip(piece.image, edge_index, strip_width=strip_width)
        feature = build_edge_feature(strip, num_samples=num_samples)
        piece.edge_features[edge_index] = feature

def mark_border_edges(pieces: List[Piece], compat_threshold: float) -> None:
    """
    Optional: mark which edges are likely outer borders.
    For each edge, if even the best match distance is > threshold, mark as border.
    This requires you to have compatibility distances computed (can be done later).
    Here you can either:
      - leave as placeholder, or
      - call into matching functions once implemented.
    """
    # TODO: implement after compatibility matrix exists.
    pass

def compute_all_features(pieces: List[Piece]) -> None:
    """
    Convenience function to compute edge features for all pieces.
    """
    for p in pieces:
        compute_piece_edge_features(p)

