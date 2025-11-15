# matching.py
from typing import Dict, Tuple, List
import numpy as np

from models import Piece

# compat key: (idA, edgeA, idB, edgeB)
CompatKey = Tuple[int, int, int, int]

def flip_edge_feature(feature: np.ndarray) -> np.ndarray:
    """
    Flip a 1D edge feature so it corresponds to the reversed direction
    (needed when matching left of one piece to right of another).
    """
    # TODO: if feature encodes samples along the edge in order,
    # flipping is just reversing that order.
    return feature[::-1].copy()

def edge_distance(f1: np.ndarray, f2: np.ndarray) -> float:
    """
    Compute a distance between two edge feature vectors.
    Example: mean squared error.
    """
    # TODO: implement MSE or L2
    return float(np.mean((f1 - f2) ** 2))

def build_compatibility_matrix(pieces: List[Piece]) -> Dict[CompatKey, float]:
    """
    Compute compatibility scores for every ordered edge pair of distinct pieces.
    Lower distance = better match.
    Returns a dict mapping (idA, edgeA, idB, edgeB) -> distance.
    """
    compat: Dict[CompatKey, float] = {}

    for i, pA in enumerate(pieces):
        for j, pB in enumerate(pieces):
            if i == j:
                continue

            for edgeA in range(4):
                fA = pA.edge_features[edgeA]
                if fA is None:
                    continue

                for edgeB in range(4):
                    fB = pB.edge_features[edgeB]
                    if fB is None:
                        continue

                    fB_flipped = flip_edge_feature(fB)
                    dist = edge_distance(fA, fB_flipped)

                    key: CompatKey = (pA.id, edgeA, pB.id, edgeB)
                    compat[key] = dist

    return compat

def find_best_matches_for_edge(
    compat: Dict[CompatKey, float],
    piece_id: int,
    edge_index: int,
    k: int = 3
) -> List[Tuple[int, int, float]]:
    """
    Get top-k best matches for a given edge.
    Returns a list of (other_piece_id, other_edge_index, distance) sorted by distance.
    """
    candidates: List[Tuple[int, int, float]] = []

    for (idA, edgeA, idB, edgeB), dist in compat.items():
        if idA == piece_id and edgeA == edge_index:
            candidates.append((idB, edgeB, dist))

    candidates.sort(key=lambda x: x[2])
    return candidates[:k]

