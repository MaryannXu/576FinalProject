#!/usr/bin/env python3
import json
import math
from pathlib import Path
from typing import Dict, Tuple, List, Optional, Set

import numpy as np

# ----------------- Constants & Types -----------------

EDGE_NAMES = ["top", "right", "bottom", "left"]
EDGE_INDEX = {name: i for i, name in enumerate(EDGE_NAMES)}

# (pieceA, edgeA, pieceB, edgeB)
CompatKey = Tuple[int, int, int, int]


# ----------------- Piece Class -----------------

class Piece:
    """
    Represents a single puzzle piece with per-edge histogram features.
    edge_features[0..3] correspond to top, right, bottom, left in canonical orientation.
    """

    def __init__(self, index: int, image_path: str, edge_features_dict: Dict[str, List[float]]):
        self.index = index
        self.image_path = image_path
        # Store features as numpy arrays for fast math
        self.edge_features: List[np.ndarray] = [None] * 4
        for name, feats in edge_features_dict.items():
            self.edge_features[EDGE_INDEX[name]] = np.array(feats, dtype=np.float32)

    def feature_for_edge(self, edge_idx: int) -> np.ndarray:
        return self.edge_features[edge_idx]


# ----------------- Load JSON -----------------

def load_pieces_from_json(json_path: str) -> List[Piece]:
    """
    Load pieces from a *_features.json file produced by preprocess.py.
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    pieces: List[Piece] = []
    for rec in data["pieces"]:
        idx = rec["index"]
        img_path = rec["image"]
        edge_feats = rec["edge_features"]  # dict: top/right/bottom/left -> list[float]
        pieces.append(Piece(idx, img_path, edge_feats))

    return pieces


# ----------------- Edge Compatibility -----------------

def edge_distance(f1: np.ndarray, f2: np.ndarray) -> float:
    """
    Distance between two 24-D edge feature vectors.
    Here we use mean squared error (MSE). Smaller = better match.
    """
    return float(np.mean((f1 - f2) ** 2))


def build_compatibility_matrix(pieces: List[Piece]) -> Dict[CompatKey, float]:
    """
    Build a compatibility matrix over all ordered pairs of edges of all pieces.

    compat[(idA, edgeA, idB, edgeB)] = distance between that edge pair.
    """
    compat: Dict[CompatKey, float] = {}

    for A in pieces:
        for B in pieces:
            if A.index == B.index:
                continue  # don't match a piece to itself
            for edgeA in range(4):
                fA = A.feature_for_edge(edgeA)
                for edgeB in range(4):
                    fB = B.feature_for_edge(edgeB)
                    d = edge_distance(fA, fB)
                    compat[(A.index, edgeA, B.index, edgeB)] = d

    return compat


def compute_max_dist_threshold(
    compat: Dict[CompatKey, float],
    percentile: float = 60.0,
) -> float:
    """
    Choose a global distance cutoff: we only accept matches whose distance is
    below this percentile of all edge distances.

    Higher percentile = more permissive, lower = stricter.
    """
    dists = np.array(list(compat.values()), dtype=np.float32)
    return float(np.percentile(dists, percentile))


# ----------------- Rotation Handling -----------------

def rotated_edge_index(edge_idx: int, orientation: int) -> int:
    """
    Map canonical edge index (0=top,1=right,2=bottom,3=left) to the edge
    that ends up in that position after rotating the piece by orientation degrees.

    orientation ∈ {0, 90, 180, 270}, clockwise.
    """
    steps = (orientation // 90) % 4
    # Rotating clockwise moves edges in order: top->right->bottom->left->top
    return (edge_idx - steps) % 4


# ----------------- Board & Search -----------------

TOP = (-1, 0)
LEFT = (0, -1)

# Map (neighbor relative position) -> (neighbor edge, current edge) in canonical indices
NEIGHBOR_EDGE_MAP = {
    LEFT: (1, 3),   # neighbor's right edge vs current left edge
    TOP:  (2, 0),   # neighbor's bottom edge vs current top edge
}


class Board:
    """
    Grid of (piece_index, orientation) or None.
    """

    def __init__(self, rows: int, cols: int):
        self.rows = rows
        self.cols = cols
        self.grid: List[List[Optional[Tuple[int, int]]]] = [
            [None for _ in range(cols)] for _ in range(rows)
        ]

    def place(self, r: int, c: int, piece_idx: int, orientation: int) -> None:
        self.grid[r][c] = (piece_idx, orientation)

    def remove(self, r: int, c: int) -> None:
        self.grid[r][c] = None

    def is_full(self) -> bool:
        return all(cell is not None for row in self.grid for cell in row)


def compute_cell_order(rows: int, cols: int) -> List[Tuple[int, int]]:
    """
    Simple row-major order: (0,0), (0,1), ..., (0,cols-1), (1,0), ...
    """
    return [(r, c) for r in range(rows) for c in range(cols)]


def placement_cost(
    board: Board,
    r: int,
    c: int,
    piece_idx: int,
    orientation: int,
    compat: Dict[CompatKey, float],
    max_dist: float
) -> Optional[float]:
    """
    Compute the local compatibility cost of placing (piece_idx, orientation)
    at cell (r, c), with respect to already-placed TOP/LEFT neighbors.

    - If any neighbor-edge pair exceeds max_dist => invalid placement => return None.
    - Otherwise return the sum of those distances (lower is better).
    """
    total = 0.0
    has_neighbor = False

    for drow, dcol in [TOP, LEFT]:
        nr, nc = r + drow, c + dcol
        if nr < 0 or nc < 0:
            continue  # out of bounds, no neighbor
        neighbor = board.grid[nr][nc]
        if neighbor is None:
            continue

        has_neighbor = True
        neigh_idx, neigh_ori = neighbor
        neigh_edge_raw, cur_edge_raw = NEIGHBOR_EDGE_MAP[(drow, dcol)]

        # Convert logical edges to canonical indices given orientation
        neigh_edge = rotated_edge_index(neigh_edge_raw, neigh_ori)
        cur_edge = rotated_edge_index(cur_edge_raw, orientation)

        key = (neigh_idx, neigh_edge, piece_idx, cur_edge)
        dist = compat.get(key, float("inf"))
        if dist > max_dist:
            return None  # too bad a match
        total += dist

    return total if has_neighbor else 0.0  # 0 cost if no neighbors yet (first cell)


def solve_layout(
    pieces: List[Piece],
    compat: Dict[CompatKey, float],
    rows: int,
    cols: int,
    max_dist: float,
    orientations: List[int],
) -> Optional[Board]:
    """
    Depth-first search with backtracking:
      - Place one piece per cell in row-major order.
      - For each cell, consider all unused pieces and all allowed orientations.
      - Use placement_cost to enforce reasonable matches.
      - Try candidates in order of increasing local cost.
    """
    board = Board(rows, cols)
    used: Set[int] = set()
    order = compute_cell_order(rows, cols)
    piece_ids = [p.index for p in pieces]

    def backtrack(pos: int) -> bool:
        if pos == len(order):
            return True  # filled all cells

        r, c = order[pos]

        # Build list of all valid candidates with their local cost
        candidates: List[Tuple[float, int, int]] = []  # (cost, piece_id, orientation)

        for pid in piece_ids:
            if pid in used:
                continue

            for ori in orientations:
                cost = placement_cost(board, r, c, pid, ori, compat, max_dist)
                if cost is None:
                    continue
                candidates.append((cost, pid, ori))

        if not candidates:
            return False  # dead end

        # Try locally best matches first
        candidates.sort(key=lambda x: x[0])

        for cost, pid, ori in candidates:
            board.place(r, c, pid, ori)
            used.add(pid)

            if backtrack(pos + 1):
                return True

            # backtrack
            board.remove(r, c)
            used.remove(pid)

        return False

    if backtrack(0):
        return board
    return None


# ----------------- Main -----------------

def main():
    import sys

    if len(sys.argv) != 2:
        print("Usage: python3 solve_puzzle.py <features.json>")
        sys.exit(1)

    json_path = sys.argv[1]
    if not Path(json_path).is_file():
        print(f"JSON file not found: {json_path}")
        sys.exit(1)

    # 1. Load pieces
    pieces = load_pieces_from_json(json_path)
    N = len(pieces)
    print(f"Loaded {N} pieces from {json_path}")

    # Assume square puzzle (e.g., 16 -> 4x4)
    side = int(round(math.sqrt(N)))
    if side * side != N:
        print(f"Warning: {N} is not a perfect square; using 1×{N} grid.")
        rows, cols = 1, N
    else:
        rows = cols = side

    # 2. Build compatibility matrix
    print("Building compatibility matrix...")
    compat = build_compatibility_matrix(pieces)
    print(f"{len(compat)} edge pair scores computed.")

    # 3. Choose distance threshold
    max_dist = compute_max_dist_threshold(compat, percentile=60.0)
    print(f"Using max_dist threshold = {max_dist:.6f}")

    # 4. Decide allowed orientations based on dataset
    name = Path(json_path).name.lower()
    if "translate" in name:
        # Pieces in translate set are already upright → don't rotate them
        orientations = [0]
    else:
        # Rotated dataset: allow 4 orientations
        orientations = [0, 90, 180, 270]
    print(f"Allowed orientations: {orientations}")

    # 5. Solve layout
    print("Solving layout...")
    board = solve_layout(pieces, compat, rows, cols, max_dist, orientations)

    if board is None:
        print("No solution found.")
        return

    print("Solution found! Final grid (piece_index, orientation):")
    for r in range(board.rows):
        row_repr = []
        for c in range(board.cols):
            row_repr.append(str(board.grid[r][c]))
        print("  ", "  ".join(row_repr))


if __name__ == "__main__":
    main()
