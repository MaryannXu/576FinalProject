# solver.py
from typing import Dict, Tuple, List, Set, Optional
import math

from models import Board, Piece
from matching import CompatKey

# Neighbor directions for grid
# For each cell we care about constraints from top and left
TOP = (-1, 0)
LEFT = (0, -1)

# Edge index mapping depending on orientation
# orientation 0: edges = [top, right, bottom, left]
# orientation 90, 180, 270: rotated accordingly.

def rotated_edge_index(edge_index: int, orientation: int) -> int:
    """
    Given an original edge index and orientation (0,90,180,270),
    return which physical side that edge maps to.
    Edge indices: 0=top, 1=right, 2=bottom, 3=left
    Rotation: 0°=no change, 90°=CW, 180°=flip, 270°=CCW
    """
    # Normalize orientation to 0, 90, 180, 270
    orientation = orientation % 360
    if orientation < 0:
        orientation += 360
    
    # Map rotation to number of 90-degree clockwise turns
    turns = orientation // 90
    
    # Rotate edge index clockwise
    # 0 (top) -> 1 (right) -> 2 (bottom) -> 3 (left) -> 0 (top)
    rotated = (edge_index + turns) % 4
    
    return rotated

def neighbor_edge_indices() -> Dict[Tuple[int, int], Tuple[int, int]]:
    """
    Returns which edges need to match for neighbors:
      - For LEFT neighbor: right edge of left piece matches left edge of current.
      - For TOP neighbor: bottom edge of top piece matches top edge of current.
    """
    return {
        LEFT: (1, 3),  # (neighbor_edge, current_edge)
        TOP: (2, 0)
    }

def compute_cell_order(rows: int, cols: int) -> List[Tuple[int, int]]:
    """
    Return an ordered list of (row, col) positions to fill.
    Example: row-major order.
    """
    order = []
    for r in range(rows):
        for c in range(cols):
            order.append((r, c))
    return order

def is_compatible_with_neighbors(
    board: Board,
    row: int,
    col: int,
    piece_id: int,
    orientation: int,
    compat: Dict[CompatKey, float],
    pieces: Dict[int, Piece],
    max_dist: float = math.inf
) -> bool:
    """
    Check if placing (piece_id, orientation) at (row,col) is compatible
    with already placed neighbors (top and left).
    """
    neigh_map = neighbor_edge_indices()

    for drow, dcol in [TOP, LEFT]:
        nr, nc = row + drow, col + dcol
        if nr < 0 or nc < 0:
            continue

        neighbor = board.grid[nr][nc]
        if neighbor is None:
            continue

        neigh_piece_id, neigh_orientation = neighbor
        # Edge indices in their oriented forms
        neigh_edge, cur_edge = neigh_map[(drow, dcol)]

        # TODO: adjust with rotated_edge_index for real implementation
        neigh_edge_idx = rotated_edge_index(neigh_edge, neigh_orientation)
        cur_edge_idx = rotated_edge_index(cur_edge, orientation)

        key: CompatKey = (neigh_piece_id, neigh_edge_idx, piece_id, cur_edge_idx)
        if key not in compat:
            return False

        if compat[key] > max_dist:
            return False

    return True

def solve_puzzle(
    pieces_list: List[Piece],
    rows: int,
    cols: int,
    compat: Dict[CompatKey, float]
) -> Optional[Board]:
    """
    High-level DFS/backtracking solver.
    Returns a filled Board if solution found, else None.
    """
    # Map id -> Piece for quick lookup
    pieces = {p.id: p for p in pieces_list}
    board = Board(rows, cols)
    used: Set[int] = set()
    order = compute_cell_order(rows, cols)

    def backtrack(idx: int) -> bool:
        if idx == len(order):
            return True  # all cells filled

        r, c = order[idx]

        for piece_id, piece in pieces.items():
            if piece_id in used:
                continue

            # Try all 4 orientations
            for orientation in [0, 90, 180, 270]:
                if not is_compatible_with_neighbors(
                    board, r, c, piece_id, orientation, compat, pieces
                ):
                    continue

                # Place piece
                board.place_piece(r, c, piece_id, orientation)
                used.add(piece_id)

                # Recurse
                if backtrack(idx + 1):
                    return True

                # Undo
                board.remove_piece(r, c)
                used.remove(piece_id)

        return False

    success = backtrack(0)
    if success:
        return board
    return None

