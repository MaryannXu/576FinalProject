# models.py
from typing import List, Optional, Tuple
import numpy as np

class Piece:
    """
    Represents a single puzzle piece.
    """
    def __init__(self, piece_id: int, image: np.ndarray):
        self.id: int = piece_id
        self.image: np.ndarray = image  # H x W x 3, uint8

        # Orientation in degrees: 0, 90, 180, 270
        self.orientation: int = 0

        # Edge features: [top, right, bottom, left]
        # Each will be a 1D numpy array or similar
        self.edge_features: List[Optional[np.ndarray]] = [None] * 4

        # True if this edge is likely on the outer border
        self.border_flags: List[bool] = [False] * 4

        # For animation: original and final positions/orientations
        self.start_center: Optional[Tuple[float, float]] = None
        self.start_angle: Optional[float] = None

        self.end_center: Optional[Tuple[float, float]] = None
        self.end_angle: Optional[float] = None


class Board:
    """
    Represents the final puzzle board as a grid of (piece_id, orientation).
    """
    def __init__(self, rows: int, cols: int):
        self.rows: int = rows
        self.cols: int = cols

        # Each cell: (piece_id, orientation) or None
        self.grid: List[List[Optional[Tuple[int, int]]]] = [
            [None for _ in range(cols)]
            for _ in range(rows)
        ]

    def place_piece(self, row: int, col: int, piece_id: int, orientation: int) -> None:
        self.grid[row][col] = (piece_id, orientation)

    def remove_piece(self, row: int, col: int) -> None:
        self.grid[row][col] = None

    def is_filled(self) -> bool:
        return all(cell is not None for row in self.grid for cell in row)

