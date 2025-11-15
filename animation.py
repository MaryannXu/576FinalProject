# animation.py
from typing import List
import numpy as np
import cv2

from models import Piece, Board

def assign_final_poses(pieces: List[Piece], board: Board, cell_size: int) -> None:
    """
    Given a solved board, assign each piece its final center position and angle.
    cell_size: the width/height of each tile in pixels.
    """
    id_to_piece = {p.id: p for p in pieces}

    for r in range(board.rows):
        for c in range(board.cols):
            cell = board.grid[r][c]
            if cell is None:
                continue
            piece_id, orientation = cell
            piece = id_to_piece[piece_id]

            # Compute final center in some canvas coords
            center_x = (c + 0.5) * cell_size
            center_y = (r + 0.5) * cell_size

            piece.end_center = (center_x, center_y)
            piece.end_angle = float(orientation)

def lerp(a: float, b: float, t: float) -> float:
    return (1.0 - t) * a + t * b

def render_frame(pieces: List[Piece], t: float, canvas_size: int) -> np.ndarray:
    """
    Render a single animation frame for time t in [0,1].
    Uses linear interpolation between start and end poses.
    """
    frame = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)

    for piece in pieces:
        if piece.start_center is None or piece.end_center is None:
            continue
        if piece.start_angle is None or piece.end_angle is None:
            continue

        cx = lerp(piece.start_center[0], piece.end_center[0], t)
        cy = lerp(piece.start_center[1], piece.end_center[1], t)
        angle = lerp(piece.start_angle, piece.end_angle, t)

        # TODO:
        # 1. Rotate piece.image by 'angle'
        # 2. Paste onto 'frame' with center at (cx, cy)
        # You can use cv2.getRotationMatrix2D + cv2.warpAffine,
        # then place it on the canvas.

    return frame

def animate_solution(
    pieces: List[Piece],
    board: Board,
    canvas_size: int,
    num_frames: int,
    output_path: str
) -> None:
    """
    Create an animation from scattered configuration to solved configuration.
    Saves as a video using OpenCV VideoWriter.
    """
    # Assume assign_final_poses already called.
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 30
    writer = cv2.VideoWriter(output_path, fourcc, fps, (canvas_size, canvas_size))

    for i in range(num_frames):
        t = i / (num_frames - 1)
        frame = render_frame(pieces, t, canvas_size)
        writer.write(frame)

    writer.release()

