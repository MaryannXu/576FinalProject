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
    frame = np.zeros((canvas_size, canvas_size, 4), dtype=np.uint8)

    for piece in pieces:
        if piece.start_center is None or piece.end_center is None:
            continue
        if piece.start_angle is None or piece.end_angle is None:
            continue

        cx = lerp(piece.start_center[0], piece.end_center[0], t)
        cy = lerp(piece.start_center[1], piece.end_center[1], t)
        angle = lerp(piece.start_angle, piece.end_angle, t)

        # Get piece image dimensions
        h, w = piece.image.shape[:2]
        center_piece = (w // 2, h // 2)
        
        # Get rotation matrix
        M = cv2.getRotationMatrix2D(center_piece, angle, 1.0)
        
        # Calculate new dimensions after rotation
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))
        
        # Adjust rotation matrix for new center
        M[0, 2] += (new_w / 2) - center_piece[0]
        M[1, 2] += (new_h / 2) - center_piece[1]
        
        # Rotate the piece image
        rotated = cv2.warpAffine(piece.image, M, (new_w, new_h), 
                                 flags=cv2.INTER_LINEAR, 
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=(0, 0, 0))
        
        # Calculate position to paste on canvas
        x1 = int(cx - new_w // 2)
        y1 = int(cy - new_h // 2)
        x2 = x1 + new_w
        y2 = y1 + new_h
        
        # Clip to canvas bounds
        src_x1 = max(0, -x1)
        src_y1 = max(0, -y1)
        src_x2 = min(new_w, canvas_size - x1)
        src_y2 = min(new_h, canvas_size - y1)
        
        dst_x1 = max(0, x1)
        dst_y1 = max(0, y1)
        dst_x2 = min(canvas_size, x2)
        dst_y2 = min(canvas_size, y2)
        
        # Paste rotated piece onto frame
        if dst_x2 > dst_x1 and dst_y2 > dst_y1:
            frame[dst_y1:dst_y2, dst_x1:dst_x2] = rotated[src_y1:src_y2, src_x1:src_x2]

    return frame

import subprocess
import shutil
import os

def animate_solution(
    pieces: List[Piece],
    board: Board,
    canvas_size: int,
    num_frames: int,
    output_path: str
) -> None:
    """
    Create an animation using ffmpeg.
    """
    # Create temp dir for frames
    frames_dir = "temp_frames"
    if os.path.exists(frames_dir):
        shutil.rmtree(frames_dir)
    os.makedirs(frames_dir)

    print(f"Rendering {num_frames} frames to {frames_dir}...")
    for i in range(num_frames):
        t = i / (num_frames - 1)
        frame = render_frame(pieces, t, canvas_size)
        # Convert RGBA to BGR for OpenCV saving if needed, but we used RGBA in render_frame
        # cv2.imwrite expects BGR or BGRA. 
        # render_frame returns numpy array. If we constructed it as RGBA (which we did),
        # we should convert to BGRA for cv2.imwrite to get correct colors.
        # But wait, render_frame creates: frame = np.zeros((..., 4), dtype=np.uint8)
        # And we paste pieces. Pieces were loaded as RGBA.
        # So frame is RGBA.
        # cv2.imwrite expects BGRA.
        frame_bgra = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGRA)
        
        frame_path = os.path.join(frames_dir, f"frame_{i:04d}.png")
        cv2.imwrite(frame_path, frame_bgra)

    print("Encoding video with ffmpeg...")
    # Ensure output ends with .mp4
    if not output_path.endswith(".mp4"):
        output_path = os.path.splitext(output_path)[0] + ".mp4"

    cmd = [
        "ffmpeg",
        "-y", # Overwrite
        "-framerate", "30",
        "-i", f"{frames_dir}/frame_%04d.png",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        output_path
    ]
    
    subprocess.run(cmd, check=True)
    
    # Cleanup
    shutil.rmtree(frames_dir)
    print(f"Animation saved to {output_path}")

