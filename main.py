# main.py
from typing import List
import cv2

from models import Piece
from segmentation import load_big_image, segment_pieces_from_canvas
from features import compute_all_features
from matching import build_compatibility_matrix
from solver import solve_puzzle
from animation import assign_final_poses, animate_solution

def main():
    # 1. Load big scattered image
    big_img = load_big_image("data/input_big_image.png")

    # 2. Segment into pieces
    pieces: List[Piece] = segment_pieces_from_canvas(big_img)
    print(f"Segmented {len(pieces)} pieces")

    # For early testing you can bypass segmentation and instead:
    #   - manually load small tile images
    #   - construct Piece objects
    # and skip directly to steps 3–6.

    # 3. Compute edge features
    compute_all_features(pieces)

    # 4. Build compatibility matrix
    compat = build_compatibility_matrix(pieces)

    # 5. Solve puzzle (assuming 4x4 for now)
    rows, cols = 4, 4
    board = solve_puzzle(pieces, rows, cols, compat)
    if board is None:
        print("No solution found.")
        return

    print("Solution found!")

    # 6. Prepare final poses and animate
    cell_size = 100        # choose based on your desired output resolution
    canvas_size = cell_size * cols  # assume square board

    assign_final_poses(pieces, board, cell_size=cell_size)
    animate_solution(
        pieces,
        board,
        canvas_size=canvas_size,
        num_frames=90,
        output_path="output_solution.mp4"
    )

if __name__ == "__main__":
    main()

