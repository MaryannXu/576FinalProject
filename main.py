# main.py
from typing import List
import cv2
import os
import math

from models import Piece
from segmentation import load_big_image, segment_pieces_from_canvas, load_pieces_from_directory
from features import compute_all_features
from matching import build_compatibility_matrix
from solver import solve_puzzle
from animation import assign_final_poses, animate_solution

def main():
    # Try to load pieces from images directory first
    images_dir = "images"
    pieces: List[Piece] = []
    
    if os.path.exists(images_dir):
        # Check if there are individual piece images
        image_files = [f for f in os.listdir(images_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
        
        if len(image_files) > 0:
            print(f"Loading {len(image_files)} pieces from {images_dir} directory...")
            pieces = load_pieces_from_directory(images_dir)
            print(f"Loaded {len(pieces)} pieces")
        else:
            # Try to load a big image and segment it
            big_image_path = os.path.join(images_dir, "starry_night.png")
            if os.path.exists(big_image_path):
                print(f"Loading and segmenting image: {big_image_path}")
                big_img = load_big_image(big_image_path)
                pieces = segment_pieces_from_canvas(big_img)
                print(f"Segmented {len(pieces)} pieces")
            else:
                print(f"No images found in {images_dir} directory")
                return
    else:
        # Fallback: try data directory
        big_image_path = "data/input_big_image.png"
        if os.path.exists(big_image_path):
            print(f"Loading and segmenting image: {big_image_path}")
            big_img = load_big_image(big_image_path)
            pieces = segment_pieces_from_canvas(big_img)
            print(f"Segmented {len(pieces)} pieces")
        else:
            print("No images found. Please place images in 'images' directory or 'data/input_big_image.png'")
            return
    
    if len(pieces) == 0:
        print("No pieces found. Exiting.")
        return
    
    # Determine puzzle dimensions
    num_pieces = len(pieces)
    # Try to find a square grid that fits
    cols = int(math.sqrt(num_pieces))
    while num_pieces % cols != 0:
        cols -= 1
    rows = num_pieces // cols
    
    print(f"Puzzle dimensions: {rows} x {cols} ({num_pieces} pieces)")

    # 3. Compute edge features
    print("Computing edge features...")
    compute_all_features(pieces)

    # 4. Build compatibility matrix
    print("Building compatibility matrix...")
    compat = build_compatibility_matrix(pieces)
    print(f"Computed {len(compat)} compatibility scores")

    # 5. Solve puzzle
    print("Solving puzzle...")
    # Calculate a reasonable distance threshold (median of all distances)
    if len(compat) > 0:
        distances = list(compat.values())
        distances.sort()
        median_dist = distances[len(distances) // 2]
        # Use a threshold that's a bit higher than median to allow flexibility
        max_dist = median_dist * 2.0
        print(f"Using compatibility threshold: {max_dist:.2f}")
    else:
        max_dist = float('inf')
    
    # Update solver to use threshold
    board = solve_puzzle_with_threshold(pieces, rows, cols, compat, max_dist)
    if board is None:
        print("No solution found. Try adjusting the compatibility threshold.")
        return

    print("Solution found!")

    # 6. Prepare final poses and animate
    # Calculate cell size based on piece dimensions
    if len(pieces) > 0:
        avg_h, avg_w = pieces[0].image.shape[:2]
        for p in pieces[1:]:
            h, w = p.image.shape[:2]
            avg_h = (avg_h + h) / 2
            avg_w = (avg_w + w) / 2
        cell_size = int(max(avg_h, avg_w))
    else:
        cell_size = 100
    
    canvas_size = max(cell_size * cols, cell_size * rows)
    
    print(f"Creating animation (canvas size: {canvas_size}x{canvas_size})...")
    assign_final_poses(pieces, board, cell_size=cell_size)
    animate_solution(
        pieces,
        board,
        canvas_size=canvas_size,
        num_frames=90,
        output_path="output_solution.mp4"
    )
    print("Animation saved to output_solution.mp4")

def solve_puzzle_with_threshold(pieces_list, rows, cols, compat, max_dist):
    """Wrapper to solve puzzle with distance threshold."""
    from solver import solve_puzzle
    # Modify the solver to accept max_dist parameter
    # For now, we'll use a modified version
    from solver import Board
    from typing import Set, Optional, Dict
    from matching import CompatKey
    
    pieces = {p.id: p for p in pieces_list}
    board = Board(rows, cols)
    used: Set[int] = set()
    
    # Compute cell order
    order = []
    for r in range(rows):
        for c in range(cols):
            order.append((r, c))
    
    def is_compatible_with_neighbors(row, col, piece_id, orientation):
        from solver import TOP, LEFT, neighbor_edge_indices, rotated_edge_index
        neigh_map = neighbor_edge_indices()
        
        for drow, dcol in [TOP, LEFT]:
            nr, nc = row + drow, col + dcol
            if nr < 0 or nc < 0:
                continue
            
            neighbor = board.grid[nr][nc]
            if neighbor is None:
                continue
            
            neigh_piece_id, neigh_orientation = neighbor
            neigh_edge, cur_edge = neigh_map[(drow, dcol)]
            
            neigh_edge_idx = rotated_edge_index(neigh_edge, neigh_orientation)
            cur_edge_idx = rotated_edge_index(cur_edge, orientation)
            
            key: CompatKey = (neigh_piece_id, neigh_edge_idx, piece_id, cur_edge_idx)
            if key not in compat:
                return False
            
            if compat[key] > max_dist:
                return False
        
        return True
    
    def backtrack(idx: int) -> bool:
        if idx == len(order):
            return True
        
        r, c = order[idx]
        
        for piece_id, piece in pieces.items():
            if piece_id in used:
                continue
            
            for orientation in [0, 90, 180, 270]:
                if not is_compatible_with_neighbors(r, c, piece_id, orientation):
                    continue
                
                board.place_piece(r, c, piece_id, orientation)
                used.add(piece_id)
                
                if backtrack(idx + 1):
                    return True
                
                board.remove_piece(r, c)
                used.remove(piece_id)
        
        return False
    
    success = backtrack(0)
    if success:
        return board
    return None

if __name__ == "__main__":
    main()

