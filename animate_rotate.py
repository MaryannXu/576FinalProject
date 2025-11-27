import json
import os
import sys
import math
import numpy as np
from PIL import Image
from typing import List, Dict, Tuple

# Add current directory to path
sys.path.append(os.getcwd())

from models import Piece, Board
from animation import assign_final_poses, animate_solution
import recombine_rotate as solver

def load_pieces_with_start_pos(json_path: str) -> Tuple[List[Piece], int]:
    """
    Load pieces and also set their start_center and start_angle based on bbox.
    """
    # Reuse the loading logic from v2 but we need to access bbox which v2 might not expose directly
    # So we will reimplement a slight variation or just load raw json here.
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    target_size = data['target_size']
    pieces_data = data['pieces']
    
    loaded_pieces = []
    base_dir = os.path.dirname(json_path)
    
    # We need to load the pieces exactly as v2 does so the IDs match
    # v2 loads them in order of the list
    
    for p_data in pieces_data:
        idx = p_data['index']
        img_rel_path = p_data['image']
        
        # Check if path is valid as is (relative to CWD)
        if os.path.exists(img_rel_path):
            img_path = img_rel_path
        else:
            # Fallback to relative to json file
            img_path = os.path.join(base_dir, img_rel_path)
        
        if not os.path.exists(img_path):
            print(f"Warning: Image not found at {img_path}")
            continue
            
        pil_img = Image.open(img_path).convert('RGBA')
        if pil_img.size != (target_size, target_size):
            pil_img = pil_img.resize((target_size, target_size))
            
        img_arr = np.array(pil_img)
        piece = Piece(idx, img_arr)
        
        # Extract features for solver
        rgb = img_arr[:, :, :3].astype(np.float32)
        piece.edge_features[0] = rgb[0, :, :]
        piece.edge_features[1] = rgb[:, -1, :]
        piece.edge_features[2] = rgb[-1, :, :]
        piece.edge_features[3] = rgb[:, 0, :]
        
        # Set start position from bbox
        bbox = p_data['bbox']
        x0, y0, x1, y1 = bbox['x0'], bbox['y0'], bbox['x1'], bbox['y1']
        
        # Center of the piece in the original image
        cx = (x0 + x1) / 2.0
        cy = (y0 + y1) / 2.0
        
        piece.start_center = (cx, cy)
        piece.start_angle = p_data.get('initial_rotation', 0.0)
        
        loaded_pieces.append(piece)
        
    return loaded_pieces, target_size

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 animate_translated.py <image_name_or_features_json>")
        return

    output_dir = sys.argv[1]
    
    # Assume the argument is the directory containing features.json
    json_path = os.path.join(output_dir, "features.json")
    
    # Derive base name from directory name for output filename
    base_name = os.path.basename(output_dir.rstrip(os.sep))

    if not os.path.exists(json_path):
        print(f"Error: {json_path} not found.")
        return

    print(f"Loading pieces with start positions from {json_path}...")
    pieces, target_size = load_pieces_with_start_pos(json_path)
    
    n = len(pieces)
    side = int(math.sqrt(n))
    rows = cols = side
    
    print("Solving puzzle...")
    # compat = solver.build_ssd_compatibility(pieces) # Internal now
    board = solver.solve_puzzle_constrained(pieces, rows, cols, {})
    
    if not board:
        print("Failed to solve puzzle.")
        return
        
    print("Solution found. Generating animation...")
    
    # Assign final poses
    assign_final_poses(pieces, board, cell_size=target_size)
    
    # Let's use 1000x1000 canvas
    CANVAS_SIZE = 1000
    
    # Shift final positions to center of 1000x1000
    final_w = cols * target_size
    final_h = rows * target_size
    
    offset_x = (CANVAS_SIZE - final_w) / 2
    offset_y = (CANVAS_SIZE - final_h) / 2
    
    for p in pieces:
        if p.end_center:
            cx, cy = p.end_center
            p.end_center = (cx + offset_x, cy + offset_y)
            
    output_path = os.path.join(output_dir, f"animation_{base_name}.mp4")
    animate_solution(pieces, board, CANVAS_SIZE, num_frames=90, output_path=output_path)
    print(f"Animation saved to {output_path}")

if __name__ == "__main__":
    main()
