import json
import os
import sys
import math
import numpy as np
from PIL import Image
from typing import List, Dict, Tuple, Set, Optional
import heapq

# Add current directory to path
sys.path.append(os.getcwd())

from models import Piece, Board

# Constants
TOP = (-1, 0)
LEFT = (0, -1)
BOTTOM = (1, 0)
RIGHT = (0, 1)

OPPOSITE = {
    TOP: BOTTOM,
    BOTTOM: TOP,
    LEFT: RIGHT,
    RIGHT: LEFT
}

def load_pieces_raw(json_path: str) -> Tuple[List[Piece], int]:
    """
    Load pieces and extract raw boundary pixels.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    target_size = data['target_size']
    pieces_data = data['pieces']
    
    loaded_pieces = []
    base_dir = os.path.dirname(json_path)
    
    for p_data in pieces_data:
        idx = p_data['index']
        img_rel_path = p_data['image']
        img_path = os.path.join(base_dir, img_rel_path)
        
        if not os.path.exists(img_path):
            print(f"Warning: Image not found at {img_path}")
            continue
            
        # Load image
        pil_img = Image.open(img_path).convert('RGBA')
        # Resize if necessary (should be already target_size)
        if pil_img.size != (target_size, target_size):
            pil_img = pil_img.resize((target_size, target_size))
            
        img_arr = np.array(pil_img)
        
        piece = Piece(idx, img_arr)
        
        # Extract raw boundary pixels (RGB only, ignore Alpha for matching usually, 
        # but here we assume full square pieces so alpha is 255)
        rgb = img_arr[:, :, :3].astype(np.float32)
        
        # Top row
        piece.edge_features[0] = rgb[0, :, :]
        # Right col
        piece.edge_features[1] = rgb[:, -1, :]
        # Bottom row
        piece.edge_features[2] = rgb[-1, :, :]
        # Left col
        piece.edge_features[3] = rgb[:, 0, :]
        
        loaded_pieces.append(piece)
        
    return loaded_pieces, target_size

def compute_ssd(p1_pixels: np.ndarray, p2_pixels: np.ndarray) -> float:
    """
    Compute Sum of Squared Differences between two pixel arrays.
    """
    diff = p1_pixels - p2_pixels
    ssd = np.sum(diff ** 2)
    return float(ssd)

def build_ssd_compatibility(pieces: List[Piece]) -> Dict[Tuple[int, int, int, int], float]:
    """
    Compute SSD for all relevant pairs.
    We only care about:
    - Right of A matching Left of B
    - Bottom of A matching Top of B
    
    Returns dict: (idA, edgeA, idB, edgeB) -> score
    edge indices: 0=Top, 1=Right, 2=Bottom, 3=Left
    """
    compat = {}
    
    for pA in pieces:
        for pB in pieces:
            if pA.id == pB.id:
                continue
            
            # Match Right of A (1) with Left of B (3)
            score_horz = compute_ssd(pA.edge_features[1], pB.edge_features[3])
            compat[(pA.id, 1, pB.id, 3)] = score_horz
            
            # Match Bottom of A (2) with Top of B (0)
            score_vert = compute_ssd(pA.edge_features[2], pB.edge_features[0])
            compat[(pA.id, 2, pB.id, 0)] = score_vert
            
    return compat

def solve_puzzle_best_first(
    pieces: List[Piece],
    rows: int,
    cols: int,
    compat: Dict[Tuple[int, int, int, int], float]
) -> Optional[Board]:
    """
    Best-First Greedy Solver.
    1. Find the globally best matching pair.
    2. Place them relative to each other.
    3. Maintain a heap of potential placements (neighbor of placed pieces).
    """
    
    # Sort all matches
    all_matches = []
    for (idA, edgeA, idB, edgeB), score in compat.items():
        all_matches.append((score, idA, edgeA, idB, edgeB))
    
    all_matches.sort()
    
    # We need to build a relative graph first?
    # Or we can just place them on a virtual infinite grid and then crop?
    # Let's use a virtual grid.
    
    # Map: (v_row, v_col) -> piece_id
    virtual_grid = {}
    placed_pieces = set()
    
    # Priority Queue: (score, piece_id_to_place, ref_v_row, ref_v_col, direction_from_ref)
    pq = []
    
    # Start with the absolute best match
    best_score, idA, edgeA, idB, edgeB = all_matches[0]
    
    # Place A at (0,0)
    virtual_grid[(0, 0)] = idA
    placed_pieces.add(idA)
    
    # Determine where B goes relative to A
    # edgeA: 1 (Right) -> B is at (0, 1)
    # edgeA: 2 (Bottom) -> B is at (1, 0)
    # Note: We only computed Right->Left and Bottom->Top in build_ssd_compatibility
    
    if edgeA == 1: # Right
        virtual_grid[(0, 1)] = idB
        placed_pieces.add(idB)
    elif edgeA == 2: # Bottom
        virtual_grid[(1, 0)] = idB
        placed_pieces.add(idB)
    else:
        # Should not happen with our compat construction
        pass
        
    # Helper to add neighbors to PQ
    def add_neighbors_to_pq(pid, vr, vc):
        # Check all 4 directions
        # 0: Top neighbor (needs to match my Top(0) with their Bottom(2))
        # 1: Right neighbor (needs to match my Right(1) with their Left(3))
        # 2: Bottom neighbor (needs to match my Bottom(2) with their Top(0))
        # 3: Left neighbor (needs to match my Left(3) with their Right(1))
        
        # We only have compat entries for (Right->Left) and (Bottom->Top)
        # So:
        # If we want neighbor to Top: It means Neighbor(Bottom) -> Me(Top). Key: (N, 2, Me, 0)
        # If we want neighbor to Right: Me(Right) -> Neighbor(Left). Key: (Me, 1, N, 3)
        # If we want neighbor to Bottom: Me(Bottom) -> Neighbor(Top). Key: (Me, 2, N, 0)
        # If we want neighbor to Left: Neighbor(Right) -> Me(Left). Key: (N, 1, Me, 3)
        
        for candidate in pieces:
            if candidate.id in placed_pieces:
                continue
                
            # Try Top neighbor
            if (vr-1, vc) not in virtual_grid:
                key = (candidate.id, 2, pid, 0)
                if key in compat:
                    heapq.heappush(pq, (compat[key], candidate.id, vr, vc, TOP))
            
            # Try Right neighbor
            if (vr, vc+1) not in virtual_grid:
                key = (pid, 1, candidate.id, 3)
                if key in compat:
                    heapq.heappush(pq, (compat[key], candidate.id, vr, vc, RIGHT))
                    
            # Try Bottom neighbor
            if (vr+1, vc) not in virtual_grid:
                key = (pid, 2, candidate.id, 0)
                if key in compat:
                    heapq.heappush(pq, (compat[key], candidate.id, vr, vc, BOTTOM))
            
            # Try Left neighbor
            if (vr, vc-1) not in virtual_grid:
                key = (candidate.id, 1, pid, 3)
                if key in compat:
                    heapq.heappush(pq, (compat[key], candidate.id, vr, vc, LEFT))

    # Add initial neighbors
    for (r, c), pid in list(virtual_grid.items()):
        add_neighbors_to_pq(pid, r, c)
        
    while len(placed_pieces) < len(pieces) and pq:
        score, pid, ref_r, ref_c, direction = heapq.heappop(pq)
        
        if pid in placed_pieces:
            continue
            
        # Determine new coordinates
        dr, dc = direction
        nr, nc = ref_r + dr, ref_c + dc
        
        if (nr, nc) in virtual_grid:
            continue
            
        # Check consistency with other neighbors?
        # For now, just greedy placement.
        
        virtual_grid[(nr, nc)] = pid
        placed_pieces.add(pid)
        add_neighbors_to_pq(pid, nr, nc)
        
    # Normalize coordinates
    min_r = min(r for r, c in virtual_grid.keys())
    min_c = min(c for r, c in virtual_grid.keys())
    
    max_r = max(r for r, c in virtual_grid.keys())
    max_c = max(c for r, c in virtual_grid.keys())
    
    h = max_r - min_r + 1
    w = max_c - min_c + 1
    
    print(f"Reconstructed grid size: {h}x{w}")
    
    if h != rows or w != cols:
        print(f"Warning: Reconstructed grid {h}x{w} does not match expected {rows}x{cols}")
        # If it doesn't match, we might have made a mistake or the grid isn't perfect.
        # But we return what we have.
        board = Board(h, w)
    else:
        board = Board(rows, cols)
        
    for (r, c), pid in virtual_grid.items():
        board.place_piece(r - min_r, c - min_c, pid, 0)
        
    return board

def stitch_solution(board: Board, pieces: List[Piece], target_size: int, output_path: str):
    rows = board.rows
    cols = board.cols
    
    canvas_width = cols * target_size
    canvas_height = rows * target_size
    
    canvas = Image.new('RGBA', (canvas_width, canvas_height))
    piece_map = {p.id: p for p in pieces}
    
    for r in range(rows):
        for c in range(cols):
            cell = board.grid[r][c]
            if cell is None:
                continue
            pid, _ = cell
            piece = piece_map[pid]
            p_img = Image.fromarray(piece.image)
            canvas.paste(p_img, (c * target_size, r * target_size))
            
    canvas.save(output_path)
    print(f"Saved to {output_path}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 recombine_translated_v2.py <features_json_path>")
        return

    json_path = sys.argv[1]
    if not os.path.exists(json_path):
        print(f"Error: {json_path} not found.")
        return

    print("Loading pieces...")
    pieces, target_size = load_pieces_raw(json_path)
    
    n = len(pieces)
    side = int(math.sqrt(n))
    rows = cols = side
    
    print("Computing SSD compatibility...")
    compat = build_ssd_compatibility(pieces)
    
    print("Solving with Best-First...")
    board = solve_puzzle_best_first(pieces, rows, cols, compat)
    
    if board:
        base_name = os.path.splitext(os.path.basename(json_path))[0]
        output_filename = f"recombined_{base_name}.png"
        stitch_solution(board, pieces, target_size, output_filename)
    else:
        print("Failed to solve.")

if __name__ == "__main__":
    main()
