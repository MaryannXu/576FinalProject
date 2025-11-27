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
TOP = 0
RIGHT = 1
BOTTOM = 2
LEFT = 3

# Direction vectors (row, col)
DIR_VEC = {
    TOP: (-1, 0),
    RIGHT: (0, 1),
    BOTTOM: (1, 0),
    LEFT: (0, -1)
}

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
        
        # Check if path is valid as is (relative to CWD)
        if os.path.exists(img_rel_path):
            img_path = img_rel_path
        else:
            # Fallback to relative to json file
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
        
        # Extract raw boundary pixels (RGB only)
        rgb = img_arr[:, :, :3].astype(np.float32)
        
        # Store features in standard order: Top, Right, Bottom, Left (relative to image)
        # User requested to look 20 pixels inside (index 19) to avoid rotation artifacts
        OFFSET = 19
        piece.edge_features[0] = rgb[OFFSET, :, :]      # Top
        piece.edge_features[1] = rgb[:, -(OFFSET+1), :] # Right
        piece.edge_features[2] = rgb[-(OFFSET+1), :, :] # Bottom
        piece.edge_features[3] = rgb[:, OFFSET, :]      # Left
        
        loaded_pieces.append(piece)
        
    return loaded_pieces, target_size

def get_edge_feature(piece: Piece, orientation: int, edge_idx: int) -> np.ndarray:
    """
    Get the feature for a specific edge, given the piece's orientation.
    orientation: 0, 1, 2, 3 (representing 0, 90, 180, 270 degrees CW)
    edge_idx: 0=Top, 1=Right, 2=Bottom, 3=Left (in the global frame)
    
    If a piece is rotated 90 deg CW (orientation=1):
    - Global Top (0) corresponds to original Left (3)
    - Global Right (1) corresponds to original Top (0)
    - Global Bottom (2) corresponds to original Right (1)
    - Global Left (3) corresponds to original Bottom (2)
    
    Formula: local_edge = (edge_idx - orientation) % 4
    
    Also need to handle pixel ordering (reverse or not).
    - Top/Bottom are rows. Left/Right are cols.
    - When rotated, a row might become a col.
    - Standard extraction:
      Top: left->right
      Right: top->bottom
      Bottom: left->right
      Left: top->bottom
      
    If we match Right of A with Left of B:
    A's Right (top->bottom) vs B's Left (top->bottom).
    SSD should match directly.
    
    However, if A is rotated 180, its Right becomes original Left.
    Original Left was extracted top->bottom.
    Rotated 180, it is now on the Right, going bottom->top in global frame?
    
    Let's simplify:
    We extract features as 1D arrays.
    We need to ensure that when we compare two edges, they are traversed in the same spatial direction.
    
    Let's assume we always compare "A's Right" vs "B's Left".
    A's Right runs Top->Bottom.
    B's Left runs Top->Bottom.
    
    If Piece is rotated:
    - We fetch the pixels that correspond to that global edge.
    - We might need to reverse them if the rotation flips the direction.
    
    Orientation 0:
    - Top: orig Top (L->R)
    - Right: orig Right (T->B)
    - Bottom: orig Bottom (L->R)
    - Left: orig Left (T->B)
    
    Orientation 1 (90 CW):
    - Top: orig Left (reversed? Orig Left is T->B. Now it's Top L->R. So yes, reversed T->B becomes B->T which is L->R)
      Wait. Orig Left is T->B (0,0) to (H,0).
      Rotate 90 CW. (0,0) moves to (0,W). (H,0) moves to (0,0).
      So it runs (0,W) -> (0,0). That is Right->Left.
      Global Top is Left->Right.
      So yes, we need to reverse.
      
    Let's define a lookup table for (orientation, global_edge) -> (local_edge, reverse_flag)
    """
    
    # (local_edge, reverse)
    # local_edge: 0=T, 1=R, 2=B, 3=L
    # reverse: True means flip the array
    
    # Map (orientation, global_edge) -> (local_edge, reverse)
    # Derived by visualizing the rotation
    
    MAPPING = {
        # Orientation 0
        (0, 0): (0, False), # Top -> Top
        (0, 1): (1, False), # Right -> Right
        (0, 2): (2, False), # Bottom -> Bottom
        (0, 3): (3, False), # Left -> Left
        
        # Orientation 1 (90 CW)
        (1, 0): (3, True),  # Top -> Left (reversed)
        (1, 1): (0, False), # Right -> Top
        (1, 2): (1, True),  # Bottom -> Right (reversed)
        (1, 3): (2, False), # Left -> Bottom
        
        # Orientation 2 (180 CW)
        (2, 0): (2, True),  # Top -> Bottom (reversed)
        (2, 1): (3, True),  # Right -> Left (reversed)
        (2, 2): (0, True),  # Bottom -> Top (reversed)
        (2, 3): (1, True),  # Left -> Right (reversed)
        
        # Orientation 3 (270 CW)
        (3, 0): (1, False), # Top -> Right
        (3, 1): (2, True),  # Right -> Bottom (reversed)
        (3, 2): (3, False), # Bottom -> Left
        (3, 3): (0, True),  # Left -> Top (reversed)
    }
    
    local_idx, reverse = MAPPING[(orientation, edge_idx)]
    feat = piece.edge_features[local_idx]
    
    if reverse:
        return feat[::-1]
    return feat

def compute_ssd(p1_pixels: np.ndarray, p2_pixels: np.ndarray) -> float:
    diff = p1_pixels - p2_pixels
    ssd = np.sum(diff ** 2)
    return float(ssd)

def build_ssd_compatibility(pieces: List[Piece]) -> Dict[Tuple[int, int, int, int], float]:
    """
    Compute SSD for all relevant pairs.
    We compute matches between ALL edges of ALL pieces in their raw (orientation=0) state.
    The solver will handle the rotation logic by looking up the correct raw edge.
    
    Actually, to make the solver efficient, we can precompute SSD between all raw edges.
    (idA, raw_edgeA, idB, raw_edgeB) -> score
    """
    compat = {}
    
    for pA in pieces:
        for pB in pieces:
            if pA.id == pB.id:
                continue
            
            # Compare every raw edge of A with every raw edge of B
            # We need to know if they match "correctly".
            # A match is valid if the pixels align.
            # But "alignment" depends on how they are placed.
            # Let's just store the raw SSD between all 4 edges of A and all 4 edges of B.
            # But wait, direction matters.
            # Standard: Right(A) vs Left(B).
            # If we match Top(A) vs Bottom(B), Top is L->R, Bottom is L->R. They match directly.
            # If we match Right(A) vs Left(B), Right is T->B, Left is T->B. They match directly.
            
            # So we just compute SSD between all pairs of raw edges, considering both direct and reversed matches?
            # No, let's stick to the solver logic computing the SSD on demand or caching it.
            # Given the small number of pieces (e.g. 20), we can compute on the fly or cache.
            pass
            
def get_factor_pairs(n: int) -> List[Tuple[int, int]]:
    """
    Find all factor pairs (r, c) such that r * c = n.
    Returns list of tuples, e.g. for 20: [(4, 5), (5, 4), (2, 10), (10, 2)...]
    Sorted by 'squareness' (aspect ratio closer to 1 first).
    """
    factors = []
    for i in range(1, int(math.sqrt(n)) + 1):
        if n % i == 0:
            r = i
            c = n // i
            factors.append((r, c))
            if r != c:
                factors.append((c, r))
                
    # Sort by abs(r-c) to prefer square shapes
    factors.sort(key=lambda x: abs(x[0] - x[1]))
    return factors

def solve_from_seed(
    pieces: List[Piece],
    seed_id: int,
    seed_orient: int,
    ssd_cache: Dict,
    max_rows: int,
    max_cols: int
) -> Tuple[Optional[Board], float]:
    """
    Run the greedy best-first solver starting from a specific seed.
    Enforces that the grid never exceeds max_rows x max_cols.
    Returns (Board, total_score). Lower score is better.
    """
    virtual_grid = {} # (r,c) -> (piece_id, orientation)
    placed_pieces = set()
    
    # Place seed
def solve_raster_scan(
    pieces: List[Piece],
    rows: int,
    cols: int,
    seed_p: Piece,
    seed_orient: int,
    ssd_cache: Dict
) -> Tuple[Optional[Board], float]:
    """
    Tries to fill a rows x cols grid in raster order (row by row),
    starting with seed_p at (0,0) with seed_orient.
    Greedily picks the best fit for each subsequent cell.
    """
    board = Board(rows, cols)
    board.place_piece(0, 0, seed_p.id, seed_orient)
    
    placed_ids = {seed_p.id}
    total_score = 0.0
    
    # Helper to get SSD
    def get_ssd(pA, edgeA, pB, edgeB, reverseB):
        key = (pA.id, edgeA, pB.id, edgeB, reverseB)
        if key in ssd_cache:
            return ssd_cache[key]
        fA = pA.edge_features[edgeA]
        fB = pB.edge_features[edgeB]
        if reverseB:
            fB = fB[::-1]
        val = compute_ssd(fA, fB)
        ssd_cache[key] = val
        return val

    for r in range(rows):
        for c in range(cols):
            if r == 0 and c == 0:
                continue
                
            # Find best piece for (r,c)
            best_cand_id = None
            best_cand_orient = -1
            best_local_score = float('inf')
            
            # Constraints:
            # Must match Left neighbor (r, c-1) if c > 0
            # Must match Top neighbor (r-1, c) if r > 0
            
            left_neighbor = None
            top_neighbor = None
            
            if c > 0:
                left_pid, left_orient = board.grid[r][c-1]
                left_neighbor = (next(p for p in pieces if p.id == left_pid), left_orient)
                
            if r > 0:
                top_pid, top_orient = board.grid[r-1][c]
                top_neighbor = (next(p for p in pieces if p.id == top_pid), top_orient)
            
            # Iterate all unused pieces
            for cand in pieces:
                if cand.id in placed_ids:
                    continue
                    
                # For translation only, orientation is always 0
                for cand_orient in range(1):
                    current_score = 0.0
                    valid = True
                    
                    # Check Left match: Left(Right) vs Cand(Left)
                    if left_neighbor:
                        lp, lo = left_neighbor
                        # lp Right edge vs cand Left edge
                        # We use get_edge_feature logic but via get_ssd helper?
                        # get_ssd expects (pA, raw_edgeA, pB, raw_edgeB, revB)
                        # But get_edge_feature handles the rotation mapping.
                        # Let's use get_edge_feature directly for clarity/correctness
                        # since we already implemented it and it works.
                        
                        f_left = get_edge_feature(lp, lo, RIGHT)
                        f_cand = get_edge_feature(cand, cand_orient, LEFT)
                        current_score += compute_ssd(f_left, f_cand)
                        
                    # Check Top match: Top(Bottom) vs Cand(Top)
                    if top_neighbor:
                        tp, to = top_neighbor
                        f_top = get_edge_feature(tp, to, BOTTOM)
                        f_cand = get_edge_feature(cand, cand_orient, TOP)
                        current_score += compute_ssd(f_top, f_cand)
                        
                    if current_score < best_local_score:
                        best_local_score = current_score
                        best_cand_id = cand.id
                        best_cand_orient = cand_orient
            
            if best_cand_id is None:
                # Could not find any piece? Should not happen unless pieces ran out (impossible by loop)
                # or logic error.
                return None, float('inf')
                
            board.place_piece(r, c, best_cand_id, best_cand_orient)
            placed_ids.add(best_cand_id)
            total_score += best_local_score
            
    return board, total_score

def solve_puzzle_constrained(
    pieces: List[Piece],
    rows_ignored: int,
    cols_ignored: int,
    compat_unused: Dict
) -> Optional[Board]:
    """
    Robust solver that enforces rectangular shapes using Raster Scan.
    """
    best_board = None
    best_avg_score = float('inf')
    
    n = len(pieces)
    shapes = get_factor_pairs(n)
    # Filter out 1xN shapes if possible, user dislikes them.
    # Keep only shapes where min_dim >= 2?
    shapes = [s for s in shapes if min(s) >= 2]
    if not shapes:
        # Fallback if prime number or something
        shapes = get_factor_pairs(n)
        
    print(f"Running constrained solver with shapes: {shapes}...")
    
    ssd_cache = {} # Not used in raster scan currently, but could be
    
    for r_dim, c_dim in shapes:
        print(f"  Trying shape {r_dim}x{c_dim}...")
        
        # Try every piece as (0,0) with every orientation
        for seed_p in pieces:
            for seed_orient in range(4):
                board, score = solve_raster_scan(pieces, r_dim, c_dim, seed_p, seed_orient, ssd_cache)
                
                if board:
                    # Normalize score by number of internal edges
                    # Internal vertical edges: r * (c-1)
                    # Internal horizontal edges: (r-1) * c
                    num_edges = r_dim * (c_dim - 1) + (r_dim - 1) * c_dim
                    avg_score = score / max(1, num_edges)
                    
                    if avg_score < best_avg_score:
                        best_avg_score = avg_score
                        best_board = board
                        
    if best_board:
        print(f"Best solution score: {best_avg_score:.2f}, Size: {best_board.rows}x{best_board.cols}")
    else:
        print("Failed to find any valid solution.")
        
    return best_board

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
            pid, orient = cell
            piece = piece_map[pid]
            
            # Rotate image
            # orient 0=0, 1=90CW, 2=180, 3=270CW
            # PIL rotate is CCW, so we use -90*orient
            p_img = Image.fromarray(piece.image)
            p_img = p_img.rotate(-90 * orient, expand=False)
            
            canvas.paste(p_img, (c * target_size, r * target_size))
            
    canvas.save(output_path)
    print(f"Saved to {output_path}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 recombine_translated.py <image_name_or_features_json>")
        return

    output_dir = sys.argv[1]
    json_path = os.path.join(output_dir, "features.json")
    base_name = os.path.basename(output_dir.rstrip(os.sep))
        
    if not os.path.exists(json_path):
        print(f"Error: {json_path} not found.")
        return

    print(f"Loading pieces from {json_path}...")
    pieces, target_size = load_pieces_raw(json_path)
    
    # n = len(pieces)
    # side = int(math.sqrt(n))
    # rows = cols = side
    
    print("Solving with Robust Multi-Seed & Shape Constraint...")
    board = solve_puzzle_constrained(pieces, 0, 0, {})
    
    if board:
        output_filename = os.path.join(output_dir, f"recombined_{base_name}.png")
        stitch_solution(board, pieces, target_size, output_filename)
    else:
        print("Failed to solve.")

if __name__ == "__main__":
    main()
