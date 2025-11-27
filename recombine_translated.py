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
        piece.edge_features[0] = rgb[0, :, :]      # Top
        piece.edge_features[1] = rgb[:, -1, :]     # Right
        piece.edge_features[2] = rgb[-1, :, :]     # Bottom
        piece.edge_features[3] = rgb[:, 0, :]      # Left
        
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
            
    return compat # Not used in this new design, we'll compute on demand or cache

def solve_puzzle_best_first(
    pieces: List[Piece],
    rows: int,
    cols: int,
    compat_unused: Dict
) -> Optional[Board]:
    
    # Precompute all pairwise edge SSDs (both normal and reversed) to speed up
    # cache[(idA, edgeA, idB, edgeB, reverseB)] = ssd
    ssd_cache = {}
    
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

    # We need to find the best starting pair (best match across all possibilities)
    # Possibilities: (PieceA, OrientA) placed at (0,0), (PieceB, OrientB) placed at (0,1) or (1,0)
    # Actually, we can fix PieceA's orientation to 0 (w.l.o.g for the first piece, assuming we don't care about global rotation)
    # But if we want "upright", we might care. But we don't know what is upright.
    # So let's just pick the best matching pair of edges and lock them.
    
    all_matches = []
    
    # Iterate all pairs of pieces
    for i in range(len(pieces)):
        for j in range(len(pieces)):
            if i == j: continue
            pA = pieces[i]
            pB = pieces[j]
            
            # Try all raw edge combinations
            for ea in range(4):
                for eb in range(4):
                    # Calculate SSD for direct match and reversed match
                    # Direct: A(ea) matches B(eb)
                    s_direct = get_ssd(pA, ea, pB, eb, False)
                    s_rev = get_ssd(pA, ea, pB, eb, True)
                    
                    # We store both. The solver will deduce orientation from this.
                    # If A(ea) matches B(eb) directly:
                    # It means in the global frame, Edge(A) and Edge(B) are adjacent and consistent.
                    all_matches.append((s_direct, pA, ea, pB, eb, False))
                    all_matches.append((s_rev, pA, ea, pB, eb, True))
                    
    all_matches.sort(key=lambda x: x[0])
    
    # Start with best match
    # We need to pick a reference frame.
    # Let's say the first piece in the match is at (0,0) with Orientation 0.
    # We need to determine the second piece's position and orientation.
    
    # Match: A(edge_a) touches B(edge_b).
    # If A is Orient 0.
    # If edge_a is Right (1). Then B is to the Right.
    # So B's Left edge must be the one touching A's Right.
    # B's Left edge in global frame must be edge_b.
    # So we need to find OrientB such that get_edge(B, OrientB, LEFT) == raw_edge_b (possibly reversed).
    
    # Let's generalize:
    # We place A at (0,0) with Orient 0.
    # We have a match between RawEdgeA and RawEdgeB with score S, reversed=R.
    # RawEdgeA corresponds to GlobalEdgeA (since OrientA=0).
    # We place B at neighbor of A in direction GlobalEdgeA.
    # The GlobalEdge of B touching A is OPPOSITE[GlobalEdgeA].
    # We need to find OrientB such that:
    #   get_edge_feature(B, OrientB, OPPOSITE[GlobalEdgeA]) == (RawEdgeB, reversed=R relative to RawEdgeA?)
    
    # Wait, the SSD was computed between RawEdgeA and (RawEdgeB potentially reversed).
    # So pixels(RawEdgeA) ~ pixels(RawEdgeB_transformed).
    # We know pixels(GlobalEdgeA) = pixels(RawEdgeA) (since OrientA=0).
    # We need pixels(GlobalEdgeB_touching) ~ pixels(GlobalEdgeA).
    # So pixels(GlobalEdgeB_touching) ~ pixels(RawEdgeB_transformed).
    
    # This implies we need to find OrientB such that the feature extracted for OPPOSITE[GlobalEdgeA]
    # is physically identical to the feature used in the match.
    
    best_match = all_matches[0]
    score, pA, raw_ea, pB, raw_eb, rev = best_match
    
    virtual_grid = {} # (r,c) -> (piece_id, orientation)
    placed_pieces = set()
    
    # Place A
    virtual_grid[(0,0)] = (pA.id, 0)
    placed_pieces.add(pA.id)
    
    pq = [] # (score, p_source_id, source_r, source_c, source_global_edge)
    
    # Helper to push neighbors
    def add_neighbors(pid, r, c, orient):
        # For each of the 4 global edges of the placed piece
        for global_edge in range(4):
            # Check if neighbor cell is empty
            dr, dc = DIR_VEC[global_edge]
            nr, nc = r + dr, c + dc
            if (nr, nc) in virtual_grid:
                continue
                
            # We want to find a piece that matches 'pid' on 'global_edge'
            # The source feature is:
            p = next(p for p in pieces if p.id == pid)
            src_feat = get_edge_feature(p, orient, global_edge)
            
            # We look for any unplaced piece that has a matching edge
            for cand in pieces:
                if cand.id in placed_pieces:
                    continue
                    
                # Try all 4 orientations of candidate
                for cand_orient in range(4):
                    # The candidate's edge touching source is OPPOSITE[global_edge]
                    cand_edge_idx = OPPOSITE[global_edge]
                    cand_feat = get_edge_feature(cand, cand_orient, cand_edge_idx)
                    
                    # Compute SSD
                    s = compute_ssd(src_feat, cand_feat)
                    
                    # Push to PQ
                    heapq.heappush(pq, (s, pid, r, c, global_edge, cand.id, cand_orient))

    add_neighbors(pA.id, 0, 0, 0)
    
    while len(placed_pieces) < len(pieces) and pq:
        s, src_id, src_r, src_c, src_edge, cand_id, cand_orient = heapq.heappop(pq)
        
        if cand_id in placed_pieces:
            continue
            
        # Determine location
        dr, dc = DIR_VEC[src_edge]
        nr, nc = src_r + dr, src_c + dc
        
        if (nr, nc) in virtual_grid:
            continue
            
        # Place it
        virtual_grid[(nr, nc)] = (cand_id, cand_orient)
        placed_pieces.add(cand_id)
        
        add_neighbors(cand_id, nr, nc, cand_orient)
        
    # Normalize grid
    if not virtual_grid:
        return None
        
    min_r = min(r for r, c in virtual_grid.keys())
    min_c = min(c for r, c in virtual_grid.keys())
    max_r = max(r for r, c in virtual_grid.keys())
    max_c = max(c for r, c in virtual_grid.keys())
    
    h = max_r - min_r + 1
    w = max_c - min_c + 1
    
    print(f"Reconstructed grid size: {h}x{w}")
    
    board = Board(h, w) # Use dynamic size
    
    for (r, c), (pid, orient) in virtual_grid.items():
        board.place_piece(r - min_r, c - min_c, pid, orient)
        
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
            pid, orient = cell
            piece = piece_map[pid]
            
            # Rotate image
            # orient 0=0, 1=90CW, 2=180, 3=270CW
            # PIL rotate is CCW, so we use -90*orient
            p_img = Image.fromarray(piece.image)
            p_img = p_img.rotate(-90 * orient, expand=False) # expand=False to keep size? 
            # Wait, if we rotate 90, size might change if not square. But pieces are square (target_size).
            
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
    
    n = len(pieces)
    side = int(math.sqrt(n))
    rows = cols = side
    
    print("Computing SSD compatibility...")
    # compat = build_ssd_compatibility(pieces) # Not used directly anymore
    
    print("Solving with Best-First...")
    board = solve_puzzle_best_first(pieces, rows, cols, {})
    
    if board:
        output_filename = os.path.join(output_dir, f"recombined_{base_name}.png")
        stitch_solution(board, pieces, target_size, output_filename)
    else:
        print("Failed to solve.")

if __name__ == "__main__":
    main()
