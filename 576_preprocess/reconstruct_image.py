#!/usr/bin/env python3
import sys
from pathlib import Path

from PIL import Image

from solve_puzzle import (
    load_pieces_from_json,
    build_compatibility_matrix,
    compute_max_dist_threshold,
    solve_layout,
)


def reconstruct_image(json_path: str, out_path: str):
    # 1. Load the pieces from the same features JSON
    pieces = load_pieces_from_json(json_path)
    N = len(pieces)
    side = int(round(N ** 0.5))
    rows = cols = side

    # 2. Rebuild compatibility + threshold (same as in solve_puzzle.py)
    compat = build_compatibility_matrix(pieces)
    max_dist = compute_max_dist_threshold(compat, percentile=60.0)

    # 3. Choose orientations based on filename
    name = Path(json_path).name.lower()
    if "translate" in name:
        orientations = [0]  # pieces already upright
        print("Reconstructing TRANSLATE puzzle (no rotations).")
    else:
        orientations = [0, 90, 180, 270]
        print("Reconstructing ROTATE puzzle (allowing rotations).")

    # 4. Solve the layout again to get the board in memory
    board = solve_layout(pieces, compat, rows, cols, max_dist, orientations)
    if board is None:
        print("No solution found; cannot reconstruct image.")
        return

    # 5. Prepare a blank canvas the right size
    first_img_path = pieces[0].image_path
    tile = Image.open(first_img_path).convert("RGBA")
    tile_w, tile_h = tile.size

    canvas_w = cols * tile_w
    canvas_h = rows * tile_h
    canvas = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 255))

    # Map piece index -> PNG path
    id_to_path = {p.index: p.image_path for p in pieces}

    # 6. Paste each solved piece into the right spot with the right rotation
    for r in range(rows):
        for c in range(cols):
            piece_idx, orientation = board.grid[r][c]
            img_path = id_to_path[piece_idx]

            piece_img = Image.open(img_path).convert("RGBA")
            rotated = piece_img.rotate(-orientation, expand=False)

            x = c * tile_w
            y = r * tile_h
            canvas.paste(rotated, (x, y), rotated)

    # 7. Save the final reconstructed puzzle image
    canvas.save(out_path)
    print(f"Saved reconstructed image to {out_path}")


def main():
    if len(sys.argv) != 2 and len(sys.argv) != 3:
        print("Usage: python3 reconstruct_image.py <features.json> [output.png]")
        sys.exit(1)

    json_path = sys.argv[1]
    if len(sys.argv) == 3:
        out_path = sys.argv[2]
    else:
        # default name if not specified
        base = Path(json_path).stem
        out_path = f"{base}_solved.png"

    if not Path(json_path).is_file():
        print(f"JSON not found: {json_path}")
        sys.exit(1)

    reconstruct_image(json_path, out_path)


if __name__ == "__main__":
    main()