# segmentation.py
from typing import List, Tuple
import numpy as np
import cv2

from models import Piece

def load_big_image(path: str) -> np.ndarray:
    """
    Load the full scattered puzzle image.
    """
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    return img

def find_piece_contours(big_image: np.ndarray) -> List[np.ndarray]:
    """
    Use edge detection + contour finding to locate potential puzzle pieces.
    Returns list of contours (each contour is a numpy array of points).
    """
    # TODO: grayscale, blur, Canny, findContours, filter by area/shape.
    return []

def crop_and_warp_piece(big_image: np.ndarray, contour: np.ndarray) -> Tuple[np.ndarray, Tuple[float, float], float]:
    """
    Given a contour, compute a minimum-area rectangle and warp/crop it
    to get a straight rectangular piece image.

    Returns:
        piece_image: cropped, straightened piece
        center: (x, y) center of the piece in original image (for animation)
        angle: rotation angle of piece in original image (for animation)
    """
    # TODO: cv2.minAreaRect, cv2.boxPoints, cv2.getPerspectiveTransform, cv2.warpPerspective
    piece_image = big_image.copy()
    center = (0.0, 0.0)
    angle = 0.0
    return piece_image, center, angle

def segment_pieces_from_canvas(big_image: np.ndarray) -> List[Piece]:
    """
    Main entry point: segment all pieces from the big scattered image.
    Creates Piece objects with start_center/start_angle set for animation.
    """
    contours = find_piece_contours(big_image)
    pieces: List[Piece] = []

    for i, contour in enumerate(contours):
        piece_img, center, angle = crop_and_warp_piece(big_image, contour)

        piece = Piece(piece_id=i, image=piece_img)
        piece.start_center = center
        piece.start_angle = angle
        pieces.append(piece)

    # Optionally normalize piece sizes here (resize all to same W×H)
    # TODO: resize all pieces if needed

    return pieces

