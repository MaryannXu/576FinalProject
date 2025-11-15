# segmentation.py
from typing import List, Tuple
import numpy as np
import cv2
import os
import glob
import random

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
    # Convert to grayscale
    gray = cv2.cvtColor(big_image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by area and shape
    min_area = 1000  # Minimum area threshold
    filtered_contours = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_area:
            # Check if contour is roughly rectangular (for puzzle pieces)
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            if len(approx) >= 4:  # At least 4 vertices for rectangular pieces
                filtered_contours.append(contour)
    
    return filtered_contours

def crop_and_warp_piece(big_image: np.ndarray, contour: np.ndarray) -> Tuple[np.ndarray, Tuple[float, float], float]:
    """
    Given a contour, compute a minimum-area rectangle and warp/crop it
    to get a straight rectangular piece image.

    Returns:
        piece_image: cropped, straightened piece
        center: (x, y) center of the piece in original image (for animation)
        angle: rotation angle of piece in original image (for animation)
    """
    # Get minimum area rectangle
    rect = cv2.minAreaRect(contour)
    center_rect, (w, h), angle = rect
    
    # Get box points (4 corners)
    box = cv2.boxPoints(rect)
    box = box.astype(np.float32)
    
    # Order points: top-left, top-right, bottom-right, bottom-left
    # Sum and difference to identify corners
    s = box.sum(axis=1)
    diff = np.diff(box, axis=1)
    
    # Top-left has smallest sum, bottom-right has largest sum
    # Top-right has smallest difference, bottom-left has largest difference
    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = box[np.argmin(s)]  # top-left
    ordered[2] = box[np.argmax(s)]  # bottom-right
    ordered[1] = box[np.argmin(diff)]  # top-right
    ordered[3] = box[np.argmax(diff)]  # bottom-left
    
    # Calculate width and height
    width = int(max(
        np.linalg.norm(ordered[1] - ordered[0]),
        np.linalg.norm(ordered[2] - ordered[3])
    ))
    height = int(max(
        np.linalg.norm(ordered[3] - ordered[0]),
        np.linalg.norm(ordered[2] - ordered[1])
    ))
    
    # Ensure minimum size
    width = max(width, 10)
    height = max(height, 10)
    
    # Define destination points for perspective transform
    dst_pts = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype=np.float32)
    
    # Get perspective transform matrix
    M = cv2.getPerspectiveTransform(ordered, dst_pts)
    
    # Warp the image
    piece_image = cv2.warpPerspective(big_image, M, (width, height))
    
    # Return center and angle for animation
    center = (float(center_rect[0]), float(center_rect[1]))
    
    return piece_image, center, float(angle)

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

    # Normalize piece sizes here (resize all to same W×H)
    if len(pieces) > 0:
        # Find the largest size to use as target
        target_h, target_w = pieces[0].image.shape[:2]
        for piece in pieces[1:]:
            h, w = piece.image.shape[:2]
            if h * w > target_h * target_w:
                target_h, target_w = h, w
        
        # Resize all pieces to target size
        for piece in pieces:
            if piece.image.shape[:2] != (target_h, target_w):
                piece.image = cv2.resize(piece.image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    return pieces

def load_pieces_from_directory(directory: str) -> List[Piece]:
    """
    Load individual puzzle piece images from a directory.
    Useful for testing when pieces are already separated.
    Assumes images are named in a way that can be sorted.
    """
    # Get all image files
    image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(directory, ext)))
        image_files.extend(glob.glob(os.path.join(directory, ext.upper())))
    
    # Sort to ensure consistent ordering
    image_files.sort()
    
    pieces: List[Piece] = []
    
    for i, img_path in enumerate(image_files):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"Warning: Could not load image {img_path}")
            continue
        
        piece = Piece(piece_id=i, image=img)
        
        # Set initial position (scattered on canvas for animation)
        h, w = img.shape[:2]
        cols_per_row = int(np.ceil(np.sqrt(len(image_files))))
        row = i // cols_per_row
        col = i % cols_per_row
        
        # Calculate center position with some randomness for scattering
        center_x = (col + 0.5) * w + random.randint(-w//2, w//2)
        center_y = (row + 0.5) * h + random.randint(-h//2, h//2)
        piece.start_center = (float(center_x), float(center_y))
        piece.start_angle = float(random.choice([0, 90, 180, 270]))
        
        pieces.append(piece)
    
    # Normalize sizes
    if len(pieces) > 0:
        target_h, target_w = pieces[0].image.shape[:2]
        for piece in pieces[1:]:
            h, w = piece.image.shape[:2]
            if h * w > target_h * target_w:
                target_h, target_w = h, w
        
        for piece in pieces:
            if piece.image.shape[:2] != (target_h, target_w):
                piece.image = cv2.resize(piece.image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    
    return pieces

