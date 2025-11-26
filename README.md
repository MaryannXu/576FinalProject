# Puzzle Reconstruction Project

This project reconstructs scrambled puzzle pieces into their original image and generates an animation of the solution.

## Prerequisites

- Python 3
- OpenCV (`opencv-python`)
- NumPy
- Pillow
- FFmpeg (for video generation)

## Usage

### 1. Preprocess Input Image
Convert a raw RGB image into puzzle pieces and extract features.

```bash
python3 576_preprocess/preprocess.py <path_to_input.rgb>
```

**Example:**
```bash
python3 576_preprocess/preprocess.py images/more_samples/sample2/sample2_translate.rgb
```
This will generate:
- A directory of piece images (e.g., `sample2_translate_pieces/`)
- A features JSON file (e.g., `sample2_translate_features.json`)

### 2. Recombine Pieces
Solve the puzzle and stitch the pieces back together.

```bash
python3 recombine_translated_v2.py <path_to_features.json>
```

**Example:**
```bash
python3 recombine_translated_v2.py sample2_translate_features.json
```
**Output:** `recombined_<basename>.png`

### 3. Generate Animation
Create a video showing the pieces moving to their final positions.

```bash
python3 animate_translated.py <path_to_features.json>
```

**Example:**
```bash
python3 animate_translated.py sample2_translate_features.json
```
**Output:** `animation_<basename>.mp4`

## Notes
- The solver assumes pieces are **translated only** (no rotation).
- Video generation uses `ffmpeg` with H.264 codec for compatibility.