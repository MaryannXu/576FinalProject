## Usage

### 1. Translation Only (No Rotation)
Use this if the puzzle pieces are already upright (axis-aligned).

```bash
# 1. Preprocess (detect pieces, extract features)
python preprocess_translate.py samples/sample2_translate.rgb

# 2. Solve (recombine pieces)
python recombine_translate.py output/sample2_translate

# 3. Animate (generate video)
python animate_translate.py output/sample2_translate
```

### 2. Rotation Support
Use this if the puzzle pieces are rotated.

```bash
# 1. Preprocess (detect pieces, normalize orientation)
python preprocess_rotate.py samples/sample2_rotate.rgb

# 2. Solve (recombine pieces with rotation)
python recombine_rotate.py output/sample2_rotate

# 3. Animate (generate video)
python animate_rotate.py output/sample2_rotate
```

## Notes
- The solver now supports **rotation** (0, 90, 180, 270 degrees).
- Video generation uses `ffmpeg` with H.264 codec.