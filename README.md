## Usage

### 1. Preprocess Input Image
Convert a raw RGB image into ```bash
python preprocess.py <path_to_image.rgb>
```

**Example:**
```bash
python preprocess.py samples/sample2_translate.rgb
```

### 2. Recombine Pieces
Solve the puzzle and stitch the pieces back together.

```bash
python recombine_translated.py output/sample2_translate
```

### 3. Generate Animation
Create a video showing the pieces moving to their final positions.

```bash
python animate_translated.py output/sample2_translate
```