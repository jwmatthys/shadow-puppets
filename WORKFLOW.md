# Adding New Shapes - Workflow Guide

This guide explains how to add new shapes to Shadow Puppets training data.

## Folder Structure

```
shadow-puppets/
├── assets/
│   ├── image_sources/             # Source images (SVG, PNG, JPG, etc.)
│   │   ├── bear/
│   │   ├── cactus/
│   │   ├── cat/
│   │   ├── elephant/
│   │   ├── giraffe/
│   │   └── lightning_bolt/
│   │
│   └── training/                  # Generated training data (64x64 PNGs)
│       ├── shapes/                # From generate_shapes.py
│       │   ├── delta/
│       │   ├── heart/
│       │   ├── moon/
│       │   ├── ring/
│       │   ├── star/
│       │   └── display_names.json
│       ├── letters/               # From generate_letters.py
│       │   ├── letter_A/
│       │   ├── letter_F/
│       │   ├── letter_H/
│       │   ├── letter_J/
│       │   ├── letter_L/
│       │   ├── letter_M/
│       │   ├── letter_S/
│       │   ├── letter_T/
│       │   ├── letter_V/
│       │   ├── letter_W/
│       │   ├── letter_X/
│       │   ├── letter_Y/
│       │   └── display_names.json
│       └── svgs/                  # From augment_images.sh
│           ├── cat/
│           ├── elephant/
│           ├── giraffe/
│           └── display_names.json
│
├── models/                        # Trained classifier
│   ├── shape_classifier.joblib
│   ├── scaler.joblib
│   └── model_metadata.json
│
├── data/                          # Runtime data
│   ├── high_score.txt
│   ├── logs/                      # Game session CSVs
│   └── captures/                  # Silhouette captures for training
│
├── fonts/                         # Custom fonts
│   ├── BOMBARD_.otf              # Headings
│   └── Carlito-Regular.ttf       # Body text
│
├── bgm/                           # Background music (.ogg)
└── sfx/                           # Sound effects
    ├── boom.ogg                   # Timer expired
    └── bell.ogg                   # Match achieved
```

## Three Types of Training Data

### 1. Generated Shapes (Code)
Geometric shapes drawn with OpenCV. Edit `generate_shapes.py`.

**Current:** ring, delta (triangle), heart, star, moon

### 2. Letters (Code)
Text rendered with OpenCV fonts. Edit `generate_letters.py`.

**Current:** A, F, H, J, L, M, S, T, V, W, X, Y

### 3. Image-Based Shapes (External)
Complex shapes from SVG/PNG/JPG sources. Use `augment_images.sh`.

**Current:** cat, elephant, giraffe (+ bear, cactus, lightning_bolt pending)

---

## Workflow: Adding Image-Based Shapes

### Step 1: Prepare Source Images

1. Find or create silhouette images (solid black on white/transparent)
2. Create a folder: `assets/image_sources/elephant/`
3. Add 1-5 image variations to the folder
4. Supported formats: SVG, PNG, JPG, JPEG, TIFF, WebP

**Good image sources:**
- [The Noun Project](https://thenounproject.com/) - icons (check license)
- [Flaticon](https://www.flaticon.com/) - silhouettes
- [SVG Repo](https://www.svgrepo.com/) - free SVGs
- Create your own in Inkscape/Illustrator

### Step 2: Generate Training Data

```bash
# Generate 500 augmented images
./augment_images.sh assets/image_sources/elephant assets/training/svgs 500

# Output goes to: assets/training/svgs/elephant/
# Creates: elephant_001.png, elephant_002.png, ... elephant_500.png
```

The script applies random augmentations:
- Scale: 85-105%
- Shear: ±15°
- Horizontal flip: 50% chance
- Output: 64x64 grayscale PNG

### Step 3: Update Display Names

Edit `assets/training/svgs/display_names.json`:

```json
{
  "cat": "Cat",
  "elephant": "Elephant",
  "giraffe": "Giraffe",
  "bear": "Bear",
  "cactus": "Cactus",
  "lightning_bolt": "Lightning Bolt"
}
```

### Step 4: Retrain the Classifier

```bash
make train
```

This trains on all folders in:
- `assets/training/shapes/`
- `assets/training/letters/`
- `assets/training/svgs/`

### Step 5: Test the Game

```bash
make run
```

---

## Quick Example: Adding Bear, Cactus, Lightning Bolt

Assuming you have source images ready:

```bash
# Step 1: Generate training data for each shape
./augment_images.sh assets/image_sources/bear assets/training/svgs 500
./augment_images.sh assets/image_sources/cactus assets/training/svgs 500
./augment_images.sh assets/image_sources/lightning_bolt assets/training/svgs 500

# Step 2: Update display names file
cat > assets/training/svgs/display_names.json << 'EOF'
{
  "cat": "Cat",
  "elephant": "Elephant",
  "giraffe": "Giraffe",
  "bear": "Bear",
  "cactus": "Cactus",
  "lightning_bolt": "Lightning Bolt"
}
EOF

# Step 3: Train the classifier
make train

# Step 4: Play!
make run
```

---

## Workflow: Adding Generated Shapes

### Step 1: Add Shape Function

Edit `generate_shapes.py` and add a drawing function:

```python
def myshape(img: np.ndarray, cx: int, cy: int, size: int, thickness: int = None, **kwargs) -> np.ndarray:
    """Draw my custom shape."""
    # Draw using OpenCV primitives:
    # cv2.circle(), cv2.line(), cv2.polylines(), cv2.fillPoly(), etc.
    return img
```

### Step 2: Register the Shape

Add to the `SHAPES` dictionary:

```python
SHAPES = {
    # ... existing shapes ...
    "myshape": {"func": myshape, "display_name": "My Shape"},
}
```

### Step 3: Generate and Train

```bash
make generate-shapes   # Regenerates all shapes (500 each)
make train             # Retrain classifier
make run               # Test
```

---

## Workflow: Adding Letters

### Step 1: Edit Letter List

In `generate_letters.py`, add to `LETTERS`:

```python
LETTERS = [
    # ... existing letters ...
    ('letter_Z', 'Letter Z'),
]
```

### Step 2: Generate and Train

```bash
make generate-letters  # Regenerates all letters (500 each)
make train             # Retrain classifier
make run               # Test
```

---

## Display Names

Each training directory needs a `display_names.json` file that maps folder names to display names shown in the game:

```json
{
  "folder_name": "Display Name",
  "lightning_bolt": "Lightning Bolt",
  "letter_A": "Letter A"
}
```

If a shape is missing from display_names.json, the classifier will auto-generate a name:
- `letter_X` → "Letter X"
- `my_shape` → "My Shape"

---

## Capturing Real Silhouettes

During gameplay:
- Press **P** to manually capture current silhouette
- Enable `AUTO_CAPTURE_ON_MATCH = True` in game.py for automatic capture

Captures are saved to `data/captures/{shape}_{timestamp}.png`

To add captures to training data:
```bash
# Copy good captures to the appropriate training folder
cp data/captures/cat_*.png assets/training/svgs/cat/

# Regenerate (or just add to existing)
make train
```

---

## Troubleshooting

### Shape not appearing in game
- Check that training folder exists and has images
- Run `make train` to rebuild classifier
- Check `models/model_metadata.json` for class list

### Poor recognition accuracy
- Add more training variations (source images or captures)
- Check that shape is visually distinct from others
- Review captures to ensure they look like training data
- Try different source images with clearer silhouettes

### augment_images.sh errors
- Ensure `librsvg2-bin` and `imagemagick` are installed
- Check image files are valid (open in viewer to test)
- Verify output directory permissions (may need `sudo` or `make fix-permissions`)

### Permission errors
```bash
make fix-permissions   # Fix ownership of Docker-created files
```

---

## Make Commands Reference

```bash
make generate           # Generate shapes + letters
make generate-shapes    # Generate only shapes
make generate-letters   # Generate only letters
make generate-images SRC=... OUT=...  # Generate from source images
make train              # Train classifier on all training data
make retrain            # Clean + generate + train
make run                # Run the game
make fix-permissions    # Fix file ownership after Docker
```