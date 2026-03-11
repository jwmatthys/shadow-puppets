#!/usr/bin/env python3
"""
Generate training silhouettes for letter recognition.

Letters are chosen for vertical symmetry so they're unambiguous to pose:
  A, H, T, V, X, Y

Each letter is generated in two variants:
  - black_letter_<L>: dark letter on white background
  - white_letter_<L>: white letter on black background

Augmentation: random position, scale, font, and skew (no rotation).
"""

import os
import random
import json
import cv2
import numpy as np


# Output settings
IMAGE_SIZE = 64
VARIATIONS_PER_LETTER = 500

# Max skew angle in degrees applied as affine shear (matches generate_shapes.py)
MAX_SKEW_DEGREES = 15

# Vertically symmetrical letters, easy to form with a body
BASE_LETTERS = [
    ('letter_A', 'A', 'Letter A'),
    ('letter_H', 'H', 'Letter H'),
    ('letter_T', 'T', 'Letter T'),
    ('letter_V', 'V', 'Letter V'),
    ('letter_X', 'X', 'Letter X'),
    ('letter_Y', 'Y', 'Letter Y'),
]

# OpenCV font options for variety
FONTS = [
    cv2.FONT_HERSHEY_SIMPLEX,
    cv2.FONT_HERSHEY_DUPLEX,
    cv2.FONT_HERSHEY_COMPLEX,
    cv2.FONT_HERSHEY_TRIPLEX,
]

# Expand into black_* and white_* variants
# Each entry: (folder_name, char, display_name, invert)
LETTERS = []
for folder, char, display in BASE_LETTERS:
    LETTERS.append((f"black_{folder}", char, f"Black {display}", False))
    LETTERS.append((f"white_{folder}", char, f"White {display}", True))


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------

def apply_skew(img: np.ndarray, bg_color: int = 255) -> np.ndarray:
    """Apply a random affine shear (skew) to the image."""
    import math
    h, w = img.shape[:2]
    skew_x = random.uniform(-MAX_SKEW_DEGREES, MAX_SKEW_DEGREES)
    skew_y = random.uniform(-MAX_SKEW_DEGREES, MAX_SKEW_DEGREES)
    shear_x = math.tan(math.radians(skew_x))
    shear_y = math.tan(math.radians(skew_y))

    cx, cy = w / 2, h / 2
    M = np.array([
        [1,       shear_x, -shear_x * cy],
        [shear_y, 1,       -shear_y * cx],
    ], dtype=np.float32)

    return cv2.warpAffine(img, M, (w, h),
                          borderMode=cv2.BORDER_CONSTANT,
                          borderValue=bg_color)


# ---------------------------------------------------------------------------
# Core generation
# ---------------------------------------------------------------------------

def generate_letter_variation(
    char: str,
    invert: bool = False,
    image_size: int = IMAGE_SIZE,
) -> np.ndarray:
    """Generate one training image with random position, scale, font, and skew."""
    # Always start black-on-white
    img = np.ones((image_size, image_size), dtype=np.uint8) * 255

    font = random.choice(FONTS)
    thickness = random.randint(4, 8)
    scale = random.uniform(0.8, 1.5)

    (tw, th), _ = cv2.getTextSize(char, font, scale, thickness)

    margin_x = tw // 2 + 2
    margin_y = th // 2 + 2
    min_x = margin_x
    max_x = max(margin_x, image_size - margin_x)
    min_y = margin_y
    max_y = max(margin_y, image_size - margin_y)

    cx = image_size // 2 if min_x >= max_x else random.randint(min_x, max_x)
    cy = image_size // 2 if min_y >= max_y else random.randint(min_y, max_y)

    x = cx - tw // 2
    y = cy + th // 2
    cv2.putText(img, char, (x, y), font, scale, 0, thickness, cv2.LINE_AA)

    # Skew augmentation (white background for black-on-white)
    img = apply_skew(img, bg_color=255)

    # Invert for white-on-black variant
    if invert:
        img = cv2.bitwise_not(img)

    return img


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------

def generate_letter_dataset(
    output_dir: str,
    letters: list = LETTERS,
    variations: int = VARIATIONS_PER_LETTER,
):
    """Generate the full training dataset for all letter variants."""
    os.makedirs(output_dir, exist_ok=True)
    display_names = {}

    for folder_name, char, display_name, invert in letters:
        print(f"Generating {variations} variations of '{folder_name}'...")

        letter_dir = os.path.join(output_dir, folder_name)
        os.makedirs(letter_dir, exist_ok=True)

        for i in range(variations):
            img = generate_letter_variation(char, invert=invert, image_size=IMAGE_SIZE)
            filepath = os.path.join(letter_dir, f"{folder_name}_{i:04d}.png")
            cv2.imwrite(filepath, img)

        display_names[folder_name] = display_name
        print(f"  ✔ {letter_dir}/")

    meta_path = os.path.join(output_dir, "display_names.json")
    with open(meta_path, "w") as f:
        json.dump(display_names, f, indent=2)

    print(f"\nDataset complete! {len(letters)} classes × {variations} images.")
    print(f"Display names saved to {meta_path}")
    return display_names


def preview_letters(output_path: str = "letter_preview.png"):
    """Write a grid image showing sample variations of every letter variant."""
    cell = 80
    padding = 10
    rows_per_letter = 2
    cols = len(LETTERS)

    w = cols * cell + (cols + 1) * padding
    h = rows_per_letter * cell + (rows_per_letter + 1) * padding + 24

    preview = np.ones((h, w), dtype=np.uint8) * 200

    for col, (folder_name, char, display_name, invert) in enumerate(LETTERS):
        x0 = padding + col * (cell + padding)
        label = ("B:" if not invert else "W:") + char
        cv2.putText(preview, label, (x0, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, 0, 1)

        for row in range(rows_per_letter):
            y0 = 24 + padding + row * (cell + padding)
            tile = generate_letter_variation(char, invert=invert, image_size=cell)
            preview[y0:y0 + cell, x0:x0 + cell] = tile

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    cv2.imwrite(output_path, preview)
    print(f"Preview saved to {output_path}")
    return output_path


if __name__ == "__main__":
    print("=" * 50)
    print("LETTER TRAINING DATA GENERATOR")
    print("=" * 50)
    print(f"\nLetters: {[c for _, c, _ in BASE_LETTERS]} "
          f"({len(BASE_LETTERS)} base × 2 color variants = {len(LETTERS)} classes)")

    preview_out = "assets/training/letters/preview.png"
    print(f"\nGenerating preview → {preview_out}")
    preview_letters(preview_out)

    print("\nGenerating training dataset...")
    generate_letter_dataset("assets/training/letters")

    print("\n✔ Done!")
