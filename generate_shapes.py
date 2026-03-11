#!/usr/bin/env python3
"""
Generate training silhouettes for shape recognition.

Shapes are simple geometric forms easy for humans to pose.
Each shape is generated in two variants:
  - black_<shape>: dark shape on white background
  - white_<shape>: white shape on black background

Augmentation: random position, scale, and skew (no rotation).
"""

import os
import math
import random
import json
import cv2
import numpy as np
from typing import Callable


# Output settings
IMAGE_SIZE = 64
VARIATIONS_PER_SHAPE = 500

# Max skew angle in degrees applied as affine shear
MAX_SKEW_DEGREES = 15


# ---------------------------------------------------------------------------
# Shape drawing functions
# All draw a BLACK shape on a WHITE background.
# The white_* variant is created by inverting the image.
# Each function receives (img, cx, cy, size, thickness) and returns img.
# ---------------------------------------------------------------------------

def draw_circle(img, cx, cy, size, thickness=None, **kwargs):
    """Filled circle."""
    radius = size // 2
    cv2.circle(img, (cx, cy), radius, 0, -1)
    return img


def draw_square(img, cx, cy, size, thickness=None, **kwargs):
    """Filled axis-aligned square."""
    half = size // 2
    cv2.rectangle(img, (cx - half, cy - half), (cx + half, cy + half), 0, -1)
    return img


def draw_triangle(img, cx, cy, size, thickness=None, **kwargs):
    """Filled equilateral triangle, point up."""
    h = size * math.sqrt(3) / 2
    pts = np.array([
        [cx,              cy - int(h * 2 / 3)],
        [cx - size // 2,  cy + int(h * 1 / 3)],
        [cx + size // 2,  cy + int(h * 1 / 3)],
    ], dtype=np.int32)
    cv2.fillPoly(img, [pts], 0)
    return img


def draw_cross(img, cx, cy, size, thickness=None, **kwargs):
    """Plus-sign cross (+)."""
    arm = size // 2
    bar = max(4, size // 5)   # half-width of each bar
    pts = np.array([
        [cx - bar, cy - arm],
        [cx + bar, cy - arm],
        [cx + bar, cy - bar],
        [cx + arm, cy - bar],
        [cx + arm, cy + bar],
        [cx + bar, cy + bar],
        [cx + bar, cy + arm],
        [cx - bar, cy + arm],
        [cx - bar, cy + bar],
        [cx - arm, cy + bar],
        [cx - arm, cy - bar],
        [cx - bar, cy - bar],
    ], dtype=np.int32)
    cv2.fillPoly(img, [pts], 0)
    return img


def draw_x(img, cx, cy, size, thickness=None, **kwargs):
    """X shape (diagonal cross)."""
    arm = size // 2
    bar = max(3, size // 6)   # half-width of each diagonal bar
    # Rotate the cross polygon by 45°
    cos45 = math.cos(math.pi / 4)
    sin45 = math.sin(math.pi / 4)

    cross_pts = np.array([
        [-bar, -arm],
        [ bar, -arm],
        [ bar, -bar],
        [ arm, -bar],
        [ arm,  bar],
        [ bar,  bar],
        [ bar,  arm],
        [-bar,  arm],
        [-bar,  bar],
        [-arm,  bar],
        [-arm, -bar],
        [-bar, -bar],
    ], dtype=np.float32)

    rotated = np.array([
        [cx + int(p[0] * cos45 - p[1] * sin45),
         cy + int(p[0] * sin45 + p[1] * cos45)]
        for p in cross_pts
    ], dtype=np.int32)

    cv2.fillPoly(img, [rotated], 0)
    return img


def draw_star(img, cx, cy, size, thickness=None, **kwargs):
    """Filled 5-pointed star."""
    outer = size // 2
    inner = int(outer * 0.4)
    pts = []
    for i in range(10):
        r = outer if i % 2 == 0 else inner
        angle = -math.pi / 2 + i * math.pi / 5
        pts.append([int(cx + r * math.cos(angle)),
                    int(cy + r * math.sin(angle))])
    cv2.fillPoly(img, [np.array(pts, dtype=np.int32)], 0)
    return img


def draw_heart(img, cx, cy, size, thickness=None, **kwargs):
    """Filled heart."""
    pts = []
    for i in range(100):
        t = i * 2 * math.pi / 100
        x = 16 * (math.sin(t) ** 3)
        y = -(13 * math.cos(t) - 5 * math.cos(2 * t)
              - 2 * math.cos(3 * t) - math.cos(4 * t))
        scale = size / 35
        pts.append([int(cx + x * scale), int(cy + y * scale)])
    cv2.fillPoly(img, [np.array(pts, dtype=np.int32)], 0)
    return img



def draw_diamond(img, cx, cy, size, thickness=None, **kwargs):
    """Filled diamond (square rotated 45°)."""
    half = size // 2
    pts = np.array([
        [cx,        cy - half],
        [cx + half, cy],
        [cx,        cy + half],
        [cx - half, cy],
    ], dtype=np.int32)
    cv2.fillPoly(img, [pts], 0)
    return img


def draw_arrow_right(img, cx, cy, size, thickness=None, **kwargs):
    """Filled right-pointing arrow."""
    hw = size // 2
    shaft_h = max(3, size // 6)
    pts = np.array([
        [cx + hw,  cy],
        [cx,       cy - hw // 2],
        [cx,       cy - shaft_h],
        [cx - hw,  cy - shaft_h],
        [cx - hw,  cy + shaft_h],
        [cx,       cy + shaft_h],
        [cx,       cy + hw // 2],
    ], dtype=np.int32)
    cv2.fillPoly(img, [pts], 0)
    return img


def draw_arrow_left(img, cx, cy, size, thickness=None, **kwargs):
    """Filled left-pointing arrow."""
    hw = size // 2
    shaft_h = max(3, size // 6)
    pts = np.array([
        [cx - hw,  cy],
        [cx,       cy + hw // 2],
        [cx,       cy + shaft_h],
        [cx + hw,  cy + shaft_h],
        [cx + hw,  cy - shaft_h],
        [cx,       cy - shaft_h],
        [cx,       cy - hw // 2],
    ], dtype=np.int32)
    cv2.fillPoly(img, [pts], 0)
    return img


# ---------------------------------------------------------------------------
# Shape registry
# ---------------------------------------------------------------------------

BASE_SHAPES = {
    "circle":              {"func": draw_circle,      "display_name": "Circle"},
    "square":              {"func": draw_square,      "display_name": "Square"},
    "triangle":            {"func": draw_triangle,    "display_name": "Triangle"},
    "cross":               {"func": draw_cross,       "display_name": "Cross"},
    "x":                   {"func": draw_x,           "display_name": "X"},
    "star":                {"func": draw_star,        "display_name": "Star"},
    "heart":               {"func": draw_heart,       "display_name": "Heart"},

    "diamond":             {"func": draw_diamond,     "display_name": "Diamond"},
    "arrow_right":         {"func": draw_arrow_right, "display_name": "Arrow ->"},
    "arrow_left":          {"func": draw_arrow_left,  "display_name": "Arrow <-"},
}

# Expand into black_* and white_* variants
SHAPES = {}
for name, info in BASE_SHAPES.items():
    SHAPES[f"black_{name}"] = {
        "func": info["func"],
        "display_name": f"Black {info['display_name']}",
        "invert": False,
    }
    SHAPES[f"white_{name}"] = {
        "func": info["func"],
        "display_name": f"White {info['display_name']}",
        "invert": True,
    }


# ---------------------------------------------------------------------------
# Augmentation helpers
# ---------------------------------------------------------------------------

def apply_skew(img: np.ndarray) -> np.ndarray:
    """Apply a random affine shear (skew) to the image."""
    h, w = img.shape[:2]
    skew_x = random.uniform(-MAX_SKEW_DEGREES, MAX_SKEW_DEGREES)
    skew_y = random.uniform(-MAX_SKEW_DEGREES, MAX_SKEW_DEGREES)
    shear_x = math.tan(math.radians(skew_x))
    shear_y = math.tan(math.radians(skew_y))

    # Build affine matrix centred on image
    cx, cy = w / 2, h / 2
    M = np.array([
        [1,       shear_x, -shear_x * cy],
        [shear_y, 1,       -shear_y * cx],
    ], dtype=np.float32)

    bg_color = 255  # white background
    return cv2.warpAffine(img, M, (w, h),
                          borderMode=cv2.BORDER_CONSTANT,
                          borderValue=bg_color)


# ---------------------------------------------------------------------------
# Core generation
# ---------------------------------------------------------------------------

def generate_variation(
    shape_func: Callable,
    invert: bool = False,
    image_size: int = IMAGE_SIZE,
) -> np.ndarray:
    """Generate one training image with random position, scale, and skew."""
    # Always start with black shape on white background
    img = np.ones((image_size, image_size), dtype=np.uint8) * 255

    # Random size (30-70% of image)
    min_size = int(image_size * 0.3)
    max_size = int(image_size * 0.7)
    size = random.randint(min_size, max_size)

    margin = size // 2 + 4
    cx = random.randint(margin, image_size - margin)
    cy = random.randint(margin, image_size - margin)
    thickness = random.randint(max(2, size // 10), max(3, size // 5))

    img = shape_func(img, cx, cy, size, thickness=thickness)

    # Apply skew augmentation
    img = apply_skew(img)

    # Invert for white-on-black variant
    if invert:
        img = cv2.bitwise_not(img)

    return img


def generate_dataset(
    output_dir: str,
    shapes: dict = SHAPES,
    variations: int = VARIATIONS_PER_SHAPE,
):
    """Generate the full training dataset for all shapes."""
    os.makedirs(output_dir, exist_ok=True)
    display_names = {}

    for shape_name, shape_info in shapes.items():
        print(f"Generating {variations} variations of '{shape_name}'...")

        shape_dir = os.path.join(output_dir, shape_name)
        os.makedirs(shape_dir, exist_ok=True)

        for i in range(variations):
            img = generate_variation(
                shape_info["func"],
                invert=shape_info.get("invert", False),
                image_size=IMAGE_SIZE,
            )
            filepath = os.path.join(shape_dir, f"{shape_name}_{i:04d}.png")
            cv2.imwrite(filepath, img)

        display_names[shape_name] = shape_info["display_name"]
        print(f"  ✔ {shape_dir}/")

    meta_path = os.path.join(output_dir, "display_names.json")
    with open(meta_path, "w") as f:
        json.dump(display_names, f, indent=2)

    print(f"\nDataset complete! {len(shapes)} classes × {variations} images.")
    print(f"Display names saved to {meta_path}")
    return display_names


def preview_shapes(output_path: str = "shape_preview.png"):
    """Write a grid image showing sample variations of every shape."""
    cell = 80
    padding = 10
    rows_per_shape = 2
    cols = len(SHAPES)

    w = cols * cell + (cols + 1) * padding
    h = rows_per_shape * cell + (rows_per_shape + 1) * padding + 24

    preview = np.ones((h, w), dtype=np.uint8) * 200

    for col, (name, info) in enumerate(SHAPES.items()):
        x0 = padding + col * (cell + padding)

        # Label (short)
        label = name.replace("black_", "B:").replace("white_", "W:")
        cv2.putText(preview, label[:10], (x0, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, 0, 1)

        for row in range(rows_per_shape):
            y0 = 24 + padding + row * (cell + padding)
            tile = generate_variation(info["func"],
                                      invert=info.get("invert", False),
                                      image_size=cell)
            preview[y0:y0 + cell, x0:x0 + cell] = tile

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    cv2.imwrite(output_path, preview)
    print(f"Preview saved to {output_path}")
    return output_path


if __name__ == "__main__":
    print("=" * 50)
    print("SHAPE TRAINING DATA GENERATOR")
    print("=" * 50)
    print(f"\nShapes: {len(BASE_SHAPES)} base shapes × 2 color variants = {len(SHAPES)} classes")

    preview_out = "assets/training/shapes/preview.png"
    print(f"\nGenerating preview → {preview_out}")
    preview_shapes(preview_out)

    print("\nGenerating training dataset...")
    generate_dataset("assets/training/shapes")

    print("\n✔ Done!")
