#!/usr/bin/env python3
"""
Generate training silhouettes for shape recognition.
Creates variations of each shape with different positions, scales, and rotations.
"""

import os
import math
import random
import cv2
import numpy as np
from typing import List, Tuple, Callable


# Output settings
IMAGE_SIZE = 64  # Small for fast training, will resize input to match
VARIATIONS_PER_SHAPE = 200  # Number of training images per shape


def filled_circle(img: np.ndarray, cx: int, cy: int, size: int, **kwargs) -> np.ndarray:
    """Draw a filled circle."""
    radius = size // 2
    cv2.circle(img, (cx, cy), radius, 0, -1)
    return img


def filled_square(img: np.ndarray, cx: int, cy: int, size: int, **kwargs) -> np.ndarray:
    """Draw a filled square (no rotation - axis aligned)."""
    half = size // 2
    top_left = (cx - half, cy - half)
    bottom_right = (cx + half, cy + half)
    cv2.rectangle(img, top_left, bottom_right, 0, -1)
    return img


def filled_triangle(img: np.ndarray, cx: int, cy: int, size: int, **kwargs) -> np.ndarray:
    """Draw a filled equilateral triangle with horizontal base at bottom."""
    # Equilateral triangle vertices (pointing up, flat bottom)
    h = size * math.sqrt(3) / 2
    pts = np.array([
        [cx, cy - int(h * 2/3)],           # Top vertex
        [cx - size//2, cy + int(h * 1/3)], # Bottom left
        [cx + size//2, cy + int(h * 1/3)], # Bottom right
    ], dtype=np.int32)
    
    cv2.fillPoly(img, [pts], 0)
    return img


def ring(img: np.ndarray, cx: int, cy: int, size: int, thickness: int = None, **kwargs) -> np.ndarray:
    """Draw a ring (outline circle)."""
    radius = size // 2
    if thickness is None:
        thickness = max(2, size // 8)
    cv2.circle(img, (cx, cy), radius, 0, thickness)
    return img


def frame(img: np.ndarray, cx: int, cy: int, size: int, thickness: int = None, **kwargs) -> np.ndarray:
    """Draw a frame (outline square, no rotation - axis aligned)."""
    if thickness is None:
        thickness = max(2, size // 8)
    
    half = size // 2
    top_left = (cx - half, cy - half)
    bottom_right = (cx + half, cy + half)
    cv2.rectangle(img, top_left, bottom_right, 0, thickness)
    return img


def delta(img: np.ndarray, cx: int, cy: int, size: int, thickness: int = None, **kwargs) -> np.ndarray:
    """Draw a delta (outline triangle) with horizontal base at bottom."""
    if thickness is None:
        thickness = max(2, size // 8)
    
    # Equilateral triangle vertices (pointing up, flat bottom)
    h = size * math.sqrt(3) / 2
    pts = np.array([
        [cx, cy - int(h * 2/3)],           # Top vertex
        [cx - size//2, cy + int(h * 1/3)], # Bottom left
        [cx + size//2, cy + int(h * 1/3)], # Bottom right
    ], dtype=np.int32)
    
    cv2.polylines(img, [pts], True, 0, thickness)
    return img


# Shape registry
SHAPES = {
    "filled_circle": {"func": filled_circle, "difficulty": "easy"},
    "filled_square": {"func": filled_square, "difficulty": "easy"},
    "filled_triangle": {"func": filled_triangle, "difficulty": "easy"},
    "ring": {"func": ring, "difficulty": "medium"},
    "frame": {"func": frame, "difficulty": "medium"},
    "delta": {"func": delta, "difficulty": "medium"},
}


def generate_variation(
    shape_func: Callable,
    image_size: int = IMAGE_SIZE,
) -> np.ndarray:
    """Generate a single shape variation with random transforms."""
    # Start with white background
    img = np.ones((image_size, image_size), dtype=np.uint8) * 255
    
    # Random size (30-70% of image to leave room for margins)
    min_size = int(image_size * 0.3)
    max_size = int(image_size * 0.7)
    size = random.randint(min_size, max_size)
    
    # Calculate margin needed to keep shape fully in bounds
    # For triangles, height is size * sqrt(3) / 2, and center is offset
    # Use conservative margin based on size
    margin = size // 2 + 4  # Extra padding for outline thickness
    
    # Random position (keep shape fully visible)
    cx = random.randint(margin, image_size - margin)
    cy = random.randint(margin, image_size - margin)
    
    # Random thickness for outline shapes
    thickness = random.randint(max(2, size // 10), max(3, size // 5))
    
    # Draw shape
    img = shape_func(img, cx, cy, size, thickness=thickness)
    
    return img


def generate_dataset(output_dir: str, shapes: dict = SHAPES, variations: int = VARIATIONS_PER_SHAPE):
    """Generate complete training dataset."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create category metadata file
    metadata = {
        "category": "shapes",
        "shapes": {},
    }
    
    for shape_name, shape_info in shapes.items():
        print(f"Generating {variations} variations of {shape_name}...")
        
        shape_dir = os.path.join(output_dir, shape_name)
        os.makedirs(shape_dir, exist_ok=True)
        
        for i in range(variations):
            img = generate_variation(
                shape_info["func"],
            )
            
            filename = f"{shape_name}_{i:04d}.png"
            filepath = os.path.join(shape_dir, filename)
            cv2.imwrite(filepath, img)
        
        metadata["shapes"][shape_name] = {
            "difficulty": shape_info["difficulty"],
            "count": variations,
        }
        
        print(f"  ✓ Saved to {shape_dir}/")
    
    # Save metadata
    import json
    meta_path = os.path.join(output_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nDataset complete! Metadata saved to {meta_path}")
    return metadata


def preview_shapes(output_path: str = "shape_preview.png"):
    """Generate a preview image showing all shapes."""
    cols = len(SHAPES)
    rows = 3  # Show 3 variations per shape
    cell_size = 80
    padding = 10
    
    width = cols * cell_size + (cols + 1) * padding
    height = rows * cell_size + (rows + 1) * padding + 30  # Extra for labels
    
    preview = np.ones((height, width), dtype=np.uint8) * 240
    
    for col, (shape_name, shape_info) in enumerate(SHAPES.items()):
        x_offset = padding + col * (cell_size + padding)
        
        # Draw label
        cv2.putText(
            preview, 
            shape_name[:8],  # Truncate long names
            (x_offset, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            0,
            1,
        )
        
        # Draw variations
        for row in range(rows):
            y_offset = 30 + padding + row * (cell_size + padding)
            
            # Generate shape at cell_size
            img = generate_variation(
                shape_info["func"],
                image_size=cell_size,
            )
            
            # Place in preview
            preview[y_offset:y_offset + cell_size, x_offset:x_offset + cell_size] = img
    
    cv2.imwrite(output_path, preview)
    print(f"Preview saved to {output_path}")
    return output_path


if __name__ == "__main__":
    import sys
    
    print("=" * 50)
    print("SHAPE TRAINING DATA GENERATOR")
    print("=" * 50)
    
    # Generate preview first
    print("\nGenerating preview...")
    preview_shapes("assets/training/shapes/preview.png")
    
    # Generate full dataset
    print("\nGenerating training dataset...")
    generate_dataset("assets/training/shapes")
    
    print("\n✓ Done!")