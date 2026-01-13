#!/usr/bin/env python3
"""
Generate training silhouettes for letter recognition.
Creates variations of capital letters with different positions, scales, and fonts.
"""

import os
import random
import cv2
import numpy as np
from typing import List

# Output settings
IMAGE_SIZE = 64
VARIATIONS_PER_LETTER = 500

# Letters that are distinct and feasible to form with bodies
# Format: (folder_name, display_name)
LETTERS = [
    ('letter_A', 'Letter A'),
    ('letter_F', 'Letter F'),
    ('letter_H', 'Letter H'),
    ('letter_J', 'Letter J'),
    ('letter_L', 'Letter L'),
    ('letter_M', 'Letter M'),
    ('letter_S', 'Letter S'),
    ('letter_T', 'Letter T'),
    ('letter_V', 'Letter V'),
    ('letter_W', 'Letter W'),
    ('letter_X', 'Letter X'),
    ('letter_Y', 'Letter Y'),
]

# OpenCV font options for variety
FONTS = [
    cv2.FONT_HERSHEY_SIMPLEX,
    cv2.FONT_HERSHEY_DUPLEX,
    cv2.FONT_HERSHEY_COMPLEX,
    cv2.FONT_HERSHEY_TRIPLEX,
]


def draw_letter(
    img: np.ndarray,
    letter: str,
    cx: int,
    cy: int,
    size: float,
    thickness: int,
    font: int,
) -> np.ndarray:
    """Draw a letter centered at (cx, cy)."""
    # Get text size to center it
    (text_width, text_height), baseline = cv2.getTextSize(letter, font, size, thickness)
    
    # Calculate position to center the letter
    x = cx - text_width // 2
    y = cy + text_height // 2
    
    cv2.putText(img, letter, (x, y), font, size, 0, thickness, cv2.LINE_AA)
    
    return img


def generate_letter_variation(letter: str, image_size: int = IMAGE_SIZE) -> np.ndarray:
    """Generate a single letter variation with random transforms."""
    # Start with white background
    img = np.ones((image_size, image_size), dtype=np.uint8) * 255
    
    # Random font
    font = random.choice(FONTS)
    
    # Random thickness - chunkier letters
    thickness = random.randint(4, 8)
    
    # Random size (scale factor for font) - adjusted for thickness
    size = random.uniform(0.8, 1.5)
    
    # Calculate approximate letter dimensions to keep in bounds
    (text_width, text_height), _ = cv2.getTextSize(letter, font, size, thickness)
    
    # Calculate margins, ensuring we have room to place the letter
    margin_x = text_width // 2 + 2
    margin_y = text_height // 2 + 2
    
    # Ensure valid range for random position
    min_x = margin_x
    max_x = max(margin_x, image_size - margin_x)
    min_y = margin_y
    max_y = max(margin_y, image_size - margin_y)
    
    # If letter is too big, just center it
    if min_x >= max_x:
        cx = image_size // 2
    else:
        cx = random.randint(min_x, max_x)
    
    if min_y >= max_y:
        cy = image_size // 2
    else:
        cy = random.randint(min_y, max_y)
    
    # Draw letter
    img = draw_letter(img, letter, cx, cy, size, thickness, font)
    
    return img


def generate_letter_dataset(
    output_dir: str,
    letters: list = LETTERS,
    variations: int = VARIATIONS_PER_LETTER,
):
    """Generate complete letter training dataset."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Build metadata for display names
    metadata = {}
    
    for folder_name, display_name in letters:
        # Extract the actual letter character
        letter = folder_name.split('_')[1]
        
        print(f"Generating {variations} variations of {display_name}...")
        
        letter_dir = os.path.join(output_dir, folder_name)
        os.makedirs(letter_dir, exist_ok=True)
        
        for i in range(variations):
            img = generate_letter_variation(letter)
            
            filename = f"{folder_name}_{i:04d}.png"
            filepath = os.path.join(letter_dir, filename)
            cv2.imwrite(filepath, img)
        
        # Store display name mapping
        metadata[folder_name] = display_name
        
        print(f"  ✓ Saved to {letter_dir}/")
    
    # Save metadata
    import json
    meta_path = os.path.join(output_dir, "display_names.json")
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"\nSaved display names to {meta_path}")
    
    print(f"\nLetter dataset complete!")
    return letters


def preview_letters(output_path: str = "letter_preview.png"):
    """Generate a preview image showing all letters."""
    cols = len(LETTERS)
    rows = 3
    cell_size = 80
    padding = 10
    
    width = cols * cell_size + (cols + 1) * padding
    height = rows * cell_size + (rows + 1) * padding + 30
    
    preview = np.ones((height, width), dtype=np.uint8) * 240
    
    for col, (folder_name, display_name) in enumerate(LETTERS):
        letter = folder_name.split('_')[1]
        x_offset = padding + col * (cell_size + padding)
        
        # Draw label
        cv2.putText(
            preview,
            letter,
            (x_offset + cell_size // 3, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            0,
            2,
        )
        
        # Draw variations
        for row in range(rows):
            y_offset = 30 + padding + row * (cell_size + padding)
            
            # Generate letter at cell_size
            img = generate_letter_variation(letter, image_size=cell_size)
            
            # Place in preview
            preview[y_offset:y_offset + cell_size, x_offset:x_offset + cell_size] = img
    
    cv2.imwrite(output_path, preview)
    print(f"Preview saved to {output_path}")
    return output_path


if __name__ == "__main__":
    import sys
    
    print("=" * 50)
    print("LETTER TRAINING DATA GENERATOR")
    print("=" * 50)
    print(f"\nLetters: {LETTERS}")
    
    # Generate preview first
    print("\nGenerating preview...")
    os.makedirs("assets/training/letters", exist_ok=True)
    preview_letters("assets/training/letters/preview.png")
    
    # Generate full dataset
    print("\nGenerating training dataset...")
    generate_letter_dataset("assets/training/letters")
    
    print("\n✓ Done!")
