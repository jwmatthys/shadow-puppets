#!/usr/bin/env python3
"""
Package training data into a zip file for upload to Google Colab.
"""

import os
import zipfile


def package_training_data(
    data_dir: str = "assets/training/shapes",
    output_file: str = "shapes_training_data.zip"
):
    """Create a zip file of training data."""
    
    if not os.path.exists(data_dir):
        print(f"ERROR: Data directory not found: {data_dir}")
        print("Run 'make generate' first!")
        return False
    
    print(f"Packaging training data from {data_dir}...")
    
    file_count = 0
    with zipfile.ZipFile(output_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file.endswith('.png'):
                    file_path = os.path.join(root, file)
                    # Store with relative path from data_dir
                    arc_name = os.path.relpath(file_path, data_dir)
                    zipf.write(file_path, arc_name)
                    file_count += 1
    
    size_mb = os.path.getsize(output_file) / (1024 * 1024)
    print(f"Created {output_file}")
    print(f"  {file_count} images")
    print(f"  {size_mb:.1f} MB")
    
    print(f"\nUpload this file to Google Colab to train the model.")
    return True


if __name__ == "__main__":
    package_training_data()
