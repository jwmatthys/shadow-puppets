#!/usr/bin/env bash

# Generate training data from all image source directories
# Usage: ./augment_all.sh [count]
# Default: 500 images per shape

COUNT="${1:-500}"
INPUT_BASE="assets/image_sources"
OUTPUT_BASE="assets/training/custom"

if [[ ! -d "$INPUT_BASE" ]]; then
  echo "ERROR: Input directory not found: $INPUT_BASE"
  exit 1
fi

# Find all subdirectories in image_sources
mapfile -t SHAPES < <(find "$INPUT_BASE" -mindepth 1 -maxdepth 1 -type d | sort)

if [[ ${#SHAPES[@]} -eq 0 ]]; then
  echo "No shape directories found in $INPUT_BASE"
  exit 1
fi

echo "Found ${#SHAPES[@]} shapes to process"
echo "Output: $OUTPUT_BASE"
echo "Images per shape: $COUNT"
echo ""

mkdir -p "$OUTPUT_BASE"

FAILED=()
for SHAPE_DIR in "${SHAPES[@]}"; do
  SHAPE_NAME=$(basename "$SHAPE_DIR")
  echo "=== Processing: $SHAPE_NAME ==="
  
  if ./augment_images.sh "$SHAPE_DIR" "$OUTPUT_BASE" "$COUNT"; then
    echo ""
  else
    echo "FAILED: $SHAPE_NAME"
    FAILED+=("$SHAPE_NAME")
    echo ""
  fi
done

echo "========================================"
echo "Done! Processed ${#SHAPES[@]} shapes"

if [[ ${#FAILED[@]} -gt 0 ]]; then
  echo "Failed: ${FAILED[*]}"
  exit 1
fi