#!/usr/bin/env bash

# Usage:
# ./augment_images.sh assets/image_sources/elephant assets/training/svgs 500
#
# Supports: SVG, PNG, JPG, JPEG, TIFF, TIF, WebP

INPUT_DIR="$1"
OUTPUT_BASE="$2"
TOTAL_IMAGES="$3"

if [[ -z "$INPUT_DIR" || -z "$OUTPUT_BASE" || -z "$TOTAL_IMAGES" || ! -d "$INPUT_DIR" ]]; then
  echo "Usage: $0 <input_folder> <output_base_folder> <total_images>"
  echo "Example: $0 assets/image_sources/elephant assets/training/svgs 500"
  echo "Supports: SVG, PNG, JPG, JPEG, TIFF, TIF, WebP"
  exit 1
fi

CLASS_NAME=$(basename "$INPUT_DIR")
OUTPUT_DIR="$OUTPUT_BASE/$CLASS_NAME"
TMP_DIR="/tmp/${CLASS_NAME}_raster"

# Create output directory and clean existing files
mkdir -p "$OUTPUT_DIR"
rm -f "$OUTPUT_DIR"/*.png
mkdir -p "$TMP_DIR"

# Find all supported image files
mapfile -t IMAGES < <(find "$INPUT_DIR" -maxdepth 1 -type f \( \
  -iname "*.svg" -o \
  -iname "*.png" -o \
  -iname "*.jpg" -o \
  -iname "*.jpeg" -o \
  -iname "*.tiff" -o \
  -iname "*.tif" -o \
  -iname "*.webp" \
\) 2>/dev/null)

NUM_IMAGES=${#IMAGES[@]}

if [[ "$NUM_IMAGES" -eq 0 ]]; then
  echo "No supported images found in $INPUT_DIR"
  echo "Supports: SVG, PNG, JPG, JPEG, TIFF, TIF, WebP"
  exit 1
fi

echo "Found $NUM_IMAGES source images in $INPUT_DIR"
echo "Output: $OUTPUT_DIR"

IMAGES_PER_SRC=$((TOTAL_IMAGES / NUM_IMAGES))
REMAINDER=$((TOTAL_IMAGES % NUM_IMAGES))

COUNT=1

echo "Rasterizing source images..."

# --- Stage 1: Rasterize all sources to consistent PNG ---
for IMG in "${IMAGES[@]}"; do
  BASENAME=$(basename "$IMG")
  BASENAME_NOEXT="${BASENAME%.*}"
  EXT="${IMG##*.}"
  EXT_LOWER=$(echo "$EXT" | tr '[:upper:]' '[:lower:]')
  
  if [[ "$EXT_LOWER" == "svg" ]]; then
    # Use librsvg for SVG
    rsvg-convert \
      --background-color=white \
      --width=384 \
      --height=384 \
      "$IMG" \
      -o "$TMP_DIR/$BASENAME_NOEXT.png"
  else
    # Use ImageMagick for raster formats
    convert "$IMG" \
      -background white \
      -flatten \
      -resize 384x384 \
      -gravity center \
      -extent 384x384 \
      "$TMP_DIR/$BASENAME_NOEXT.png"
  fi
done

# --- Stage 2: Augmentation ---
MAX_RETRIES=8

mapfile -t TMP_PNGS < <(ls "$TMP_DIR"/*.png 2>/dev/null)

for INDEX in "${!TMP_PNGS[@]}"; do
  SRC_PNG="${TMP_PNGS[$INDEX]}"
  BASENAME=$(basename "$SRC_PNG" .png)

  N="$IMAGES_PER_SRC"
  [[ "$INDEX" -lt "$REMAINDER" ]] && N=$((N + 1))

  for ((i=0; i<N; i++)); do
    ATTEMPT=1

    while [[ "$ATTEMPT" -le "$MAX_RETRIES" ]]; do
      SCALE=$(awk -v min=85 -v max=105 'BEGIN{srand(); print min+rand()*(max-min)}')
      SHEAR_X=$(awk -v min=-15 -v max=15 'BEGIN{srand(); print min+rand()*(max-min)}')
      FLIP=$((RANDOM % 2))
      [[ "$FLIP" -eq 1 ]] && FLIP_OP="-flop" || FLIP_OP=""

      TMP_OUT=$(mktemp --suffix=.png)

      convert \
        -background white \
        "$SRC_PNG" \
        -resize "${SCALE}%" \
        -shear "${SHEAR_X}x0" \
        $FLIP_OP \
        -gravity center \
        -extent 512x512 \
        -colorspace Gray \
        -threshold 50% \
        -resize 64x64! \
        "$TMP_OUT" 2>/dev/null

      if [[ -s "$TMP_OUT" ]]; then
        OUTFILE=$(printf "%s/%s_%03d.png" "$OUTPUT_DIR" "$CLASS_NAME" "$COUNT")
        mv "$TMP_OUT" "$OUTFILE"
        COUNT=$((COUNT + 1))
        break
      fi

      rm -f "$TMP_OUT"
      ATTEMPT=$((ATTEMPT + 1))
    done

    if [[ "$ATTEMPT" -gt "$MAX_RETRIES" ]]; then
      echo "Warning: dropped one image from $BASENAME after $MAX_RETRIES failed attempts"
    fi
  done
done

rm -rf "$TMP_DIR"

echo "Done. Generated $((COUNT - 1)) training images in $OUTPUT_DIR"