#!/usr/bin/env bash

# Usage:
# ./augment_svgs.sh svgs/cat 500

INPUT_DIR="$1"
TOTAL_IMAGES="$2"

if [[ -z "$INPUT_DIR" || -z "$TOTAL_IMAGES" || ! -d "$INPUT_DIR" ]]; then
  echo "Usage: $0 <svg_folder> <total_images>"
  exit 1
fi

CLASS_NAME=$(basename "$INPUT_DIR")
OUTPUT_DIR="training_data/$CLASS_NAME"
TMP_DIR="/tmp/${CLASS_NAME}_raster"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$TMP_DIR"

# Clean output
rm -f "$OUTPUT_DIR"/*.png
rm -f "$TMP_DIR"/*.png

mapfile -t SVGS < <(ls "$INPUT_DIR"/*.svg 2>/dev/null)
NUM_SVGS=${#SVGS[@]}

if [[ "$NUM_SVGS" -eq 0 ]]; then
  echo "No SVGs found in $INPUT_DIR"
  exit 1
fi

IMAGES_PER_SVG=$((TOTAL_IMAGES / NUM_SVGS))
REMAINDER=$((TOTAL_IMAGES % NUM_SVGS))

COUNT=1

echo "Rasterizing SVGs with librsvg…"

# --- Stage 1: Reliable rasterization ---
for SVG in "${SVGS[@]}"; do
  BASENAME=$(basename "$SVG" .svg)
  rsvg-convert \
    --background-color=white \
    --width=384 \
    --height=384 \
    "$SVG" \
    -o "$TMP_DIR/$BASENAME.png"
done

# --- Stage 2: Augmentation ---
MAX_RETRIES=8

for INDEX in "${!SVGS[@]}"; do
  SVG="${SVGS[$INDEX]}"
  BASENAME=$(basename "$SVG" .svg)
  SRC_PNG="$TMP_DIR/$BASENAME.png"

  N="$IMAGES_PER_SVG"
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

echo "Done. Generated $((COUNT - 1)) valid images in $OUTPUT_DIR"
