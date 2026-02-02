# Shape Classifier Border Improvement

## Problem

The HOG-based shape classifier was underrating live camera silhouettes compared to training data. The key issue:

- **Live camera input**: People standing in front of the camera create silhouettes that touch the bottom edge of the frame
- **Training data**: Generated shapes (rings, triangles, letters, etc.) have padding around them and don't touch the edges
- **Result**: Distribution mismatch causes the classifier to rate live silhouettes lower than it should

## Solution

Add a 2-pixel black border around the silhouette before HOG feature extraction. This:

1. **Shrinks the shape slightly** - providing padding around edge-touching silhouettes
2. **Matches training data distribution** - making live input look more like training examples
3. **Doesn't affect display** - border is added only for classification, not shown to users
4. **Works for both training and inference** - consistent preprocessing in both cases

## Implementation

Modified `shape_classifier.py` in the `extract_hog_features()` function:

```python
# Add 2-pixel white border to match training data expectations
border_width = 2
image = cv2.copyMakeBorder(
    image, 
    border_width, border_width, border_width, border_width,
    cv2.BORDER_CONSTANT, 
    value=255  # White border (background color)
)

# Resize back to INPUT_SIZE (border added 4 pixels in each dimension)
image = cv2.resize(image, (INPUT_SIZE, INPUT_SIZE))
```

The border is added **after** converting to grayscale but **before** computing HOG features.

## Visual Effect

See `border_effect_visualization.png` for a comparison:
- **Left**: Original silhouette touching bottom edge (typical for standing people)
- **Right**: Same silhouette with 2-pixel border added and resized back to 64x64

The shape is slightly smaller and has padding around it, matching the training data distribution.

## Next Steps

**Important:** You'll need to retrain the classifier for this change to take full effect:

```bash
make train
```

This ensures both training and inference use the same border preprocessing.

## Expected Improvements

- Better recognition of standing poses (legs, full-body shapes)
- More consistent confidence scores across different poses
- Reduced false negatives when people are close to camera
- Better overall accuracy matching the training distribution

## Technical Notes

- Border width: 2 pixels (on each side, so 4 pixels added to each dimension)
- Border color: White (255) to match the background in training images
- Applied in: `extract_hog_features()` function in `shape_classifier.py`
- Affects: Both training and inference (consistent preprocessing)
- No visual impact: Border only added for classification, not for display
