# Shadow Puppets - Technical Design Document

**Version:** 1.0  
**Date:** January 12, 2026  
**Status:** Prototype Complete, Expanding Shape Library

---

## 1. Project Overview

### 1.1 Concept
Shadow Puppets is a party game where players use their body silhouettes to match target shapes displayed on screen. Teams work together to form shapes like letters, animals, objects, and geometric forms within a time limit.

### 1.2 Current Status
- Core gameplay loop implemented and functional
- Real-time silhouette detection using MediaPipe
- Shape classification using HOG features + SVM (scikit-learn)
- Improved silhouette processing with guided filtering
- Audio system (background music, sound effects)
- High score persistence and game logging
- Training data capture for iterative improvement

### 1.3 Target Platform
- Linux (Ubuntu 24) running in Docker
- Tested on Intel Celeron G5900T (low-end CPU, no GPU)
- Built-in webcam at 320x240 @ 15 FPS

---

## 2. System Architecture

### 2.1 Technology Stack

| Component | Technology |
|-----------|------------|
| Container | Docker (Python 3.10-slim) |
| Segmentation | MediaPipe Selfie Segmentation |
| Silhouette Processing | OpenCV (guided filter, temporal smoothing) |
| Classification | scikit-learn (HOG + SVM) |
| Display | Pygame |
| Audio | Pygame mixer + PulseAudio |

### 2.2 File Structure

```
shadow-puppets/
├── Dockerfile
├── Makefile
├── game.py                    # Main game loop
├── camera.py                  # Webcam capture
├── silhouette.py              # Original silhouette processor
├── improved_silhouette.py     # Guided filter processor
├── shape_classifier.py        # HOG + SVM classifier
├── generate_shapes.py         # Geometric shape training data
├── generate_letters.py        # Letter training data
├── augment_svgs.sh           # SVG to training data pipeline
├── fonts/                     # Custom fonts
│   ├── BOMBARD_.otf          # Headings
│   └── Carlito-Regular.ttf   # Body text
├── bgm/                       # Background music (.ogg)
├── sfx/                       # Sound effects
│   ├── boom.ogg              # Timer expired
│   └── bell.ogg              # Match achieved
├── assets/training/           # Training data
│   ├── shapes/               # Geometric shapes
│   └── letters/              # Letter silhouettes
├── svgs/                      # Source SVGs for shapes
│   ├── cat/
│   ├── elephant/
│   └── giraffe/
├── models/                    # Trained classifier
│   ├── shape_classifier.joblib
│   ├── scaler.joblib
│   └── model_metadata.json
└── data/                      # Runtime data
    ├── high_score.txt
    ├── logs/                  # Game session CSVs
    └── captures/              # Training captures
```

### 2.3 Processing Pipeline

```
Webcam (320x240)
    ↓
MediaPipe Segmentation (~6ms)
    ↓
Guided Filter + Temporal EMA (~6.5ms)
    ↓
Binary Silhouette (64x64)
    ↓
HOG Feature Extraction
    ↓
SVM Classification
    ↓
Display + Game Logic
```

---

## 3. Current Game Settings

| Setting | Value |
|---------|-------|
| Game Duration | 60 seconds |
| Shape Timeout | 15 seconds |
| Countdown Duration | 3 seconds |
| Match Threshold | 40% |
| Match Delay | 2 seconds |
| Training Variations | 500 per shape |

---

## 4. Silhouette Processing

### 4.1 Original Pipeline (silhouette.py)
- Gaussian blur on probability mask
- Temporal blending with previous frame
- Threshold to binary
- Morphological close (fills holes) ← **Problem: arms touching head = blob**
- Morphological open (removes noise)
- Contour approximation

### 4.2 Improved Pipeline (improved_silhouette.py)
- Temporal EMA smoothing on probability mask (α=0.6)
- Guided filter using RGB frame for edge preservation
- Threshold to binary
- Median filter for denoising
- Noise removal (preserves internal holes)

**Key improvement:** No morphological close, so gaps between body parts are preserved.

---

## 5. Training Data Generation

### 5.1 Geometric Shapes (generate_shapes.py)
- Programmatically generated with OpenCV
- Random position, size, thickness variations
- Current shapes: Ring, Delta (Triangle)

### 5.2 Letters (generate_letters.py)
- OpenCV text rendering with multiple fonts
- Random position, size, thickness
- Current letters: A, C, E, L, M, S, T, V, X, Y

### 5.3 SVG-Based Shapes (augment_svgs.sh)
- Source SVGs rasterized with librsvg
- ImageMagick augmentation: scale, shear, flip
- Output: 64x64 grayscale PNGs
- Current: Cat, Elephant, Giraffe

### 5.4 Gameplay Captures
- Press P during gameplay to capture silhouette
- Auto-capture on successful match (configurable)
- Saved to `data/captures/{shape}_{timestamp}.png`

---

## 6. Planned Shape Library

### 6.1 Letters (12)
A, F, H, J, L, M, S, T, V, W, X, Y

### 6.2 Geometric Shapes (5) - Generated
- Ring (circle outline)
- Triangle (delta)
- Heart
- Star (5-pointed)
- Moon (crescent)

### 6.3 Animals (10) - SVG-sourced
- Cat
- Elephant
- Giraffe
- Rabbit
- Bear
- Horse
- Kangaroo
- Crocodile
- Dinosaur (T-Rex)
- Bird (flying silhouette)

### 6.4 Objects (12) - SVG-sourced
- House
- Castle
- Bridge
- Chair
- Lamp
- Umbrella
- Glasses
- Camera
- Telephone
- Key
- Crown
- Anchor

### 6.5 Vehicles (4) - SVG-sourced
- Car
- Airplane
- Bicycle
- Boat/Ship

### 6.6 Nature (3) - SVG-sourced
- Tree
- Flower
- Mountain

### 6.7 Recommended Additions
Shapes that work well for human silhouettes (distinct, achievable poses):

**Easy (single person):**
- Cactus (arms up and bent)
- Rocket (arms up together, legs together)
- Arrow (pointing direction)
- Lightning bolt

**Medium (1-2 people):**
- Teapot (one person as body, arm as spout)
- Windmill (arms and legs spread)
- Scissors (two people crossing)

**Hard (2+ people):**
- Bridge (people arching together)
- Boat (people forming hull and sail)
- Elephant (multiple people for trunk, body, legs)

### 6.8 Shapes to Avoid
- **Too similar:** Guitar/Violin, Cup/Mug, Dog/Wolf
- **Too detailed:** Faces, hands, complex machinery
- **Impossible poses:** Shapes requiring disconnected parts
- **Ambiguous:** Shapes that look like multiple things

### 6.9 Total Target: ~45 shapes
- 12 letters
- 5 generated geometric
- ~28 SVG-sourced shapes

---

## 7. Development Roadmap

### 7.1 Phase 1: Shape Library Expansion (Current)
- [ ] Source SVGs for all planned shapes
- [ ] Generate training data (500 per shape)
- [ ] Retrain classifier
- [ ] Playtest and capture real silhouettes
- [ ] Iterate on difficult shapes

### 7.2 Phase 2: Gameplay Polish
- [ ] Difficulty levels (easy/medium/hard shapes)
- [ ] Shape categories (animals, letters, objects)
- [ ] Visual feedback improvements
- [ ] Tutorial mode

### 7.3 Phase 3: Multiplayer Features
- [ ] Team scoring
- [ ] Head-to-head mode
- [ ] Tournament brackets

### 7.4 Phase 4: Platform Expansion
- [ ] Windows/Mac support
- [ ] Standalone executable
- [ ] Web version (WebRTC + TensorFlow.js)

---

## 8. Build & Run Commands

```bash
# Initial setup
make build              # Build Docker image
make xhost              # Enable X11 (after reboot)

# Training
make generate           # Generate all training data
make train              # Train classifier
make retrain            # Clean + generate + train

# Running
make run                # Start the game
make live               # Live silhouette display

# Testing
make compare-silhouette # Compare original vs improved processing
make test-mediapipe     # Test segmentation

# SVG augmentation (manual)
./augment_svgs.sh svgs/cat 500
```

---

## 9. Configuration Reference

### 9.1 Game Settings (game.py)
```python
GAME_DURATION = 60           # Total game time (seconds)
SHAPE_TIMEOUT = 15           # Max time per shape
COUNTDOWN_DURATION = 3       # Countdown before shape
MATCH_THRESHOLD = 0.4        # 40% to match
MATCH_DELAY = 2.0            # Delay before match can trigger
AUTO_CAPTURE_ON_MATCH = True # Save silhouette on match
```

### 9.2 Silhouette Settings (improved_silhouette.py)
```python
threshold = 0.5              # Mask confidence threshold
guided_radius = 8            # Guided filter radius
guided_eps = 0.01            # Guided filter epsilon
temporal_alpha = 0.6         # EMA smoothing (0.6 = responsive)
min_contour_area = 300       # Noise removal threshold
```

### 9.3 Classifier Settings (shape_classifier.py)
```python
INPUT_SIZE = 64              # Image size for classification
HOG_CELL_SIZE = 8
HOG_BLOCK_SIZE = 2
HOG_BINS = 9
SVM kernel = 'rbf', C = 10
```

---

## 10. Known Issues & Limitations

1. **Classification accuracy:** Depends heavily on training data quality
2. **Similar shapes:** Some shapes may be confused (e.g., similar letters)
3. **Lighting sensitivity:** MediaPipe performance varies with lighting
4. **Multi-person detection:** Works but not optimized - see Section 11

---

## 11. Multi-Person Detection Optimization

### 11.1 Current Approach
MediaPipe Selfie Segmentation treats all people as a single foreground mask. This is ideal for Shadow Puppets since players combine their bodies into unified silhouettes.

### 11.2 Optimization Strategies

**A. MediaPipe Model Selection**
```python
# model_selection=0: General model (faster, less accurate)
# model_selection=1: Landscape model (current, better for multiple people)
SelfieSegmentation(model_selection=1)
```

**B. Higher Resolution Input**
Increase capture resolution for better edge detection with groups:
```python
CAPTURE_WIDTH = 640   # Up from 320
CAPTURE_HEIGHT = 480  # Up from 240
```
Trade-off: Higher CPU usage, but modern machines handle this easily.

**C. Post-Processing Tuning**
- Lower threshold to catch more edge pixels when people are close
- Increase guided filter radius to smooth small gaps
- Increase `min_contour_area` to ignore small disconnected fragments
- More temporal smoothing for stability with moving groups

### 11.3 Recommended Configuration for Groups

```python
# improved_silhouette.py - "group" preset
"group": {
    "threshold": 0.4,           # Lower threshold catches more edge pixels
    "guided_radius": 12,        # Larger radius smooths gaps
    "guided_eps": 0.02,         # Slightly more smoothing
    "temporal_alpha": 0.5,      # More temporal smoothing for stability
    "min_contour_area": 500,    # Ignore small fragments
    "denoise_strength": 5,
}
```

### 11.4 Testing Multi-Person Performance
```bash
make compare-silhouette  # Visual comparison with multiple people
make benchmark           # Measure FPS with different group sizes
```

---

## 12. Appendix: Display Name Mapping

Shape class names map to display names for UI:

| Class Name | Display Name |
|------------|--------------|
| ring | Ring |
| delta | Triangle |
| letter_A | Letter A |
| letter_M | Letter M |
| cat | Cat |
| elephant | Elephant |
| giraffe | Giraffe |

Display names are stored in `model_metadata.json` and loaded by the classifier.
