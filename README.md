# Shadow Puppets - Development Setup

## Prerequisites
- Docker installed on your HP All-in-One
- Webcam (built-in or USB)

## Docker Commands

### 1. Build the Docker Image
```bash
cd shadow-puppets
docker build -t shadow-puppets .
```
This will take a few minutes the first time (downloading Python, MediaPipe, etc.)

### 2. Test Display Output
This tests that Pygame can create a window through Docker:
```bash
docker run --rm -it \
    --device /dev/dri \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v $(pwd):/app \
    shadow-puppets \
    python test_display.py
```

**If you get a display error**, first run this on your host (outside Docker):
```bash
xhost +local:docker
```

### 3. Test MediaPipe Segmentation
This tests MediaPipe with webcam:
```bash
docker run --rm -it \
    --device /dev/dri \
    --device /dev/video0 \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v $(pwd):/app \
    shadow-puppets \
    python test_mediapipe.py
```

**Note:** Your webcam might be `/dev/video0`, `/dev/video1`, etc. Check with:
```bash
ls -la /dev/video*
```

### 4. Run Interactively (for debugging)
```bash
docker run --rm -it \
    --device /dev/dri \
    --device /dev/video0 \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v $(pwd):/app \
    shadow-puppets \
    bash
```
Then you can run Python interactively to debug.

## Troubleshooting

### "Cannot open display" error
1. Run `xhost +local:docker` on your host
2. Make sure you're running from a terminal in your desktop environment (not SSH)

### "No webcam available"
1. Check `ls -la /dev/video*` to find your webcam device
2. Make sure no other app is using the webcam
3. Try `/dev/video0`, `/dev/video1`, etc.

### MediaPipe silent failures
The test_mediapipe.py script has detailed diagnostics. Run it and check:
- Are imports succeeding?
- Is the SelfieSegmentation object created?
- Is process() returning a mask?

### Slow performance
On your Celeron, expect:
- ~50-100ms per frame for segmentation at 320x240
- 10-20 FPS effective rate
We'll optimize further once the basics work!

## Project Structure
```
shadow-puppets/
├── Dockerfile          # Docker image definition
├── README.md           # This file
├── test_display.py     # Test Pygame display
├── test_mediapipe.py   # Test MediaPipe segmentation
└── main.py             # Main game (coming soon)
```

## Next Steps
Once these tests pass:
1. Build the game loop with real-time silhouette display
2. Create the shape matching neural network
3. Add game modes, scoring, and UI
