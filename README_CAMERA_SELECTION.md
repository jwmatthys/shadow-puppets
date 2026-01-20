# Shadow Puppets - Camera Selection Guide

This update allows you to easily specify which camera device to use when running the game.

## Quick Start

### Use your USB camera (video2):
```bash
make game CAMERA=2
```

### Use built-in webcam (video0):
```bash
make game CAMERA=0
```

### Use any camera:
```bash
make game CAMERA=1
make live CAMERA=2
make test-camera CAMERA=3
```

## Files Modified

### 1. **game.py**
Added support for `CAMERA_DEVICE` environment variable:

```python
# Camera device - can be overridden with CAMERA_DEVICE environment variable
CAMERA_DEVICE = int(os.environ.get('CAMERA_DEVICE', '0'))
```

Then in the setup function:
```python
self.camera = Camera(device=CAMERA_DEVICE, width=CAPTURE_WIDTH, height=CAPTURE_HEIGHT)
```

### 2. **Makefile**
Added `CAMERA` parameter that:
- Mounts the correct `/dev/videoX` device in Docker
- Passes `CAMERA_DEVICE` environment variable to the game

```makefile
# Camera device (default: 0, override with: make game CAMERA=2)
CAMERA ?= 0

DOCKER_RUN_DISPLAY = docker run --rm -it \
    --device /dev/dri \
    --device /dev/video$(CAMERA) \
    -e CAMERA_DEVICE=$(CAMERA) \
    ...
```

### 3. **camera.py** (already supports int or string)
No changes needed - already accepts device numbers.

## Installation

1. Replace your existing files:
   ```bash
   cp game.py /path/to/your/shadow-puppets/
   cp Makefile /path/to/your/shadow-puppets/
   ```

2. Run the game with your USB camera:
   ```bash
   make game CAMERA=2
   ```

## Usage Examples

```bash
# Run game with USB camera
make game CAMERA=2

# Run live silhouette viewer with USB camera
make live CAMERA=2

# Test camera settings
make test-camera CAMERA=2

# Run benchmark with specific camera
make benchmark CAMERA=1

# Open shell with camera access
make shell CAMERA=2
```

## How It Works

When you run `make game CAMERA=2`:

1. **Makefile** sets `CAMERA=2`
2. Docker mounts `/dev/video2` into the container
3. Environment variable `CAMERA_DEVICE=2` is passed to the container
4. **game.py** reads `CAMERA_DEVICE` and initializes camera with device 2
5. **camera.py** opens `/dev/video2` via OpenCV

## Finding Your Camera Device

### List all video devices:
```bash
ls -l /dev/video*
```

### Get detailed device info:
```bash
v4l2-ctl --list-devices
```

Example output:
```
Integrated Camera: Integrated C (usb-0000:00:14.0-5):
	/dev/video0
	/dev/video1

USB2.0 HD UVC WebCam: USB2.0 HD (usb-0000:00:14.0-1):
	/dev/video2
	/dev/video3
```

### Test a specific camera:
```bash
# Test with VLC
vlc v4l2:///dev/video2

# Test with Python
python3 -c "import cv2; cap = cv2.VideoCapture(2); print('Works!' if cap.isOpened() else 'Failed'); cap.release()"
```

## Default Camera

If you want to change the default camera (so you don't have to type `CAMERA=2` every time):

### Option 1: Set in Makefile
Edit the Makefile and change line 6:
```makefile
CAMERA ?= 2  # Changed from 0 to 2
```

### Option 2: Set environment variable
```bash
export CAMERA=2
make game  # Will now use video2 by default
```

### Option 3: Create an alias
Add to your `~/.bashrc` or `~/.zshrc`:
```bash
alias shadow-puppets='make game CAMERA=2'
```

Then just run:
```bash
shadow-puppets
```

## Troubleshooting

### "Cannot find camera device"
- Check that the device exists: `ls -l /dev/video2`
- Make sure it's not in use by another application
- Try unplugging and replugging the USB camera

### Permission denied
Add your user to the video group:
```bash
sudo usermod -a -G video $USER
```
Then log out and back in.

### Docker can't see the camera
Make sure you're using the updated Makefile that includes:
```makefile
--device /dev/video$(CAMERA)
```

### Wrong camera opens
Double-check the device number:
```bash
v4l2-ctl --list-devices
```

Some cameras create multiple `/dev/videoX` entries - you may need to try video2, video3, etc.

## Complete Command Reference

```bash
# Game commands with camera selection
make game CAMERA=2          # Main game
make live CAMERA=2          # Live silhouette viewer
make test-camera CAMERA=2   # Camera test
make test-mediapipe CAMERA=2  # MediaPipe test
make benchmark CAMERA=2     # Performance benchmark
make shell CAMERA=2         # Interactive shell

# Without camera (training/building)
make build                  # Build Docker image
make generate              # Generate training data
make train                 # Train classifier
make retrain               # Clean + generate + train
make help                  # Show all commands
```

## Summary

✅ **Before**: Could only use `/dev/video0`  
✅ **Now**: Use any camera with `make game CAMERA=X`  
✅ **Best Practice**: `make game CAMERA=2` for your USB camera

That's it! Your Shadow Puppets game now has flexible camera selection. 🎮📷
