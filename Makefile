# Shadow Puppets - Makefile
# Simplifies common Docker commands

IMAGE_NAME = shadow-puppets

# Camera device (default: 0, override with: make game CAMERA=2)
CAMERA ?= 0

DOCKER_RUN = docker run --rm -it -v $(PWD):/app $(IMAGE_NAME)
DOCKER_RUN_DISPLAY = docker run --rm -it \
	--device /dev/dri \
	--device /dev/video$(CAMERA) \
	-e DISPLAY=$(DISPLAY) \
	-e CAMERA_DEVICE=$(CAMERA) \
	-e PULSE_SERVER=unix:$(XDG_RUNTIME_DIR)/pulse/native \
	-v /tmp/.X11-unix:/tmp/.X11-unix \
	-v $(XDG_RUNTIME_DIR)/pulse/native:$(XDG_RUNTIME_DIR)/pulse/native \
	-v $(PWD):/app \
	$(IMAGE_NAME)

.PHONY: build run game test-display test-mediapipe test-smoothing test-camera \
        benchmark generate train clean-training clean help xhost

# Enable X11 access for Docker (run once after reboot)
xhost:
	xhost +local:docker

# Build the Docker image
build:
	docker build -t $(IMAGE_NAME) .

# Run the main game
run: game

game:
	@echo "Using camera: /dev/video$(CAMERA)"
	$(DOCKER_RUN_DISPLAY) python game.py

# Run live silhouette display
live:
	@echo "Using camera: /dev/video$(CAMERA)"
	$(DOCKER_RUN_DISPLAY) python live_silhouette.py

# Test commands
test-display:
	$(DOCKER_RUN_DISPLAY) python test_display.py

test-mediapipe:
	@echo "Using camera: /dev/video$(CAMERA)"
	$(DOCKER_RUN_DISPLAY) python test_mediapipe.py

test-smoothing:
	@echo "Using camera: /dev/video$(CAMERA)"
	$(DOCKER_RUN_DISPLAY) python test_smoothing.py

test-camera:
	@echo "Using camera: /dev/video$(CAMERA)"
	$(DOCKER_RUN_DISPLAY) python test_camera.py

test-mobilenet:
	@echo "Using camera: /dev/video$(CAMERA)"
	$(DOCKER_RUN_DISPLAY) python mobilenet_segmentation.py

compare-segmentation:
	@echo "Using camera: /dev/video$(CAMERA)"
	$(DOCKER_RUN_DISPLAY) python compare_segmentation.py

compare-silhouette:
	@echo "Using camera: /dev/video$(CAMERA)"
	$(DOCKER_RUN_DISPLAY) python compare_silhouette.py

benchmark:
	@echo "Using camera: /dev/video$(CAMERA)"
	$(DOCKER_RUN_DISPLAY) python benchmark.py

# Training commands
generate:
	$(DOCKER_RUN) python generate_shapes.py
	$(DOCKER_RUN) python generate_letters.py

generate-shapes:
	$(DOCKER_RUN) python generate_shapes.py

generate-letters:
	$(DOCKER_RUN) python generate_letters.py

# Generate training data from images (run from host, not Docker - needs librsvg/imagemagick)
# Usage: make generate-images SRC=assets/image_sources/elephant OUT=assets/training/custom
# Supports: SVG, PNG, JPG, TIFF, WebP
generate-images:
	@if [ -z "$(SRC)" ] || [ -z "$(OUT)" ]; then \
		echo "Usage: make generate-images SRC=assets/image_sources/elephant OUT=assets/training/custom"; \
		exit 1; \
	fi
	./augment_images.sh "$(SRC)" "$(OUT)" 500

# Generate training data from ALL image sources
# Usage: make augment-all [COUNT=500]
augment-all:
	./augment_all.sh $(COUNT)

# Fix ownership of files created by Docker (run with sudo if needed)
fix-permissions:
	sudo chown -R $(USER):$(USER) assets/ models/ data/ 2>/dev/null || true

train:
	$(DOCKER_RUN) python shape_classifier.py

# Clean, generate, and train in one step
retrain: clean-training generate train

# Clean training data (run inside Docker to handle permissions)
clean-training:
	$(DOCKER_RUN) rm -rf assets/training/shapes/*/
	$(DOCKER_RUN) rm -rf assets/training/letters/*/

# Clean models
clean-models:
	$(DOCKER_RUN) rm -rf models/*

# Clean everything
clean: clean-training clean-models

# Fix clock skew issues (timestamps from Docker)
fix-timestamps:
	$(DOCKER_RUN) find . -type f -exec touch {} +

# Interactive shell
shell:
	@echo "Using camera: /dev/video$(CAMERA)"
	$(DOCKER_RUN_DISPLAY) bash

# Help
help:
	@echo "Shadow Puppets - Available commands:"
	@echo ""
	@echo "  make build          - Build the Docker image"
	@echo "  make run / game     - Run the main game"
	@echo "  make live           - Run live silhouette display"
	@echo ""
	@echo "CAMERA SELECTION:"
	@echo "  make game           - Use default camera (/dev/video0)"
	@echo "  make game CAMERA=2  - Use /dev/video2 (USB camera)"
	@echo "  make live CAMERA=1  - Use /dev/video1"
	@echo ""
	@echo "  make test-display   - Test Pygame display"
	@echo "  make test-mediapipe - Test MediaPipe segmentation"
	@echo "  make test-smoothing - Test silhouette smoothing"
	@echo "  make test-camera    - Test camera settings"
	@echo "  make benchmark      - Run performance benchmark"
	@echo ""
	@echo "  make generate       - Generate training data (shapes + letters)"
	@echo "  make train          - Train the classifier"
	@echo "  make retrain        - Clean, generate, and train"
	@echo ""
	@echo "  make clean-training - Remove training images"
	@echo "  make clean-models   - Remove trained models"
	@echo "  make clean          - Remove all generated files"
	@echo ""
	@echo "  make shell          - Open interactive bash shell"
	@echo "  make help           - Show this help"
	@echo ""
	@echo "Audio setup:"
	@echo "  - Put background music in bgm/*.ogg"
	@echo "  - Put sound effects in sfx/boom.ogg and sfx/bell.ogg"
