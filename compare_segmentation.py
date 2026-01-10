#!/usr/bin/env python3
"""
Compare MediaPipe vs MobileNet segmentation side by side.
"""

import sys
import time
import numpy as np
import pygame
import cv2
import mediapipe as mp

from camera import Camera
from silhouette import create_processor

# Try to import MobileNet segmentation
try:
    from mobilenet_segmentation import MobileNetSegmentation
    MOBILENET_AVAILABLE = True
except Exception as e:
    print(f"MobileNet not available: {e}")
    MOBILENET_AVAILABLE = False


# Settings
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 480
CAPTURE_WIDTH = 320
CAPTURE_HEIGHT = 240


def main():
    print("=" * 50)
    print("SEGMENTATION COMPARISON")
    print("MediaPipe vs MobileNet (DeepLabV3)")
    print("=" * 50)
    print()
    
    # Initialize camera
    print("Opening camera...")
    camera = Camera(width=CAPTURE_WIDTH, height=CAPTURE_HEIGHT)
    if not camera.open():
        print("ERROR: Could not open camera")
        return 1
    
    # Initialize MediaPipe
    print("Loading MediaPipe...")
    mp_selfie = mp.solutions.selfie_segmentation
    mediapipe_seg = mp_selfie.SelfieSegmentation(model_selection=1)
    
    # Initialize MobileNet
    mobilenet_seg = None
    if MOBILENET_AVAILABLE:
        print("Loading MobileNet...")
        try:
            mobilenet_seg = MobileNetSegmentation()
        except Exception as e:
            print(f"  Failed to load MobileNet: {e}")
    else:
        print("MobileNet not available")
    
    # Silhouette processor
    processor = create_processor("default")
    
    # Initialize Pygame
    print("Initializing display...")
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Segmentation Comparison: MediaPipe (left) vs MobileNet (right)")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 36)
    
    # Timing stats
    mp_times = []
    mn_times = []
    
    print("\nRunning... Press ESC to quit\n")
    
    running = True
    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
        
        # Capture frame
        ret, frame_rgb = camera.read()
        if not ret:
            continue
        
        # MediaPipe segmentation
        mp_start = time.time()
        mp_result = mediapipe_seg.process(frame_rgb)
        mp_time = time.time() - mp_start
        mp_times.append(mp_time)
        if len(mp_times) > 30:
            mp_times.pop(0)
        
        mp_silhouette = processor.create_silhouette_image(
            mp_result.segmentation_mask,
            CAPTURE_WIDTH,
            CAPTURE_HEIGHT
        )
        
        # MobileNet segmentation
        mn_silhouette = None
        mn_time = 0
        if mobilenet_seg:
            mn_start = time.time()
            mn_result = mobilenet_seg.process(frame_rgb)
            mn_time = time.time() - mn_start
            mn_times.append(mn_time)
            if len(mn_times) > 30:
                mn_times.pop(0)
            
            mn_silhouette = processor.create_silhouette_image(
                mn_result.segmentation_mask,
                CAPTURE_WIDTH,
                CAPTURE_HEIGHT
            )
        
        # Draw
        screen.fill((40, 40, 40))
        
        # Helper to convert silhouette to pygame surface
        def to_surface(silhouette):
            display = np.transpose(silhouette, (1, 0, 2))
            display = np.flip(display, axis=0)
            return pygame.surfarray.make_surface(display)
        
        # Scale factor
        scale_h = WINDOW_HEIGHT - 60
        scale_w = int(scale_h * CAPTURE_WIDTH / CAPTURE_HEIGHT)
        
        # MediaPipe (left side)
        mp_surface = to_surface(mp_silhouette)
        mp_scaled = pygame.transform.scale(mp_surface, (scale_w, scale_h))
        mp_x = (WINDOW_WIDTH // 2 - scale_w) // 2
        screen.blit(mp_scaled, (mp_x, 50))
        
        # MediaPipe label and FPS
        mp_avg = sum(mp_times) / len(mp_times) if mp_times else 0
        mp_fps = 1 / mp_avg if mp_avg > 0 else 0
        mp_label = font.render(f"MediaPipe: {mp_fps:.1f} FPS ({mp_avg*1000:.0f}ms)", True, (255, 255, 255))
        screen.blit(mp_label, (mp_x, 10))
        
        # MobileNet (right side)
        if mn_silhouette is not None:
            mn_surface = to_surface(mn_silhouette)
            mn_scaled = pygame.transform.scale(mn_surface, (scale_w, scale_h))
            mn_x = WINDOW_WIDTH // 2 + (WINDOW_WIDTH // 2 - scale_w) // 2
            screen.blit(mn_scaled, (mn_x, 50))
            
            # MobileNet label and FPS
            mn_avg = sum(mn_times) / len(mn_times) if mn_times else 0
            mn_fps = 1 / mn_avg if mn_avg > 0 else 0
            mn_label = font.render(f"MobileNet: {mn_fps:.1f} FPS ({mn_avg*1000:.0f}ms)", True, (255, 255, 255))
            screen.blit(mn_label, (mn_x, 10))
        else:
            # Show "not available" message
            mn_x = WINDOW_WIDTH // 2 + (WINDOW_WIDTH // 2 - scale_w) // 2
            na_label = font.render("MobileNet: Not available", True, (255, 100, 100))
            screen.blit(na_label, (mn_x, 10))
        
        # Divider line
        pygame.draw.line(screen, (100, 100, 100), (WINDOW_WIDTH // 2, 0), (WINDOW_WIDTH // 2, WINDOW_HEIGHT), 2)
        
        pygame.display.flip()
        clock.tick(30)
    
    # Cleanup
    camera.close()
    mediapipe_seg.close()
    if mobilenet_seg:
        mobilenet_seg.close()
    pygame.quit()
    
    # Print summary
    print("\nSummary:")
    if mp_times:
        print(f"  MediaPipe: {1/np.mean(mp_times):.1f} FPS avg")
    if mn_times:
        print(f"  MobileNet: {1/np.mean(mn_times):.1f} FPS avg")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
