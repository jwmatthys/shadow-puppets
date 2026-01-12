#!/usr/bin/env python3
"""
Compare original vs improved silhouette processing side by side.
Shows the effect of guided filtering and hole preservation.
"""

import sys
import time
import numpy as np
import pygame
import cv2
import mediapipe as mp

from camera import Camera
from silhouette import create_processor as create_original_processor
from improved_silhouette import create_improved_processor


# Settings
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 480
CAPTURE_WIDTH = 320
CAPTURE_HEIGHT = 240


def main():
    print("=" * 50)
    print("SILHOUETTE PROCESSING COMPARISON")
    print("Original (left) vs Improved (right)")
    print("=" * 50)
    print()
    print("Improvements in new version:")
    print("  - Guided filter preserves edges from RGB frame")
    print("  - No MORPH_CLOSE (preserves holes between limbs)")
    print("  - Temporal EMA smoothing on probability mask")
    print("  - Median filter instead of Gaussian for denoising")
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
    segmentation = mp_selfie.SelfieSegmentation(model_selection=1)
    
    # Create both processors
    original_processor = create_original_processor("default")
    improved_processor = create_improved_processor("default")
    
    # Initialize Pygame
    print("Initializing display...")
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Silhouette Comparison: Original (left) vs Improved (right)")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 36)
    small_font = pygame.font.Font(None, 24)
    
    # Timing stats
    orig_times = []
    impr_times = []
    
    # Current preset for improved processor
    presets = ["default", "sharp", "stable", "responsive", "raw"]
    current_preset = 0
    
    print("\nRunning... Press ESC to quit, 1-5 to change improved preset\n")
    
    running = True
    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key in [pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4, pygame.K_5]:
                    idx = event.key - pygame.K_1
                    if idx < len(presets):
                        current_preset = idx
                        improved_processor = create_improved_processor(presets[current_preset])
                        print(f"Switched to preset: {presets[current_preset]}")
        
        # Capture frame
        ret, frame_rgb = camera.read()
        if not ret:
            continue
        
        # Get segmentation mask
        results = segmentation.process(frame_rgb)
        mask = results.segmentation_mask
        
        # Original processing
        orig_start = time.time()
        orig_silhouette = original_processor.create_silhouette_image(
            mask, CAPTURE_WIDTH, CAPTURE_HEIGHT
        )
        orig_time = time.time() - orig_start
        orig_times.append(orig_time)
        if len(orig_times) > 30:
            orig_times.pop(0)
        
        # Improved processing (with guide frame)
        impr_start = time.time()
        impr_silhouette = improved_processor.create_silhouette_image(
            mask, CAPTURE_WIDTH, CAPTURE_HEIGHT, guide_frame=frame_rgb
        )
        impr_time = time.time() - impr_start
        impr_times.append(impr_time)
        if len(impr_times) > 30:
            impr_times.pop(0)
        
        # Draw
        screen.fill((40, 40, 40))
        
        # Helper to convert silhouette to pygame surface
        def to_surface(silhouette):
            display = np.transpose(silhouette, (1, 0, 2))
            display = np.flip(display, axis=0)
            return pygame.surfarray.make_surface(display)
        
        # Scale factor
        scale_h = WINDOW_HEIGHT - 80
        scale_w = int(scale_h * CAPTURE_WIDTH / CAPTURE_HEIGHT)
        
        # Original (left side)
        orig_surface = to_surface(orig_silhouette)
        orig_scaled = pygame.transform.scale(orig_surface, (scale_w, scale_h))
        orig_x = (WINDOW_WIDTH // 2 - scale_w) // 2
        screen.blit(orig_scaled, (orig_x, 60))
        
        # Original label
        orig_avg = sum(orig_times) / len(orig_times) if orig_times else 0
        orig_label = font.render(f"Original: {orig_avg*1000:.1f}ms", True, (255, 255, 255))
        screen.blit(orig_label, (orig_x, 10))
        orig_info = small_font.render("MORPH_CLOSE fills holes", True, (200, 200, 200))
        screen.blit(orig_info, (orig_x, 38))
        
        # Improved (right side)
        impr_surface = to_surface(impr_silhouette)
        impr_scaled = pygame.transform.scale(impr_surface, (scale_w, scale_h))
        impr_x = WINDOW_WIDTH // 2 + (WINDOW_WIDTH // 2 - scale_w) // 2
        screen.blit(impr_scaled, (impr_x, 60))
        
        # Improved label
        impr_avg = sum(impr_times) / len(impr_times) if impr_times else 0
        impr_label = font.render(f"Improved [{presets[current_preset]}]: {impr_avg*1000:.1f}ms", True, (255, 255, 255))
        screen.blit(impr_label, (impr_x, 10))
        impr_info = small_font.render("Guided filter + holes preserved", True, (200, 200, 200))
        screen.blit(impr_info, (impr_x, 38))
        
        # Divider line
        pygame.draw.line(screen, (100, 100, 100), (WINDOW_WIDTH // 2, 0), (WINDOW_WIDTH // 2, WINDOW_HEIGHT), 2)
        
        # Instructions
        instr = small_font.render("Press 1-5 to change improved preset | ESC to quit", True, (150, 150, 150))
        instr_rect = instr.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT - 15))
        screen.blit(instr, instr_rect)
        
        pygame.display.flip()
        clock.tick(30)
    
    # Cleanup
    camera.close()
    segmentation.close()
    pygame.quit()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
