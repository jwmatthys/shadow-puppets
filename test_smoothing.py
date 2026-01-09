#!/usr/bin/env python3
"""
Test silhouette smoothing with live comparison of presets.
Press 1-4 to switch presets, ESC to exit.
"""

import sys
import time
import cv2
import numpy as np
import pygame
import mediapipe as mp
from silhouette import SilhouetteProcessor, PRESETS, create_processor


def main():
    print("=" * 50)
    print("SILHOUETTE SMOOTHING TEST")
    print("=" * 50)
    print("\nControls:")
    print("  1 = Default preset")
    print("  2 = Responsive preset (less smoothing)")
    print("  3 = Smooth preset (more smoothing)")
    print("  4 = Raw preset (no smoothing)")
    print("  ESC = Exit")
    print()
    
    # Initialize MediaPipe
    mp_selfie = mp.solutions.selfie_segmentation
    segmentation = mp_selfie.SelfieSegmentation(model_selection=1)
    
    # Initialize processor with default preset
    processor = create_processor("default")
    current_preset = "default"
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam")
        return 1
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    
    # Initialize Pygame
    pygame.init()
    screen = pygame.display.set_mode((320, 240))
    pygame.display.set_caption("Silhouette Smoothing Test")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 24)
    
    print("Running... (press ESC to exit)")
    
    frame_times = []
    running = True
    
    while running:
        frame_start = time.time()
        
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_1:
                    processor = create_processor("default")
                    current_preset = "default"
                    print("Switched to: default")
                elif event.key == pygame.K_2:
                    processor = create_processor("responsive")
                    current_preset = "responsive"
                    print("Switched to: responsive")
                elif event.key == pygame.K_3:
                    processor = create_processor("smooth")
                    current_preset = "smooth"
                    print("Switched to: smooth")
                elif event.key == pygame.K_4:
                    processor = create_processor("raw")
                    current_preset = "raw"
                    print("Switched to: raw")
        
        # Capture frame
        ret, frame = cap.read()
        if not ret:
            continue
        
        # Get segmentation mask
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = segmentation.process(frame_rgb)
        
        # Process silhouette
        silhouette = processor.create_silhouette_image(
            results.segmentation_mask, 320, 240
        )
        
        # Convert to Pygame surface
        silhouette = np.transpose(silhouette, (1, 0, 2))
        silhouette = np.flip(silhouette, axis=0)
        surface = pygame.surfarray.make_surface(silhouette)
        
        # Draw
        screen.blit(surface, (0, 0))
        
        # Draw preset name and FPS
        frame_time = time.time() - frame_start
        frame_times.append(frame_time)
        if len(frame_times) > 30:
            frame_times.pop(0)
        avg_fps = 1.0 / (sum(frame_times) / len(frame_times))
        
        text = font.render(f"{current_preset} | {avg_fps:.0f} FPS", True, (128, 128, 128))
        screen.blit(text, (5, 5))
        
        pygame.display.flip()
        clock.tick(30)
    
    # Cleanup
    cap.release()
    segmentation.close()
    pygame.quit()
    
    print("\nDone!")
    return 0


if __name__ == "__main__":
    sys.exit(main())