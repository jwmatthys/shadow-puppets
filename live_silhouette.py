#!/usr/bin/env python3
"""
Shadow Puppets - Live silhouette display.
This is the core visualization that will be used in the game.
"""

import sys
import time
import numpy as np
import pygame
import mediapipe as mp

from camera import Camera
from silhouette import create_processor


class SilhouetteDisplay:
    """Real-time silhouette display using Pygame."""
    
    def __init__(
        self,
        width: int = 320,
        height: int = 240,
        fullscreen: bool = False,
        show_fps: bool = True,
    ):
        self.width = width
        self.height = height
        self.fullscreen = fullscreen
        self.show_fps = show_fps
        
        # Components
        self.camera: Camera = None
        self.segmentation = None
        self.processor = None
        
        # Pygame
        self.screen = None
        self.clock = None
        self.font = None
        
        # Stats
        self.frame_times = []
        self.fps = 0.0
    
    def setup(self) -> bool:
        """Initialize all components. Returns True on success."""
        print("Setting up Shadow Puppets...")
        
        # Camera
        print("  Opening camera...")
        self.camera = Camera(width=self.width, height=self.height)
        if not self.camera.open():
            print("  ERROR: Could not open camera")
            return False
        print(f"  Camera ready: {self.camera.resolution}, {self.camera.fps} FPS reported")
        
        # MediaPipe
        print("  Loading MediaPipe...")
        mp_selfie = mp.solutions.selfie_segmentation
        self.segmentation = mp_selfie.SelfieSegmentation(model_selection=1)
        print("  MediaPipe ready")
        
        # Silhouette processor
        self.processor = create_processor("default")
        print("  Silhouette processor ready")
        
        # Pygame
        print("  Initializing display...")
        pygame.init()
        
        flags = pygame.FULLSCREEN if self.fullscreen else 0
        if self.fullscreen:
            # Get native resolution for fullscreen
            info = pygame.display.Info()
            self.display_width = info.current_w
            self.display_height = info.current_h
        else:
            self.display_width = self.width
            self.display_height = self.height
        
        self.screen = pygame.display.set_mode(
            (self.display_width, self.display_height), 
            flags
        )
        pygame.display.set_caption("Shadow Puppets")
        
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        
        print("  Display ready")
        print("Setup complete!\n")
        
        return True
    
    def cleanup(self):
        """Release all resources."""
        if self.camera:
            self.camera.close()
        if self.segmentation:
            self.segmentation.close()
        pygame.quit()
    
    def process_frame(self, frame_rgb: np.ndarray) -> np.ndarray:
        """Process a frame through segmentation and smoothing."""
        # Get segmentation mask
        results = self.segmentation.process(frame_rgb)
        
        # Create silhouette image
        silhouette = self.processor.create_silhouette_image(
            results.segmentation_mask,
            self.width,
            self.height
        )
        
        return silhouette
    
    def render(self, silhouette: np.ndarray):
        """Render silhouette to screen."""
        # Convert to Pygame surface
        # Match test_smoothing.py: transpose then flip
        display_array = np.transpose(silhouette, (1, 0, 2))
        display_array = np.flip(display_array, axis=0)
        surface = pygame.surfarray.make_surface(display_array)
        
        # Scale if fullscreen
        if self.fullscreen:
            surface = pygame.transform.scale(
                surface, 
                (self.display_width, self.display_height)
            )
        
        # Draw to screen
        self.screen.blit(surface, (0, 0))
        
        # FPS counter
        if self.show_fps:
            fps_text = self.font.render(f"{self.fps:.0f} FPS", True, (128, 128, 128))
            self.screen.blit(fps_text, (10, 10))
        
        pygame.display.flip()
    
    def update_fps(self, frame_time: float):
        """Update FPS calculation."""
        self.frame_times.append(frame_time)
        if len(self.frame_times) > 30:
            self.frame_times.pop(0)
        
        if self.frame_times:
            avg_time = sum(self.frame_times) / len(self.frame_times)
            self.fps = 1.0 / avg_time if avg_time > 0 else 0
    
    def run(self):
        """Main loop."""
        print("Running... (ESC or Q to quit, F for fullscreen toggle)")
        
        running = True
        while running:
            frame_start = time.time()
            
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_ESCAPE, pygame.K_q):
                        running = False
                    elif event.key == pygame.K_f:
                        self.toggle_fullscreen()
            
            # Capture frame
            ret, frame_rgb = self.camera.read()
            if not ret:
                continue
            
            # Process
            silhouette = self.process_frame(frame_rgb)
            
            # Render
            self.render(silhouette)
            
            # Stats
            frame_time = time.time() - frame_start
            self.update_fps(frame_time)
            
            # Don't limit FPS - camera is the bottleneck
            # self.clock.tick(30)
        
        print(f"\nFinal average: {self.fps:.1f} FPS")
    
    def toggle_fullscreen(self):
        """Toggle between windowed and fullscreen."""
        self.fullscreen = not self.fullscreen
        
        if self.fullscreen:
            info = pygame.display.Info()
            self.display_width = info.current_w
            self.display_height = info.current_h
            self.screen = pygame.display.set_mode(
                (self.display_width, self.display_height),
                pygame.FULLSCREEN
            )
        else:
            self.display_width = self.width
            self.display_height = self.height
            self.screen = pygame.display.set_mode(
                (self.display_width, self.display_height)
            )


def main():
    print("=" * 50)
    print("SHADOW PUPPETS - Live Silhouette")
    print("=" * 50)
    print()
    
    display = SilhouetteDisplay(
        width=320,
        height=240,
        fullscreen=False,
        show_fps=True,
    )
    
    if not display.setup():
        print("Setup failed!")
        return 1
    
    try:
        display.run()
    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        display.cleanup()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())