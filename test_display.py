#!/usr/bin/env python3
"""
Test 1: Display Output Test
Verifies that Pygame can create a window and render graphics.
"""

import pygame
import sys
import time

def main():
    print("=" * 50)
    print("SHADOW PUPPETS - Display Test")
    print("=" * 50)
    
    # Initialize Pygame
    print("\n[1] Initializing Pygame...")
    try:
        pygame.init()
        print(f"    ✓ Pygame initialized successfully")
        print(f"    ✓ Pygame version: {pygame.version.ver}")
    except Exception as e:
        print(f"    ✗ Failed to initialize Pygame: {e}")
        sys.exit(1)
    
    # Get display info
    print("\n[2] Checking display...")
    try:
        display_info = pygame.display.Info()
        print(f"    ✓ Display available")
        print(f"    ✓ Current resolution: {display_info.current_w}x{display_info.current_h}")
    except Exception as e:
        print(f"    ✗ Display error: {e}")
        sys.exit(1)
    
    # Create window
    print("\n[3] Creating window (320x240)...")
    try:
        screen = pygame.display.set_mode((320, 240))
        pygame.display.set_caption("Shadow Puppets - Display Test")
        print(f"    ✓ Window created successfully")
    except Exception as e:
        print(f"    ✗ Failed to create window: {e}")
        sys.exit(1)
    
    # Test rendering
    print("\n[4] Testing render cycle...")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 36)
    
    colors = [
        (255, 255, 255, "White"),
        (0, 0, 0, "Black"),
        (255, 0, 0, "Red"),
        (0, 255, 0, "Green"),
        (0, 0, 255, "Blue"),
    ]
    
    start_time = time.time()
    frame_count = 0
    color_index = 0
    
    print("    Running for 5 seconds...")
    print("    (Window should cycle through colors)")
    
    running = True
    while running and (time.time() - start_time) < 5.0:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
        
        # Change color every second
        elapsed = time.time() - start_time
        new_color_index = int(elapsed) % len(colors)
        if new_color_index != color_index:
            color_index = new_color_index
            print(f"    → Color: {colors[color_index][3]}")
        
        # Fill with current color
        color = colors[color_index][:3]
        screen.fill(color)
        
        # Draw contrasting text
        text_color = (0, 0, 0) if sum(color) > 382 else (255, 255, 255)
        text = font.render(f"Test: {colors[color_index][3]}", True, text_color)
        text_rect = text.get_rect(center=(160, 120))
        screen.blit(text, text_rect)
        
        pygame.display.flip()
        clock.tick(30)
        frame_count += 1
    
    elapsed = time.time() - start_time
    fps = frame_count / elapsed if elapsed > 0 else 0
    
    print(f"\n    ✓ Rendered {frame_count} frames in {elapsed:.2f}s")
    print(f"    ✓ Average FPS: {fps:.1f}")
    
    pygame.quit()
    
    print("\n" + "=" * 50)
    print("DISPLAY TEST PASSED!")
    print("=" * 50)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
