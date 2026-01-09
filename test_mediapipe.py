#!/usr/bin/env python3
"""
Test 2: MediaPipe Segmentation Test
Tests MediaPipe selfie segmentation with detailed diagnostics.
Uses a generated test image first, then webcam if available.
"""

import sys
import time
import numpy as np

def test_imports():
    """Test that all required imports work."""
    print("\n[1] Testing imports...")
    
    errors = []
    
    try:
        import cv2
        print(f"    ✓ OpenCV {cv2.__version__}")
    except ImportError as e:
        errors.append(f"OpenCV: {e}")
    
    try:
        import mediapipe as mp
        print(f"    ✓ MediaPipe {mp.__version__}")
    except ImportError as e:
        errors.append(f"MediaPipe: {e}")
    
    try:
        import pygame
        print(f"    ✓ Pygame {pygame.version.ver}")
    except ImportError as e:
        errors.append(f"Pygame: {e}")
    
    try:
        import numpy as np
        print(f"    ✓ NumPy {np.__version__}")
    except ImportError as e:
        errors.append(f"NumPy: {e}")
    
    if errors:
        print("\n    ✗ Import errors:")
        for err in errors:
            print(f"      - {err}")
        return False
    
    return True


def test_mediapipe_init():
    """Test MediaPipe segmentation initialization."""
    print("\n[2] Initializing MediaPipe Selfie Segmentation...")
    
    try:
        import mediapipe as mp
        
        mp_selfie = mp.solutions.selfie_segmentation
        print(f"    ✓ Loaded selfie_segmentation module")
        
        # Use model_selection=1 for landscape model (faster, good for groups)
        # model_selection=0 is general model (more accurate for single person)
        segmentation = mp_selfie.SelfieSegmentation(model_selection=1)
        print(f"    ✓ Created SelfieSegmentation instance (landscape model)")
        
        return segmentation
        
    except Exception as e:
        print(f"    ✗ Failed to initialize MediaPipe: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_with_synthetic_image(segmentation):
    """Test segmentation with a synthetic image containing person-like shapes."""
    print("\n[3] Testing with synthetic image...")
    
    import cv2
    import numpy as np
    
    # Create a 320x240 test image with a simple "person" silhouette
    width, height = 320, 240
    image = np.ones((height, width, 3), dtype=np.uint8) * 200  # Gray background
    
    # Draw a simple stick figure / person shape
    # Head
    cv2.circle(image, (160, 50), 25, (100, 80, 60), -1)
    # Body
    cv2.rectangle(image, (130, 75), (190, 160), (100, 80, 60), -1)
    # Arms
    cv2.rectangle(image, (80, 80), (130, 100), (100, 80, 60), -1)
    cv2.rectangle(image, (190, 80), (240, 100), (100, 80, 60), -1)
    # Legs
    cv2.rectangle(image, (130, 160), (155, 230), (100, 80, 60), -1)
    cv2.rectangle(image, (165, 160), (190, 230), (100, 80, 60), -1)
    
    print(f"    ✓ Created synthetic test image ({width}x{height})")
    
    # Convert BGR to RGB for MediaPipe
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process the image
    print("    Processing image with MediaPipe...")
    start_time = time.time()
    
    try:
        results = segmentation.process(image_rgb)
        process_time = (time.time() - start_time) * 1000
        
        if results.segmentation_mask is None:
            print(f"    ⚠ Segmentation returned None (no person detected)")
            print(f"      This is expected for synthetic images")
            return True, image, None
        
        mask = results.segmentation_mask
        print(f"    ✓ Got segmentation mask: {mask.shape}, dtype={mask.dtype}")
        print(f"    ✓ Mask value range: {mask.min():.3f} to {mask.max():.3f}")
        print(f"    ✓ Processing time: {process_time:.1f}ms")
        
        # Count pixels detected as person (threshold 0.5)
        person_pixels = np.sum(mask > 0.5)
        total_pixels = mask.shape[0] * mask.shape[1]
        percentage = (person_pixels / total_pixels) * 100
        print(f"    ✓ Person pixels: {person_pixels} ({percentage:.1f}%)")
        
        return True, image, mask
        
    except Exception as e:
        print(f"    ✗ Segmentation failed: {e}")
        import traceback
        traceback.print_exc()
        return False, image, None


def test_with_webcam(segmentation):
    """Test segmentation with webcam input."""
    print("\n[4] Testing with webcam...")
    
    import cv2
    
    # Try to open webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("    ⚠ No webcam available (this is OK for initial testing)")
        return True, None
    
    # Set resolution to 320x240 for performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"    ✓ Webcam opened at {actual_width}x{actual_height}")
    
    # Capture a few frames to let camera warm up
    print("    Warming up camera...")
    for _ in range(10):
        cap.read()
    
    # Capture test frame
    ret, frame = cap.read()
    if not ret:
        print("    ✗ Failed to capture frame")
        cap.release()
        return False, None
    
    print(f"    ✓ Captured frame: {frame.shape}")
    
    # Process with MediaPipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    print("    Processing with MediaPipe...")
    times = []
    masks = []
    
    for i in range(5):
        ret, frame = cap.read()
        if not ret:
            continue
            
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        start = time.time()
        results = segmentation.process(frame_rgb)
        elapsed = (time.time() - start) * 1000
        times.append(elapsed)
        
        if results.segmentation_mask is not None:
            masks.append(results.segmentation_mask)
    
    cap.release()
    
    if times:
        avg_time = sum(times) / len(times)
        print(f"    ✓ Average processing time: {avg_time:.1f}ms ({1000/avg_time:.1f} FPS potential)")
    
    if masks:
        last_mask = masks[-1]
        person_pixels = np.sum(last_mask > 0.5)
        total_pixels = last_mask.shape[0] * last_mask.shape[1]
        percentage = (person_pixels / total_pixels) * 100
        print(f"    ✓ Person detected: {percentage:.1f}% of frame")
        return True, last_mask
    else:
        print("    ⚠ No person detected in webcam frames")
        return True, None


def test_display_silhouette(segmentation):
    """Display silhouette output in Pygame window."""
    print("\n[5] Testing Pygame display with segmentation...")
    
    import cv2
    import pygame
    import numpy as np
    
    pygame.init()
    screen = pygame.display.set_mode((320, 240))
    pygame.display.set_caption("Shadow Puppets - Segmentation Test")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 24)
    
    # Try webcam first
    cap = cv2.VideoCapture(0)
    use_webcam = cap.isOpened()
    
    if use_webcam:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        print("    ✓ Using webcam for display test")
    else:
        print("    ⚠ No webcam - using animated synthetic image")
    
    print("    Running for 10 seconds (press ESC to exit early)...")
    
    start_time = time.time()
    frame_count = 0
    
    running = True
    while running and (time.time() - start_time) < 10.0:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
        
        if use_webcam:
            ret, frame = cap.read()
            if not ret:
                continue
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            # Create animated synthetic person
            t = time.time() - start_time
            frame_rgb = create_animated_person(320, 240, t)
        
        # Get segmentation
        process_start = time.time()
        results = segmentation.process(frame_rgb)
        process_time = (time.time() - process_start) * 1000
        
        # Create silhouette image (black person on white background)
        if results.segmentation_mask is not None:
            mask = results.segmentation_mask
            # Threshold mask
            binary_mask = (mask > 0.5).astype(np.uint8)
            # Create silhouette: white background, black where person is
            silhouette = np.ones((240, 320, 3), dtype=np.uint8) * 255
            silhouette[binary_mask == 1] = [0, 0, 0]
        else:
            # No detection - show white screen
            silhouette = np.ones((240, 320, 3), dtype=np.uint8) * 255
        
        # Convert to Pygame surface
        # Need to rotate/flip for correct orientation
        silhouette = np.transpose(silhouette, (1, 0, 2))
        silhouette = np.flip(silhouette, axis=0)
        surface = pygame.surfarray.make_surface(silhouette)
        
        # Draw to screen
        screen.blit(surface, (0, 0))
        
        # Draw FPS
        fps_text = font.render(f"Process: {process_time:.0f}ms", True, (128, 128, 128))
        screen.blit(fps_text, (5, 5))
        
        pygame.display.flip()
        clock.tick(30)
        frame_count += 1
    
    if use_webcam:
        cap.release()
    pygame.quit()
    
    elapsed = time.time() - start_time
    print(f"    ✓ Displayed {frame_count} frames in {elapsed:.1f}s ({frame_count/elapsed:.1f} FPS)")
    
    return True


def create_animated_person(width, height, t):
    """Create a simple animated person shape for testing without webcam."""
    import cv2
    import numpy as np
    
    image = np.ones((height, width, 3), dtype=np.uint8) * 200
    
    # Animate position
    x_offset = int(np.sin(t) * 30)
    arm_angle = int(np.sin(t * 2) * 20)
    
    center_x = 160 + x_offset
    
    # Head
    cv2.circle(image, (center_x, 50), 25, (100, 80, 60), -1)
    # Body
    cv2.rectangle(image, (center_x - 30, 75), (center_x + 30, 160), (100, 80, 60), -1)
    # Left arm
    cv2.rectangle(image, (center_x - 80 + arm_angle, 80), (center_x - 30, 100), (100, 80, 60), -1)
    # Right arm
    cv2.rectangle(image, (center_x + 30, 80), (center_x + 80 - arm_angle, 100), (100, 80, 60), -1)
    # Legs
    cv2.rectangle(image, (center_x - 25, 160), (center_x - 5, 230), (100, 80, 60), -1)
    cv2.rectangle(image, (center_x + 5, 160), (center_x + 25, 230), (100, 80, 60), -1)
    
    return image


def main():
    print("=" * 50)
    print("SHADOW PUPPETS - MediaPipe Segmentation Test")
    print("=" * 50)
    
    # Test imports
    if not test_imports():
        print("\n✗ Import test failed!")
        return 1
    
    # Initialize MediaPipe
    segmentation = test_mediapipe_init()
    if segmentation is None:
        print("\n✗ MediaPipe initialization failed!")
        return 1
    
    # Test with synthetic image
    success, _, _ = test_with_synthetic_image(segmentation)
    if not success:
        print("\n✗ Synthetic image test failed!")
        return 1
    
    # Test with webcam
    success, _ = test_with_webcam(segmentation)
    if not success:
        print("\n✗ Webcam test failed!")
        return 1
    
    # Test display
    try:
        success = test_display_silhouette(segmentation)
        if not success:
            print("\n✗ Display test failed!")
            return 1
    except Exception as e:
        print(f"\n⚠ Display test skipped (no display available): {e}")
    
    # Cleanup
    segmentation.close()
    
    print("\n" + "=" * 50)
    print("ALL MEDIAPIPE TESTS PASSED!")
    print("=" * 50)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
