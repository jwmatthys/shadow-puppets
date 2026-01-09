#!/usr/bin/env python3
"""
Camera diagnostics and optimization tests.
"""

import sys
import time
import cv2


def test_camera_settings():
    print("=" * 50)
    print("CAMERA DIAGNOSTICS")
    print("=" * 50)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam")
        return 1
    
    # Query current settings
    print("\n[1] Current camera settings:")
    settings = [
        ("CAP_PROP_FRAME_WIDTH", cv2.CAP_PROP_FRAME_WIDTH),
        ("CAP_PROP_FRAME_HEIGHT", cv2.CAP_PROP_FRAME_HEIGHT),
        ("CAP_PROP_FPS", cv2.CAP_PROP_FPS),
        ("CAP_PROP_FOURCC", cv2.CAP_PROP_FOURCC),
        ("CAP_PROP_BUFFERSIZE", cv2.CAP_PROP_BUFFERSIZE),
        ("CAP_PROP_BACKEND", cv2.CAP_PROP_BACKEND),
    ]
    
    for name, prop in settings:
        val = cap.get(prop)
        if prop == cv2.CAP_PROP_FOURCC and val > 0:
            fourcc = int(val)
            fourcc_str = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
            print(f"    {name}: {fourcc_str}")
        else:
            print(f"    {name}: {val}")
    
    cap.release()
    
    # Test different configurations
    configs = [
        {"name": "Default 320x240", "width": 320, "height": 240, "fps": None, "buffersize": None, "fourcc": None},
        {"name": "320x240 + 30fps request", "width": 320, "height": 240, "fps": 30, "buffersize": None, "fourcc": None},
        {"name": "320x240 + buffer=1", "width": 320, "height": 240, "fps": 30, "buffersize": 1, "fourcc": None},
        {"name": "320x240 + MJPG", "width": 320, "height": 240, "fps": 30, "buffersize": 1, "fourcc": "MJPG"},
        {"name": "640x480 + MJPG + buffer=1", "width": 640, "height": 480, "fps": 30, "buffersize": 1, "fourcc": "MJPG"},
        {"name": "160x120 + buffer=1", "width": 160, "height": 120, "fps": 30, "buffersize": 1, "fourcc": None},
    ]
    
    print("\n[2] Testing configurations:")
    print("-" * 60)
    
    results = []
    
    for config in configs:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            continue
        
        # Apply settings
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config["width"])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config["height"])
        
        if config["fps"]:
            cap.set(cv2.CAP_PROP_FPS, config["fps"])
        
        if config["buffersize"]:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, config["buffersize"])
        
        if config["fourcc"]:
            fourcc = cv2.VideoWriter_fourcc(*config["fourcc"])
            cap.set(cv2.CAP_PROP_FOURCC, fourcc)
        
        # Verify actual settings
        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Warm up
        for _ in range(5):
            cap.read()
        
        # Benchmark capture
        times = []
        for _ in range(30):
            t0 = time.time()
            ret, frame = cap.read()
            if ret:
                times.append(time.time() - t0)
        
        cap.release()
        
        if times:
            avg_ms = (sum(times) / len(times)) * 1000
            effective_fps = 1000 / avg_ms
            results.append((config["name"], avg_ms, effective_fps, actual_w, actual_h))
            print(f"  {config['name']:30s}: {avg_ms:5.1f}ms = {effective_fps:5.1f} FPS (actual: {actual_w}x{actual_h})")
        else:
            print(f"  {config['name']:30s}: FAILED")
        
        time.sleep(0.2)  # Let camera settle
    
    # Find best
    if results:
        print("-" * 60)
        best = max(results, key=lambda x: x[2])
        print(f"\n  BEST: {best[0]} at {best[2]:.1f} FPS")
    
    print("\n" + "=" * 50)
    return 0


if __name__ == "__main__":
    sys.exit(test_camera_settings())