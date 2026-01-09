#!/usr/bin/env python3
"""
Benchmark to identify where time is spent in the pipeline.
"""

import sys
import time
import cv2
import numpy as np
import mediapipe as mp
from silhouette import create_processor


def benchmark(iterations=60):
    print("=" * 50)
    print("PERFORMANCE BENCHMARK")
    print("=" * 50)
    
    # Initialize
    mp_selfie = mp.solutions.selfie_segmentation
    segmentation = mp_selfie.SelfieSegmentation(model_selection=1)
    processor = create_processor("default")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam")
        return 1
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    
    # Warm up
    print("\nWarming up...")
    for _ in range(10):
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            segmentation.process(frame_rgb)
    
    # Benchmark
    print(f"Running {iterations} iterations...\n")
    
    times = {
        "capture": [],
        "convert": [],
        "mediapipe": [],
        "smoothing": [],
        "total": [],
    }
    
    for i in range(iterations):
        total_start = time.time()
        
        # Capture
        t0 = time.time()
        ret, frame = cap.read()
        if not ret:
            continue
        times["capture"].append(time.time() - t0)
        
        # Convert BGR to RGB
        t0 = time.time()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        times["convert"].append(time.time() - t0)
        
        # MediaPipe segmentation
        t0 = time.time()
        results = segmentation.process(frame_rgb)
        times["mediapipe"].append(time.time() - t0)
        
        # Silhouette processing
        t0 = time.time()
        silhouette = processor.create_silhouette_image(
            results.segmentation_mask, 320, 240
        )
        times["smoothing"].append(time.time() - t0)
        
        times["total"].append(time.time() - total_start)
    
    # Results
    print("Results (averaged over {} frames):".format(len(times["total"])))
    print("-" * 40)
    
    total_avg = 0
    for name, data in times.items():
        if name == "total":
            continue
        avg_ms = (sum(data) / len(data)) * 1000
        total_avg += avg_ms
        pct = (avg_ms / (sum(times["total"]) / len(times["total"]) * 1000)) * 100
        print(f"  {name:12s}: {avg_ms:6.1f}ms ({pct:4.1f}%)")
    
    print("-" * 40)
    total_ms = (sum(times["total"]) / len(times["total"])) * 1000
    fps = 1000 / total_ms
    print(f"  {'TOTAL':12s}: {total_ms:6.1f}ms")
    print(f"  {'FPS':12s}: {fps:6.1f}")
    
    cap.release()
    segmentation.close()
    
    print("\n" + "=" * 50)
    return 0


if __name__ == "__main__":
    sys.exit(benchmark())