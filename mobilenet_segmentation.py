#!/usr/bin/env python3
"""
MobileNet-based segmentation using DeepLabV3 with MobileNetV3 backbone.
Alternative to MediaPipe selfie segmentation for potentially better results.
"""

import numpy as np
import cv2
from typing import Optional, Tuple

# Try to import torch - will be installed in updated Dockerfile
try:
    import torch
    import torchvision.transforms as T
    from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large, DeepLabV3_MobileNet_V3_Large_Weights
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available, MobileNet segmentation disabled")


class MobileNetSegmentation:
    """Person segmentation using DeepLabV3 with MobileNetV3 backbone."""
    
    # COCO class index for 'person'
    PERSON_CLASS = 15
    
    def __init__(self, device: str = None, threshold: float = 0.5):
        """
        Initialize the segmentation model.
        
        Args:
            device: 'cuda', 'cpu', or None for auto-detect
            threshold: Confidence threshold for person mask
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for MobileNet segmentation")
        
        # Auto-detect device
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.threshold = threshold
        
        print(f"  Loading DeepLabV3-MobileNetV3 on {device}...")
        
        # Load pretrained model
        weights = DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT
        self.model = deeplabv3_mobilenet_v3_large(weights=weights)
        self.model.to(device)
        self.model.eval()
        
        # Preprocessing transform
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        print(f"  MobileNet segmentation ready")
    
    def process(self, frame_rgb: np.ndarray) -> 'SegmentationResult':
        """
        Process a frame and return segmentation mask.
        
        Args:
            frame_rgb: RGB image as numpy array (H, W, 3)
            
        Returns:
            SegmentationResult with segmentation_mask attribute
        """
        original_size = (frame_rgb.shape[1], frame_rgb.shape[0])  # (W, H)
        
        # Preprocess
        input_tensor = self.transform(frame_rgb).unsqueeze(0).to(self.device)
        
        # Run inference
        with torch.no_grad():
            output = self.model(input_tensor)['out'][0]
        
        # Get person class probabilities
        probs = torch.softmax(output, dim=0)
        person_mask = probs[self.PERSON_CLASS].cpu().numpy()
        
        # Resize mask back to original size
        mask = cv2.resize(person_mask, original_size, interpolation=cv2.INTER_LINEAR)
        
        return SegmentationResult(mask)
    
    def close(self):
        """Release resources."""
        # Clear model from GPU memory if applicable
        if hasattr(self, 'model'):
            del self.model
        if self.device == 'cuda':
            torch.cuda.empty_cache()


class SegmentationResult:
    """Container for segmentation results, matching MediaPipe's interface."""
    
    def __init__(self, segmentation_mask: np.ndarray):
        self.segmentation_mask = segmentation_mask


def test_mobilenet_segmentation():
    """Test the MobileNet segmentation with a sample image."""
    if not TORCH_AVAILABLE:
        print("PyTorch not available, skipping test")
        return False
    
    print("Testing MobileNet segmentation...")
    
    # Create test image (random noise simulating a frame)
    test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Initialize segmentation
    seg = MobileNetSegmentation()
    
    # Process frame
    import time
    start = time.time()
    result = seg.process(test_frame)
    elapsed = time.time() - start
    
    print(f"  Mask shape: {result.segmentation_mask.shape}")
    print(f"  Mask range: [{result.segmentation_mask.min():.3f}, {result.segmentation_mask.max():.3f}]")
    print(f"  Inference time: {elapsed*1000:.1f}ms")
    
    # Benchmark multiple frames
    print("\n  Benchmarking 10 frames...")
    times = []
    for _ in range(10):
        start = time.time()
        seg.process(test_frame)
        times.append(time.time() - start)
    
    avg_time = sum(times) / len(times)
    print(f"  Average: {avg_time*1000:.1f}ms ({1/avg_time:.1f} FPS)")
    
    seg.close()
    print("\n✓ MobileNet segmentation test passed!")
    return True


if __name__ == "__main__":
    test_mobilenet_segmentation()
