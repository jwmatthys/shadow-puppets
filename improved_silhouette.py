#!/usr/bin/env python3
"""
Improved silhouette processing with edge-preserving guided filter.
Addresses the "blobby" silhouettes and hole-filling problems.
"""

import cv2
import numpy as np
from typing import Optional, Tuple


class ImprovedSilhouetteProcessor:
    """
    Improved silhouette processor that:
    - Uses guided filter for edge-preserving smoothing
    - Does NOT fill holes (preserves gaps between arms/head/body)
    - Applies temporal EMA smoothing for stability
    - Works well with static poses
    """
    
    def __init__(
        self,
        threshold: float = 0.5,
        guided_radius: int = 8,
        guided_eps: float = 0.01,
        temporal_alpha: float = 0.6,  # Higher = more responsive, lower = smoother
        min_contour_area: int = 300,
        denoise_strength: int = 3,
    ):
        """
        Initialize the improved processor.
        
        Args:
            threshold: Confidence threshold for mask (0.0-1.0)
            guided_radius: Radius for guided filter (larger = smoother edges)
            guided_eps: Epsilon for guided filter (smaller = sharper edges)
            temporal_alpha: EMA smoothing factor (0.6 = 60% new frame, 40% previous)
            min_contour_area: Minimum contour area to keep (removes noise)
            denoise_strength: Median filter size for denoising (0 to disable)
        """
        self.threshold = threshold
        self.guided_radius = guided_radius
        self.guided_eps = guided_eps
        self.temporal_alpha = temporal_alpha
        self.min_contour_area = min_contour_area
        self.denoise_strength = denoise_strength
        
        # Previous frame for temporal smoothing
        self.prev_mask: Optional[np.ndarray] = None
    
    def guided_filter(
        self, 
        guide: np.ndarray, 
        src: np.ndarray, 
        radius: int, 
        eps: float
    ) -> np.ndarray:
        """
        Apply guided filter for edge-preserving smoothing.
        
        The guided filter smooths the source image while preserving edges
        that exist in the guide image (typically the RGB frame).
        
        Args:
            guide: Guide image (RGB frame, used for edge detection)
            src: Source image (segmentation mask to smooth)
            radius: Filter radius
            eps: Regularization parameter
            
        Returns:
            Filtered image with edges preserved
        """
        # Convert guide to grayscale if needed
        if len(guide.shape) == 3:
            guide = cv2.cvtColor(guide, cv2.COLOR_RGB2GRAY)
        
        # Ensure float32
        guide = guide.astype(np.float32) / 255.0
        src = src.astype(np.float32)
        
        # Box filter computations
        mean_guide = cv2.boxFilter(guide, -1, (radius, radius))
        mean_src = cv2.boxFilter(src, -1, (radius, radius))
        mean_guide_src = cv2.boxFilter(guide * src, -1, (radius, radius))
        mean_guide_sq = cv2.boxFilter(guide * guide, -1, (radius, radius))
        
        # Covariance and variance
        cov_guide_src = mean_guide_src - mean_guide * mean_src
        var_guide = mean_guide_sq - mean_guide * mean_guide
        
        # Linear coefficients
        a = cov_guide_src / (var_guide + eps)
        b = mean_src - a * mean_guide
        
        # Mean of coefficients
        mean_a = cv2.boxFilter(a, -1, (radius, radius))
        mean_b = cv2.boxFilter(b, -1, (radius, radius))
        
        # Output
        output = mean_a * guide + mean_b
        
        return output
    
    def process(
        self, 
        mask: np.ndarray, 
        guide_frame: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Process a segmentation mask into a clean binary silhouette.
        
        Args:
            mask: Float segmentation mask from MediaPipe (0.0-1.0)
            guide_frame: Optional RGB frame for guided filtering
            
        Returns:
            Binary mask (0 or 255) with clean edges and preserved holes
        """
        if mask is None:
            if self.prev_mask is not None:
                return np.zeros_like(self.prev_mask)
            return None
        
        # Step 1: Temporal EMA smoothing (on raw probability mask)
        if self.prev_mask is not None and self.temporal_alpha < 1.0:
            # EMA: new_value = alpha * current + (1 - alpha) * previous
            mask = self.temporal_alpha * mask + (1 - self.temporal_alpha) * self.prev_mask
        
        # Store for next frame (before any other processing)
        self.prev_mask = mask.copy()
        
        # Step 2: Guided filter for edge-preserving smoothing
        if guide_frame is not None and self.guided_radius > 0:
            # Resize guide to match mask if needed
            if guide_frame.shape[:2] != mask.shape[:2]:
                guide_frame = cv2.resize(guide_frame, (mask.shape[1], mask.shape[0]))
            
            mask = self.guided_filter(guide_frame, mask, self.guided_radius, self.guided_eps)
        
        # Step 3: Threshold to binary
        binary = (mask > self.threshold).astype(np.uint8) * 255
        
        # Step 4: Light denoising with median filter (preserves edges better than Gaussian)
        if self.denoise_strength > 0:
            ksize = self.denoise_strength * 2 + 1  # Ensure odd
            binary = cv2.medianBlur(binary, ksize)
        
        # Step 5: Remove small noise contours (but DON'T fill holes!)
        binary = self._remove_small_contours(binary)
        
        return binary
    
    def _remove_small_contours(self, binary: np.ndarray) -> np.ndarray:
        """Remove small noise contours without filling holes."""
        # Find all contours (external only - we want to preserve internal holes)
        contours, hierarchy = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            return binary
        
        # Create fresh mask, keeping only large enough contours
        result = np.zeros_like(binary)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= self.min_contour_area:
                # Draw filled contour
                cv2.drawContours(result, [contour], -1, 255, -1)
        
        # Now we need to restore any internal holes from the original
        # by finding internal contours and subtracting them
        
        # Get internal holes from the original image
        all_contours, hierarchy = cv2.findContours(
            binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        
        if hierarchy is not None:
            # hierarchy: [Next, Previous, First_Child, Parent]
            for i, (contour, hier) in enumerate(zip(all_contours, hierarchy[0])):
                parent = hier[3]
                # If this contour has a parent, it's a hole
                if parent != -1:
                    area = cv2.contourArea(contour)
                    # Only keep significant holes (not tiny noise)
                    if area >= self.min_contour_area // 2:
                        cv2.drawContours(result, [contour], -1, 0, -1)
        
        return result
    
    def create_silhouette_image(
        self,
        mask: np.ndarray,
        width: int = 320,
        height: int = 240,
        invert: bool = False,
        guide_frame: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Create a silhouette image (black figure on white, or vice versa).
        
        Args:
            mask: Raw segmentation mask from MediaPipe
            width: Output width
            height: Output height
            invert: If True, white figure on black background
            guide_frame: Optional RGB frame for guided filtering
            
        Returns:
            RGB image array suitable for Pygame
        """
        # Process the mask
        binary = self.process(mask, guide_frame)
        
        if binary is None:
            bg_color = 0 if invert else 255
            return np.full((height, width, 3), bg_color, dtype=np.uint8)
        
        # Resize if needed
        if binary.shape != (height, width):
            binary = cv2.resize(binary, (width, height), interpolation=cv2.INTER_AREA)
        
        # Create RGB image
        if invert:
            silhouette = np.zeros((height, width, 3), dtype=np.uint8)
            silhouette[binary == 255] = [255, 255, 255]
        else:
            silhouette = np.full((height, width, 3), 255, dtype=np.uint8)
            silhouette[binary == 255] = [0, 0, 0]
        
        return silhouette
    
    def reset(self):
        """Reset temporal state."""
        self.prev_mask = None


# Preset configurations
IMPROVED_PRESETS = {
    "default": {
        "threshold": 0.5,
        "guided_radius": 8,
        "guided_eps": 0.01,
        "temporal_alpha": 0.6,
        "min_contour_area": 300,
        "denoise_strength": 3,
    },
    "sharp": {
        "threshold": 0.5,
        "guided_radius": 4,
        "guided_eps": 0.001,  # Sharper edges
        "temporal_alpha": 0.7,
        "min_contour_area": 200,
        "denoise_strength": 1,
    },
    "stable": {
        "threshold": 0.5,
        "guided_radius": 12,
        "guided_eps": 0.02,
        "temporal_alpha": 0.4,  # More smoothing for stable poses
        "min_contour_area": 400,
        "denoise_strength": 5,
    },
    "responsive": {
        "threshold": 0.5,
        "guided_radius": 6,
        "guided_eps": 0.01,
        "temporal_alpha": 0.8,  # Quick response
        "min_contour_area": 200,
        "denoise_strength": 1,
    },
    "raw": {
        "threshold": 0.5,
        "guided_radius": 0,  # No guided filter
        "guided_eps": 0.01,
        "temporal_alpha": 1.0,  # No temporal smoothing
        "min_contour_area": 100,
        "denoise_strength": 0,
    },
}


def create_improved_processor(preset: str = "default") -> ImprovedSilhouetteProcessor:
    """Create an ImprovedSilhouetteProcessor with a preset configuration."""
    if preset not in IMPROVED_PRESETS:
        raise ValueError(f"Unknown preset: {preset}. Choose from: {list(IMPROVED_PRESETS.keys())}")
    return ImprovedSilhouetteProcessor(**IMPROVED_PRESETS[preset])


if __name__ == "__main__":
    print("Improved Silhouette Processor")
    print("Available presets:", list(IMPROVED_PRESETS.keys()))
