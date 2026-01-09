#!/usr/bin/env python3
"""
Silhouette processing with smoothing and refinement.
Converts MediaPipe segmentation masks into clean silhouettes.
"""

import cv2
import numpy as np


class SilhouetteProcessor:
    """Processes segmentation masks into clean silhouettes."""
    
    def __init__(
        self,
        threshold: float = 0.5,
        blur_size: int = 5,
        morph_size: int = 5,
        temporal_smoothing: float = 0.3,
        edge_smoothing: bool = True,
    ):
        """
        Initialize the silhouette processor.
        
        Args:
            threshold: Confidence threshold for segmentation (0.0-1.0)
            blur_size: Gaussian blur kernel size (odd number, 0 to disable)
            morph_size: Morphological operation kernel size (odd number)
            temporal_smoothing: Blend factor with previous frame (0.0-1.0, 0 to disable)
            edge_smoothing: Apply edge smoothing via contour approximation
        """
        self.threshold = threshold
        self.blur_size = blur_size if blur_size % 2 == 1 else blur_size + 1
        self.morph_size = morph_size if morph_size % 2 == 1 else morph_size + 1
        self.temporal_smoothing = temporal_smoothing
        self.edge_smoothing = edge_smoothing
        
        # Morphological kernels
        self.morph_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (self.morph_size, self.morph_size)
        )
        
        # Previous frame for temporal smoothing
        self.prev_mask = None
    
    def process(self, mask: np.ndarray) -> np.ndarray:
        """
        Process a segmentation mask into a clean binary silhouette.
        
        Args:
            mask: Float segmentation mask from MediaPipe (0.0-1.0)
            
        Returns:
            Binary mask (0 or 255) with smoothed silhouette
        """
        if mask is None:
            # Return white (no silhouette) if no mask
            if self.prev_mask is not None:
                return np.zeros_like(self.prev_mask)
            return None
        
        # Step 1: Apply Gaussian blur to raw mask (reduces noise)
        if self.blur_size > 1:
            mask = cv2.GaussianBlur(mask, (self.blur_size, self.blur_size), 0)
        
        # Step 2: Temporal smoothing (blend with previous frame)
        if self.temporal_smoothing > 0 and self.prev_mask is not None:
            mask = cv2.addWeighted(
                mask, 1 - self.temporal_smoothing,
                self.prev_mask, self.temporal_smoothing,
                0
            )
        
        # Store for next frame
        self.prev_mask = mask.copy()
        
        # Step 3: Threshold to binary
        binary = (mask > self.threshold).astype(np.uint8) * 255
        
        # Step 4: Morphological operations to clean up
        # Close (fill small holes)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, self.morph_kernel)
        # Open (remove small noise)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, self.morph_kernel)
        
        # Step 5: Edge smoothing via contour approximation
        if self.edge_smoothing:
            binary = self._smooth_edges(binary)
        
        return binary
    
    def _smooth_edges(self, binary: np.ndarray) -> np.ndarray:
        """Smooth edges by finding and redrawing contours."""
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            return binary
        
        # Create fresh mask
        smoothed = np.zeros_like(binary)
        
        for contour in contours:
            # Skip tiny contours (noise)
            if cv2.contourArea(contour) < 500:
                continue
            
            # Approximate contour to reduce jaggedness
            epsilon = 0.002 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Draw filled contour
            cv2.drawContours(smoothed, [approx], -1, 255, -1)
        
        return smoothed
    
    def create_silhouette_image(
        self,
        mask: np.ndarray,
        width: int = 320,
        height: int = 240,
        invert: bool = False,
    ) -> np.ndarray:
        """
        Create a silhouette image (black figure on white, or vice versa).
        
        Args:
            mask: Raw segmentation mask from MediaPipe
            width: Output width
            height: Output height
            invert: If True, white figure on black background
            
        Returns:
            RGB image array suitable for Pygame
        """
        # Process the mask
        binary = self.process(mask)
        
        if binary is None:
            # No detection - return blank
            bg_color = 0 if invert else 255
            return np.full((height, width, 3), bg_color, dtype=np.uint8)
        
        # Resize if needed
        if binary.shape != (height, width):
            binary = cv2.resize(binary, (width, height), interpolation=cv2.INTER_AREA)
        
        # Create RGB image
        if invert:
            # White figure on black
            silhouette = np.zeros((height, width, 3), dtype=np.uint8)
            silhouette[binary == 255] = [255, 255, 255]
        else:
            # Black figure on white (default for shadow puppets)
            silhouette = np.full((height, width, 3), 255, dtype=np.uint8)
            silhouette[binary == 255] = [0, 0, 0]
        
        return silhouette
    
    def reset(self):
        """Reset temporal state (call when starting new game/round)."""
        self.prev_mask = None


# Preset configurations for different use cases
PRESETS = {
    "default": {
        "threshold": 0.5,
        "blur_size": 5,
        "morph_size": 5,
        "temporal_smoothing": 0.3,
        "edge_smoothing": True,
    },
    "responsive": {
        "threshold": 0.5,
        "blur_size": 3,
        "morph_size": 3,
        "temporal_smoothing": 0.1,
        "edge_smoothing": True,
    },
    "smooth": {
        "threshold": 0.5,
        "blur_size": 7,
        "morph_size": 7,
        "temporal_smoothing": 0.5,
        "edge_smoothing": True,
    },
    "raw": {
        "threshold": 0.5,
        "blur_size": 0,
        "morph_size": 3,
        "temporal_smoothing": 0.0,
        "edge_smoothing": False,
    },
}


def create_processor(preset: str = "default") -> SilhouetteProcessor:
    """Create a SilhouetteProcessor with a preset configuration."""
    if preset not in PRESETS:
        raise ValueError(f"Unknown preset: {preset}. Choose from: {list(PRESETS.keys())}")
    return SilhouetteProcessor(**PRESETS[preset])