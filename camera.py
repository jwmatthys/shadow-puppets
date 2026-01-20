#!/usr/bin/env python3
"""
Camera capture module optimized for Shadow Puppets.
"""

import cv2
import numpy as np
from typing import Optional, Tuple, Union


class Camera:
    """Webcam capture with settings optimized for silhouette detection."""
    
    def __init__(
        self,
        device: Union[int, str] = 0,
        width: int = 320,
        height: int = 240,
        flip_horizontal: bool = True,
    ):
        """
        Initialize camera.
        
        Args:
            device: Camera device index (e.g., 0, 2) or device path (e.g., '/dev/video2')
            width: Desired capture width
            height: Desired capture height
            flip_horizontal: Mirror the image (natural for facing camera)
        """
        self.device = device
        self.target_width = width
        self.target_height = height
        self.flip_horizontal = flip_horizontal
        
        self.cap: Optional[cv2.VideoCapture] = None
        self.actual_width = width
        self.actual_height = height
        self.actual_fps = 15.0  # Default assumption
    
    def open(self) -> bool:
        """Open the camera. Returns True on success."""
        self.cap = cv2.VideoCapture(self.device)
        
        if not self.cap.isOpened():
            return False
        
        # Apply settings
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize latency
        
        # Read actual values
        self.actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        # Warm up camera (first few frames are often bad)
        for _ in range(5):
            self.cap.read()
        
        return True
    
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read a frame from the camera.
        
        Returns:
            (success, frame) where frame is RGB numpy array or None
        """
        if self.cap is None:
            return False, None
        
        ret, frame = self.cap.read()
        
        if not ret or frame is None:
            return False, None
        
        # Resize if camera gave us different resolution
        if frame.shape[1] != self.target_width or frame.shape[0] != self.target_height:
            frame = cv2.resize(
                frame, 
                (self.target_width, self.target_height),
                interpolation=cv2.INTER_AREA
            )
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        return True, frame_rgb
    
    def close(self):
        """Release the camera."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
    
    def __enter__(self):
        """Context manager entry."""
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False
    
    @property
    def is_open(self) -> bool:
        """Check if camera is open."""
        return self.cap is not None and self.cap.isOpened()
    
    @property
    def resolution(self) -> Tuple[int, int]:
        """Get actual capture resolution."""
        return (self.actual_width, self.actual_height)
    
    @property
    def fps(self) -> float:
        """Get reported FPS (may not reflect actual capture rate)."""
        return self.actual_fps
