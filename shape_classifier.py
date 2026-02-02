#!/usr/bin/env python3
"""
Lightweight shape classifier using scikit-learn.
Uses HOG features + SVM for fast CPU inference.
"""

import os
import json
import numpy as np
import cv2
import joblib
from typing import List, Tuple, Dict, Optional


# Feature extraction settings
INPUT_SIZE = 64
HOG_CELL_SIZE = 8
HOG_BLOCK_SIZE = 2
HOG_BINS = 9


def extract_hog_features(image: np.ndarray) -> np.ndarray:
    """
    Extract HOG (Histogram of Oriented Gradients) features from an image.
    HOG is excellent for shape recognition and very fast.
    """
    # Ensure correct size
    if image.shape[0] != INPUT_SIZE or image.shape[1] != INPUT_SIZE:
        image = cv2.resize(image, (INPUT_SIZE, INPUT_SIZE))
    
    # Ensure grayscale
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Add 2-pixel white border to match training data expectations
    # Real silhouettes from camera often touch the bottom edge, but training
    # data typically has padding. This border helps the classifier by making
    # the input match training distribution.
    border_width = 2
    image = cv2.copyMakeBorder(
        image, 
        border_width, border_width, border_width, border_width,
        cv2.BORDER_CONSTANT, 
        value=255  # White border (background color)
    )
    
    # Resize back to INPUT_SIZE (border added 4 pixels in each dimension)
    image = cv2.resize(image, (INPUT_SIZE, INPUT_SIZE))
    
    # Compute HOG features
    win_size = (INPUT_SIZE, INPUT_SIZE)
    cell_size = (HOG_CELL_SIZE, HOG_CELL_SIZE)
    block_size = (HOG_CELL_SIZE * HOG_BLOCK_SIZE, HOG_CELL_SIZE * HOG_BLOCK_SIZE)
    block_stride = (HOG_CELL_SIZE, HOG_CELL_SIZE)
    
    hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, HOG_BINS)
    features = hog.compute(image)
    
    return features.flatten()


def load_training_data(data_dirs: list) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load training data and extract features from multiple directories.
    
    Args:
        data_dirs: List of directories containing class subdirectories
    
    Returns:
        X: feature array (N, num_features)
        y: labels array (N,)
        class_names: list of class names
    """
    print("Loading training images...")
    
    features = []
    labels = []
    class_names = []
    
    for data_dir in data_dirs:
        if not os.path.exists(data_dir):
            print(f"  Skipping {data_dir} (not found)")
            continue
            
        # Find all class directories
        for entry in sorted(os.listdir(data_dir)):
            entry_path = os.path.join(data_dir, entry)
            if os.path.isdir(entry_path) and not entry.startswith('.'):
                class_idx = len(class_names)
                class_names.append(entry)
                
                count = 0
                # Load all images in this directory
                for filename in os.listdir(entry_path):
                    if filename.endswith('.png'):
                        filepath = os.path.join(entry_path, filename)
                        
                        # Load as grayscale
                        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
                        if img is None:
                            continue
                        
                        # Extract features
                        feat = extract_hog_features(img)
                        features.append(feat)
                        labels.append(class_idx)
                        count += 1
                
                print(f"  {entry}: {count} images")
    
    X = np.array(features)
    y = np.array(labels)
    
    print(f"  Total: {len(X)} images, {X.shape[1]} features per image")
    
    return X, y, class_names


def train_model(
    data_dirs: list,
    output_dir: str,
):
    """Train the shape classifier and save model."""
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report
    
    # Load data
    X, y, class_names = load_training_data(data_dirs)
    
    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTrain: {len(X_train)}, Test: {len(X_test)}")
    
    # Scale features
    print("\nScaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train SVM
    print("Training SVM classifier...")
    model = SVC(kernel='rbf', probability=True, C=10, gamma='scale')
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nTest accuracy: {accuracy:.1%}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=class_names))
    
    # Save model
    os.makedirs(output_dir, exist_ok=True)
    
    model_path = os.path.join(output_dir, "shape_classifier.joblib")
    scaler_path = os.path.join(output_dir, "scaler.joblib")
    meta_path = os.path.join(output_dir, "model_metadata.json")
    
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    
    # Build display names from training data metadata
    display_names = {}
    for data_dir in data_dirs:
        # All directories now use display_names.json
        names_file = os.path.join(data_dir, "display_names.json")
        if os.path.exists(names_file):
            with open(names_file, 'r') as f:
                names_data = json.load(f)
            display_names.update(names_data)
    
    # Generate default display names for any missing
    for name in class_names:
        if name not in display_names:
            if name.startswith("letter_"):
                display_names[name] = f"Letter {name.split('_')[1]}"
            else:
                display_names[name] = name.replace("_", " ").title()
    
    metadata = {
        "class_names": class_names,
        "display_names": display_names,
        "input_size": INPUT_SIZE,
        "num_classes": len(class_names),
        "accuracy": float(accuracy),
    }
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nSaved model to {model_path}")
    print(f"Saved scaler to {scaler_path}")
    print(f"Saved metadata to {meta_path}")
    
    return model, scaler, class_names


class ShapeClassifier:
    """Fast shape classifier using HOG + SVM."""
    
    def __init__(self, model_dir: str):
        """Load model from directory."""
        self.model_dir = model_dir
        self.model = None
        self.scaler = None
        self.class_names = []
        self.display_names = {}  # Maps class_name -> display_name
        self.input_size = INPUT_SIZE
        
        self._load_model()
    
    def _load_model(self):
        """Load model, scaler, and metadata."""
        # Load metadata
        meta_path = os.path.join(self.model_dir, "model_metadata.json")
        with open(meta_path, 'r') as f:
            metadata = json.load(f)
        
        self.class_names = metadata["class_names"]
        self.input_size = metadata.get("input_size", INPUT_SIZE)
        self.display_names = metadata.get("display_names", {})
        
        # If no display names in metadata, generate defaults
        if not self.display_names:
            for name in self.class_names:
                # Convert "letter_X" to "Letter X", "ring" to "Ring", etc.
                if name.startswith("letter_"):
                    self.display_names[name] = f"Letter {name.split('_')[1]}"
                else:
                    self.display_names[name] = name.replace("_", " ").title()
        
        # Load model and scaler
        model_path = os.path.join(self.model_dir, "shape_classifier.joblib")
        scaler_path = os.path.join(self.model_dir, "scaler.joblib")
        
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
    
    def get_display_name(self, class_name: str) -> str:
        """Get the display name for a class."""
        return self.display_names.get(class_name, class_name)
    
    def predict(self, silhouette: np.ndarray) -> Tuple[str, float, Dict[str, float]]:
        """
        Classify a silhouette.
        
        Args:
            silhouette: Binary silhouette image (any size, grayscale or RGB)
            
        Returns:
            (predicted_class, confidence, all_scores)
        """
        # Extract features
        features = extract_hog_features(silhouette)
        features = features.reshape(1, -1)
        
        # Scale
        features_scaled = self.scaler.transform(features)
        
        # Predict
        predicted_idx = self.model.predict(features_scaled)[0]
        probabilities = self.model.predict_proba(features_scaled)[0]
        
        predicted_class = self.class_names[predicted_idx]
        confidence = probabilities[predicted_idx]
        
        # All scores as dict
        all_scores = {name: float(prob) for name, prob in zip(self.class_names, probabilities)}
        
        return predicted_class, float(confidence), all_scores


if __name__ == "__main__":
    import sys
    
    print("=" * 50)
    print("SHAPE CLASSIFIER TRAINING (scikit-learn)")
    print("=" * 50)
    
    # Training data directories
    data_dirs = [
        "assets/training/shapes",
        "assets/training/letters",
        "assets/training/custom",
    ]
    model_dir = "models"
    
    # Check what training data exists
    found_dirs = []
    for d in data_dirs:
        if os.path.exists(d):
            subdirs = [s for s in os.listdir(d) 
                      if os.path.isdir(os.path.join(d, s)) and not s.startswith('.')]
            if subdirs:
                found_dirs.append(d)
                print(f"\nFound {d}: {subdirs}")
    
    if not found_dirs:
        print("ERROR: No training data found!")
        print("Run 'make generate' first!")
        sys.exit(1)
    
    model, scaler, class_names = train_model(
        data_dirs=found_dirs,
        output_dir=model_dir,
    )
    
    print("\nâœ“ Training complete!")
