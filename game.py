#!/usr/bin/env python3
"""
Shadow Puppets - Main Game
Teams form silhouettes to match target shapes!
"""

import sys
import time
import os
import random
import glob
import csv
from datetime import datetime
import numpy as np
import pygame
import cv2
import mediapipe as mp

from camera import Camera
from improved_silhouette import create_improved_processor
from shape_classifier import ShapeClassifier


# Game settings
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 800
CAPTURE_WIDTH = 320
CAPTURE_HEIGHT = 240
GAME_DURATION = 60  # 1 minute total game time
SHAPE_TIMEOUT = 15  # seconds per shape max
COUNTDOWN_DURATION = 3  # seconds before each shape
MATCH_THRESHOLD = 0.4  # minimum match to count as success
MATCH_DELAY = 2.0  # seconds before match can trigger after shape starts
AUTO_CAPTURE_ON_MATCH = True  # automatically save silhouette when match threshold is reached
FULLSCREEN = True

# Camera device - can be overridden with CAMERA_DEVICE environment variable
CAMERA_DEVICE = int(os.environ.get('CAMERA_DEVICE', '0'))

# File paths
HIGH_SCORE_FILE = "data/high_score.txt"
LOG_DIR = "data/logs"
CAPTURE_DIR = "data/captures"

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (128, 128, 128)
DARK_GRAY = (40, 40, 40)
GREEN = (50, 205, 50)
YELLOW = (255, 215, 0)
RED = (220, 60, 60)
BLUE = (70, 130, 180)


class Game:
    """Main game class for Shadow Puppets."""
    
    def __init__(self, model_dir: str = "models"):
        self.model_dir = model_dir
        
        # Game state
        self.state = "title"  # title, countdown, playing, shape_result, game_over
        self.score = 0
        self.high_score = 0
        self.shapes_completed = 0  # number of shapes matched this game
        
        # Game timing
        self.game_time_remaining = GAME_DURATION  # 3 minute game clock
        self.game_start_time = 0
        self.game_time_spent = 0  # accumulated time (paused during countdown)
        
        # Shape timing
        self.shape_time_remaining = SHAPE_TIMEOUT
        self.shape_start_time = 0
        
        # Countdown
        self.countdown_start_time = 0
        
        # Current shape state
        self.target_shape = ""
        self.best_match_this_shape = 0.0
        self.match_achieved = False
        self.success_sound_played = False
        
        # Shape queue (prevents repeats until all shapes used)
        self.available_shapes = []
        
        # Classification
        self.current_prediction = ""
        self.current_confidence = 0.0
        self.target_confidence = 0.0  # How well current pose matches target
        
        # Smoothed classification (average over multiple frames)
        self.classification_history = []  # List of (prediction, confidence, all_scores)
        self.classification_window = 10  # Number of frames to average
        self.display_prediction = ""
        self.display_confidence = 0.0
        self.display_target_confidence = 0.0
        self.last_display_update = 0
        self.display_update_interval = 2.0  # seconds
        
        # Components
        self.camera: Camera = None
        self.segmentation = None
        self.processor = None
        self.classifier: ShapeClassifier = None
        
        # Pygame
        self.screen = None
        self.clock = None
        self.fonts = {}
        
        # Audio
        self.bgm_files = []
        self.previous_bgm = None  # Track last played BGM
        self.sound_boom = None
        self.sound_bell = None
        
        # Logging
        self.current_log_file = None
        self.current_log_writer = None
        self.current_silhouette = None  # Store for capture
        
        # Load high score
        self.high_score = self._load_high_score()
        
        # Silhouette surface (for scaling)
        self.silhouette_surface = None
        
        # FPS tracking
        self.frame_times = []
        self.fps = 0.0
    
    def setup(self) -> bool:
        """Initialize all components."""
        print("Setting up Shadow Puppets...")
        
        # Camera
        print(f"  Opening camera (device={CAMERA_DEVICE})...")
        self.camera = Camera(device=CAMERA_DEVICE, width=CAPTURE_WIDTH, height=CAPTURE_HEIGHT)
        if not self.camera.open():
            print(f"  ERROR: Could not open camera device {CAMERA_DEVICE}")
            return False
        print(f"  Camera ready")
        
        # MediaPipe
        print("  Loading MediaPipe...")
        mp_selfie = mp.solutions.selfie_segmentation
        self.segmentation = mp_selfie.SelfieSegmentation(model_selection=1)
        print("  MediaPipe ready")
        
        # Silhouette processor (improved version with guided filter)
        self.processor = create_improved_processor("default")
        print("  Silhouette processor ready (improved)")
        
        # Shape classifier
        if os.path.exists(self.model_dir):
            print("  Loading shape classifier...")
            try:
                self.classifier = ShapeClassifier(self.model_dir)
                print(f"  Classifier ready ({len(self.classifier.class_names)} shapes)")
            except Exception as e:
                print(f"  ERROR: Could not load classifier: {e}")
                return False
        else:
            print(f"  ERROR: Model directory not found: {self.model_dir}")
            return False
        
        # Pygame
        print("  Initializing display...")
        pygame.init()
        
        # Try to initialize audio (optional - game works without it)
        self.audio_available = False
        try:
            pygame.mixer.init()
            self.audio_available = True
            print("  Audio ready")
        except pygame.error as e:
            print(f"  Audio not available: {e}")
            print("  (Game will run without sound)")
        
        if FULLSCREEN:
            info = pygame.display.Info()
            screen_w, screen_h = info.current_w, info.current_h
            self.screen = pygame.display.set_mode((screen_w, screen_h), pygame.FULLSCREEN)
            # Update layout constants to match actual screen
            global WINDOW_WIDTH, WINDOW_HEIGHT
            WINDOW_WIDTH, WINDOW_HEIGHT = screen_w, screen_h
        else:
            self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))

        pygame.display.set_caption("Shadow Puppets")
        self.clock = pygame.time.Clock()
        
        # Fonts - Bombardier for headings, Carlito for body text
        heading_font = "fonts/BOMBARD_.otf"
        body_font = "fonts/Carlito-Regular.ttf"
        
        # Check for custom fonts
        have_heading = os.path.exists(heading_font)
        have_body = os.path.exists(body_font)
        
        if have_heading:
            print(f"  Heading font: {heading_font}")
        else:
            print(f"  Heading font not found: {heading_font} (using default)")
            heading_font = None
            
        if have_body:
            print(f"  Body font: {body_font}")
        else:
            print(f"  Body font not found: {body_font} (using default)")
            body_font = None
        
        self.fonts = {
            # Heading fonts (Bombardier)
            "huge": pygame.font.Font(heading_font, 90) if heading_font else pygame.font.Font(None, 120),
            "countdown": pygame.font.Font(heading_font, 120) if heading_font else pygame.font.Font(None, 200),
            "large": pygame.font.Font(heading_font, 54) if heading_font else pygame.font.Font(None, 72),
            # Body fonts (Carlito)
            "medium": pygame.font.Font(body_font, 36) if body_font else pygame.font.Font(None, 48),
            "small": pygame.font.Font(body_font, 24) if body_font else pygame.font.Font(None, 32),
        }
        
        print("  Display ready")
        
        # Load audio
        print("  Loading audio...")
        self._load_audio()
        
        print("Setup complete!\n")
        
        return True
    
    def _load_audio(self):
        """Load background music and sound effects."""
        if not self.audio_available:
            return
        
        # Find BGM files
        bgm_dir = "bgm"
        if os.path.exists(bgm_dir):
            self.bgm_files = glob.glob(os.path.join(bgm_dir, "*.ogg"))
            if self.bgm_files:
                print(f"    Found {len(self.bgm_files)} background music files")
            else:
                print(f"    No .ogg files found in {bgm_dir}/")
        else:
            print(f"    BGM directory not found ({bgm_dir}/)")
        
        # Load sound effects
        sfx_dir = "sfx"
        if os.path.exists(sfx_dir):
            boom_path = os.path.join(sfx_dir, "boom.ogg")
            bell_path = os.path.join(sfx_dir, "bell.ogg")
            
            if os.path.exists(boom_path):
                self.sound_boom = pygame.mixer.Sound(boom_path)
                print(f"    Loaded boom sound")
            else:
                print(f"    No boom.ogg found in {sfx_dir}/")
            
            if os.path.exists(bell_path):
                self.sound_bell = pygame.mixer.Sound(bell_path)
                print(f"    Loaded bell sound")
            else:
                print(f"    No bell.ogg found in {sfx_dir}/")
        else:
            print(f"    SFX directory not found ({sfx_dir}/)")
    
    def _play_random_bgm(self):
        """Start playing a random background music track (avoiding repeats)."""
        if not self.audio_available or not self.bgm_files:
            return
        
        # Choose a different track than last time
        available = [t for t in self.bgm_files if t != self.previous_bgm]
        if not available:
            available = self.bgm_files  # Only one track, use it
        
        track = random.choice(available)
        self.previous_bgm = track
        
        try:
            pygame.mixer.music.load(track)
            pygame.mixer.music.play(-1)  # Loop indefinitely
            print(f"  Playing: {os.path.basename(track)}")
        except Exception as e:
            print(f"  Could not play music: {e}")
    
    def _stop_bgm(self):
        """Stop background music."""
        if not self.audio_available:
            return
        pygame.mixer.music.stop()
    
    def _load_high_score(self) -> int:
        """Load high score from file."""
        try:
            if os.path.exists(HIGH_SCORE_FILE):
                with open(HIGH_SCORE_FILE, 'r') as f:
                    return int(f.read().strip())
        except (ValueError, IOError):
            pass
        return 0
    
    def _save_high_score(self):
        """Save high score to file."""
        try:
            os.makedirs(os.path.dirname(HIGH_SCORE_FILE), exist_ok=True)
            with open(HIGH_SCORE_FILE, 'w') as f:
                f.write(str(self.high_score))
        except IOError as e:
            print(f"  Could not save high score: {e}")
    
    def _start_game_log(self):
        """Start a new log file for this game session."""
        os.makedirs(LOG_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(LOG_DIR, f"game_{timestamp}.csv")
        
        self.current_log_file = open(log_path, 'w', newline='')
        self.current_log_writer = csv.writer(self.current_log_file)
        self.current_log_writer.writerow(['shape', 'matched', 'match_percent', 'time_seconds'])
        print(f"  Logging to {log_path}")
    
    def _log_shape_result(self, shape: str, matched: bool, match_percent: float, time_seconds: float):
        """Log a shape result to the CSV file."""
        if self.current_log_writer:
            self.current_log_writer.writerow([
                shape,
                1 if matched else 0,
                f"{match_percent:.1f}",
                f"{time_seconds:.1f}"
            ])
            self.current_log_file.flush()
    
    def _close_game_log(self):
        """Close the current log file."""
        if self.current_log_file:
            self.current_log_file.close()
            self.current_log_file = None
            self.current_log_writer = None
    
    def _capture_silhouette(self):
        """Save current silhouette image for training data."""
        if self.current_silhouette is None or not self.target_shape:
            print("  No silhouette to capture")
            return
        
        os.makedirs(CAPTURE_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%m%d%Y%H%M%S")
        filename = f"{self.target_shape}_{timestamp}.png"
        filepath = os.path.join(CAPTURE_DIR, filename)
        
        # Convert RGB silhouette to grayscale for saving
        # The silhouette is black on white, we want to save it as-is
        gray = cv2.cvtColor(self.current_silhouette, cv2.COLOR_RGB2GRAY)
        cv2.imwrite(filepath, gray)
        print(f"  Captured: {filepath}")
    
    def _play_boom(self):
        """Play the boom sound effect."""
        if self.sound_boom:
            self.sound_boom.play()
    
    def _play_bell(self):
        """Play the bell/success sound effect."""
        if self.sound_bell:
            self.sound_bell.play()
    
    def cleanup(self):
        """Release all resources."""
        self._close_game_log()
        if self.camera:
            self.camera.close()
        if self.segmentation:
            self.segmentation.close()
        if self.audio_available:
            pygame.mixer.quit()
        pygame.quit()
    
    def start_game(self):
        """Start a new game."""
        self.score = 0
        self.shapes_completed = 0
        self.game_time_spent = 0
        self.game_time_remaining = GAME_DURATION
        # Reset shape queue - shuffle all shapes
        self.available_shapes = list(self.classifier.class_names)
        random.shuffle(self.available_shapes)
        self._start_game_log()
        self.start_countdown()
    
    def start_countdown(self):
        """Start the countdown before a shape."""
        # Pick next shape from queue (refill if empty)
        if not self.available_shapes:
            self.available_shapes = list(self.classifier.class_names)
            random.shuffle(self.available_shapes)
        
        self.target_shape = self.available_shapes.pop()
        self.countdown_start_time = time.time()
        self.state = "countdown"
        self.success_sound_played = False
        self.best_match_this_shape = 0.0
        self.match_achieved = False
        print(f"Next shape: {self.target_shape} (Game time: {int(self.game_time_remaining)}s remaining)")
    
    def start_shape(self):
        """Start the actual shape attempt after countdown."""
        self.shape_start_time = time.time()
        self.shape_time_remaining = SHAPE_TIMEOUT
        self.state = "playing"
        
        # Reset classification state
        self.classification_history = []
        self.display_prediction = ""
        self.display_confidence = 0.0
        self.display_target_confidence = 0.0
        self.last_display_update = 0
        
        # Start music
        self._play_random_bgm()
        
        print(f"  GO! Make a {self.target_shape}!")
    
    def complete_shape(self, success: bool):
        """Complete the current shape (either matched or timed out)."""
        # Stop music
        self._stop_bgm()
        
        # Calculate time spent on this shape
        shape_elapsed = time.time() - self.shape_start_time
        
        # Add elapsed shape time to game time spent
        self.game_time_spent += shape_elapsed
        self.game_time_remaining = GAME_DURATION - self.game_time_spent
        
        # Log the result
        self._log_shape_result(
            self.target_shape,
            success,
            self.best_match_this_shape * 100,
            shape_elapsed
        )
        
        if success:
            # Award points based on best match
            points = int(self.best_match_this_shape * 200)
            self.score += points
            self.shapes_completed += 1
            self.match_achieved = True
            self._play_bell()
            print(f"  Matched! +{points} points (total: {self.score})")
        else:
            # Shape timed out
            self._play_boom()
            print(f"  Time's up! Best: {self.best_match_this_shape:.0%}")
        
        # Update high score
        if self.score > self.high_score:
            self.high_score = self.score
            self._save_high_score()
        
        # Check if game is over
        if self.game_time_remaining <= 0:
            self._close_game_log()
            self.state = "game_over"
            print(f"\nGame Over! Final score: {self.score} ({self.shapes_completed} shapes)")
        else:
            # Immediately start next shape
            self.start_countdown()
    
    def process_frame(self, frame_rgb: np.ndarray) -> np.ndarray:
        """Process a frame and update classification."""
        # Get segmentation mask
        results = self.segmentation.process(frame_rgb)
        
        # Create silhouette image (pass RGB frame for guided filtering)
        silhouette = self.processor.create_silhouette_image(
            results.segmentation_mask,
            CAPTURE_WIDTH,
            CAPTURE_HEIGHT,
            guide_frame=frame_rgb
        )
        
        # Store for potential capture
        self.current_silhouette = silhouette
        
        # Classify if we have enough silhouette
        if self.classifier is not None and results.segmentation_mask is not None:
            mask = results.segmentation_mask
            person_pixels = np.sum(mask > 0.5)
            total_pixels = mask.shape[0] * mask.shape[1]
            coverage = person_pixels / total_pixels
            
            if coverage > 0.05:
                pred, conf, all_scores = self.classifier.predict(silhouette)
                
                # Store in history
                self.classification_history.append((pred, conf, all_scores))
                if len(self.classification_history) > self.classification_window:
                    self.classification_history.pop(0)
                
                # Get confidence for target shape (instant, for best tracking)
                self.target_confidence = all_scores.get(self.target_shape, 0.0)
                
                # Track best match this shape (use instant value)
                if self.target_confidence > self.best_match_this_shape:
                    self.best_match_this_shape = self.target_confidence
                
                # Check for success - complete shape when threshold reached
                # But only after MATCH_DELAY seconds have passed since shape started
                shape_elapsed = time.time() - self.shape_start_time
                if (self.state == "playing" and 
                    shape_elapsed >= MATCH_DELAY and
                    self.target_confidence >= MATCH_THRESHOLD and 
                    not self.success_sound_played):
                    # Auto-capture silhouette if enabled
                    if AUTO_CAPTURE_ON_MATCH:
                        self._capture_silhouette()
                    self.success_sound_played = True
                    self.complete_shape(success=True)
            else:
                self.target_confidence = 0.0
        
        # Update display values periodically
        current_time = time.time()
        if current_time - self.last_display_update >= self.display_update_interval:
            self._update_display_classification()
            self.last_display_update = current_time
        
        return silhouette
    
    def _update_display_classification(self):
        """Update the displayed classification by averaging recent frames."""
        if not self.classification_history:
            self.display_prediction = ""
            self.display_confidence = 0.0
            self.display_target_confidence = 0.0
            return
        
        # Aggregate scores across all frames
        score_totals = {}
        for pred, conf, all_scores in self.classification_history:
            for shape, score in all_scores.items():
                if shape not in score_totals:
                    score_totals[shape] = []
                score_totals[shape].append(score)
        
        # Average scores
        avg_scores = {shape: sum(scores) / len(scores) for shape, scores in score_totals.items()}
        
        # Find best prediction
        best_shape = max(avg_scores, key=avg_scores.get)
        self.display_prediction = best_shape
        self.display_confidence = avg_scores[best_shape]
        self.display_target_confidence = avg_scores.get(self.target_shape, 0.0)
    
    def update_fps(self, frame_time: float):
        """Update FPS calculation."""
        self.frame_times.append(frame_time)
        if len(self.frame_times) > 30:
            self.frame_times.pop(0)
        if self.frame_times:
            avg_time = sum(self.frame_times) / len(self.frame_times)
            self.fps = 1.0 / avg_time if avg_time > 0 else 0
    
    def draw_title_screen(self):
        """Draw the title screen."""
        self.screen.fill(DARK_GRAY)
        
        # Title
        title = self.fonts["huge"].render("SHADOW PUPPETS", True, WHITE)
        title_rect = title.get_rect(center=(WINDOW_WIDTH // 2, 200))
        self.screen.blit(title, title_rect)
        
        # Subtitle
        subtitle = self.fonts["medium"].render("Form shapes with your silhouette!", True, GRAY)
        subtitle_rect = subtitle.get_rect(center=(WINDOW_WIDTH // 2, 300))
        self.screen.blit(subtitle, subtitle_rect)
        
        # Instructions
        instructions = [
            "Stand in front of the camera",
            "Match as many shapes as you can in 1 minute!",
            f"You have {SHAPE_TIMEOUT} seconds max per shape",
        ]
        for i, text in enumerate(instructions):
            inst = self.fonts["small"].render(text, True, GRAY)
            inst_rect = inst.get_rect(center=(WINDOW_WIDTH // 2, 420 + i * 40))
            self.screen.blit(inst, inst_rect)
        
        # High score
        if self.high_score > 0:
            hs = self.fonts["medium"].render(f"High Score: {self.high_score}", True, YELLOW)
            hs_rect = hs.get_rect(center=(WINDOW_WIDTH // 2, 580))
            self.screen.blit(hs, hs_rect)
        
        # Start prompt
        start = self.fonts["large"].render("Press SPACE to start", True, GREEN)
        start_rect = start.get_rect(center=(WINDOW_WIDTH // 2, 680))
        self.screen.blit(start, start_rect)
    
    def draw_countdown_screen(self):
        """Draw the countdown screen before shape attempt."""
        self.screen.fill(BLACK)
        
        # Calculate countdown
        elapsed = time.time() - self.countdown_start_time
        remaining = COUNTDOWN_DURATION - elapsed
        
        if remaining <= 0:
            # Countdown finished, start the shape
            self.start_shape()
            return
        
        # Game time remaining (top)
        game_mins = int(self.game_time_remaining) // 60
        game_secs = int(self.game_time_remaining) % 60
        time_text = self.fonts["medium"].render(f"Game Time: {game_mins}:{game_secs:02d}", True, GRAY)
        time_rect = time_text.get_rect(center=(WINDOW_WIDTH // 2, 80))
        self.screen.blit(time_text, time_rect)
        
        # Shapes completed
        shapes_text = self.fonts["small"].render(f"Shapes: {self.shapes_completed} | Score: {self.score}", True, GRAY)
        shapes_rect = shapes_text.get_rect(center=(WINDOW_WIDTH // 2, 120))
        self.screen.blit(shapes_text, shapes_rect)
        
        # "Next Shape" label
        next_text = self.fonts["large"].render("Next Shape:", True, WHITE)
        next_rect = next_text.get_rect(center=(WINDOW_WIDTH // 2, 220))
        self.screen.blit(next_text, next_rect)
        
        # Target shape (use display name)
        display_name = self.classifier.get_display_name(self.target_shape)
        target_text = self.fonts["huge"].render(display_name.upper(), True, YELLOW)
        target_rect = target_text.get_rect(center=(WINDOW_WIDTH // 2, 350))
        self.screen.blit(target_text, target_rect)
        
        # Countdown number
        countdown_num = int(remaining) + 1
        countdown_text = self.fonts["countdown"].render(str(countdown_num), True, WHITE)
        countdown_rect = countdown_text.get_rect(center=(WINDOW_WIDTH // 2, 550))
        self.screen.blit(countdown_text, countdown_rect)
        
        # "Get Ready" text
        ready_text = self.fonts["medium"].render("Get Ready!", True, GRAY)
        ready_rect = ready_text.get_rect(center=(WINDOW_WIDTH // 2, 700))
        self.screen.blit(ready_text, ready_rect)
    
    def draw_game_screen(self, silhouette: np.ndarray):
        """Draw the main game screen."""
        self.screen.fill(DARK_GRAY)
        
        # Scale and draw silhouette (centered, large)
        silhouette_display = np.transpose(silhouette, (1, 0, 2))
        silhouette_display = np.flip(silhouette_display, axis=0)
        surface = pygame.surfarray.make_surface(silhouette_display)
        
        # Scale to fit nicely (leave room for UI)
        scale_height = WINDOW_HEIGHT - 180  # Room for top and bottom UI
        scale_width = int(scale_height * CAPTURE_WIDTH / CAPTURE_HEIGHT)
        scaled = pygame.transform.scale(surface, (scale_width, scale_height))
        
        silhouette_x = (WINDOW_WIDTH - scale_width) // 2
        silhouette_y = 80
        self.screen.blit(scaled, (silhouette_x, silhouette_y))
        
        # Draw border around silhouette
        pygame.draw.rect(self.screen, GRAY, (silhouette_x - 2, silhouette_y - 2, scale_width + 4, scale_height + 4), 2)
        
        # Top bar - Target shape, score, and shapes completed
        pygame.draw.rect(self.screen, BLACK, (0, 0, WINDOW_WIDTH, 70))
        
        # Target shape (left) - use display name
        display_name = self.classifier.get_display_name(self.target_shape)
        target_text = self.fonts["large"].render(f"Make: {display_name.upper()}", True, YELLOW)
        self.screen.blit(target_text, (20, 15))
        
        # Score and shapes (right)
        score_text = self.fonts["medium"].render(f"Score: {self.score}", True, WHITE)
        score_rect = score_text.get_rect(topright=(WINDOW_WIDTH - 20, 8))
        self.screen.blit(score_text, score_rect)
        
        shapes_text = self.fonts["small"].render(f"Shapes: {self.shapes_completed}", True, GRAY)
        shapes_rect = shapes_text.get_rect(topright=(WINDOW_WIDTH - 20, 42))
        self.screen.blit(shapes_text, shapes_rect)
        
        # Shape timer (center top) - seconds left for this shape
        shape_secs = int(self.shape_time_remaining)
        if shape_secs <= 10:
            timer_color = RED
        elif shape_secs <= 20:
            timer_color = YELLOW
        else:
            timer_color = WHITE
        shape_timer_text = self.fonts["medium"].render(f"{shape_secs}s", True, timer_color)
        shape_timer_rect = shape_timer_text.get_rect(center=(WINDOW_WIDTH // 2, 35))
        self.screen.blit(shape_timer_text, shape_timer_rect)
        
        # Bottom bar - Match info and game timer
        bottom_bar_y = WINDOW_HEIGHT - 100
        pygame.draw.rect(self.screen, BLACK, (0, bottom_bar_y, WINDOW_WIDTH, 100))
        
        # Current best guess (left side) - uses smoothed display values
        if self.display_prediction:
            pred_display_name = self.classifier.get_display_name(self.display_prediction)
            guess_color = GREEN if self.display_prediction == self.target_shape else WHITE
            guess_text = self.fonts["medium"].render(
                f"Looks like: {pred_display_name} ({int(self.display_confidence * 100)}%)",
                True, guess_color
            )
            self.screen.blit(guess_text, (20, bottom_bar_y + 10))
        
        # Match percentage for target (center) - uses instant value for responsiveness
        match_pct = int(self.target_confidence * 100)
        if match_pct >= 30:
            match_color = GREEN
        elif match_pct >= 20:
            match_color = YELLOW
        else:
            match_color = RED
        
        match_text = self.fonts["medium"].render(f"Match: {match_pct}%", True, match_color)
        match_rect = match_text.get_rect(center=(WINDOW_WIDTH // 2, bottom_bar_y + 25))
        self.screen.blit(match_text, match_rect)
        
        # Best this shape
        best_text = self.fonts["small"].render(f"Best: {int(self.best_match_this_shape * 100)}%", True, GRAY)
        best_rect = best_text.get_rect(center=(WINDOW_WIDTH // 2, bottom_bar_y + 55))
        self.screen.blit(best_text, best_rect)
        
        # Game timer bar (shows 3 minute game time, not shape time)
        timer_bar_y = bottom_bar_y + 75
        timer_bar_height = 20
        timer_bar_width = WINDOW_WIDTH - 40
        timer_x = 20
        
        # Background
        pygame.draw.rect(self.screen, GRAY, (timer_x, timer_bar_y, timer_bar_width, timer_bar_height))
        
        # Filled portion (game time remaining)
        fill_width = int(timer_bar_width * (self.game_time_remaining / GAME_DURATION))
        if self.game_time_remaining > 60:
            timer_color = GREEN
        elif self.game_time_remaining > 30:
            timer_color = YELLOW
        else:
            timer_color = RED
        pygame.draw.rect(self.screen, timer_color, (timer_x, timer_bar_y, fill_width, timer_bar_height))
        
        # Game time text
        game_mins = int(self.game_time_remaining) // 60
        game_secs = int(self.game_time_remaining) % 60
        time_text = self.fonts["small"].render(f"{game_mins}:{game_secs:02d}", True, WHITE)
        time_rect = time_text.get_rect(midright=(WINDOW_WIDTH - 25, timer_bar_y + timer_bar_height // 2))
        self.screen.blit(time_text, time_rect)
        
        # FPS (small, corner)
        fps_text = self.fonts["small"].render(f"{self.fps:.0f} FPS", True, GRAY)
        self.screen.blit(fps_text, (silhouette_x + 5, silhouette_y + 5))
    
    def draw_game_over_screen(self):
        """Draw the game over screen."""
        self.screen.fill(DARK_GRAY)
        
        # Title
        title = self.fonts["huge"].render("GAME OVER!", True, WHITE)
        title_rect = title.get_rect(center=(WINDOW_WIDTH // 2, 200))
        self.screen.blit(title, title_rect)
        
        # Final score
        score_text = self.fonts["huge"].render(f"{self.score}", True, YELLOW)
        score_rect = score_text.get_rect(center=(WINDOW_WIDTH // 2, 320))
        self.screen.blit(score_text, score_rect)
        
        score_label = self.fonts["medium"].render("Final Score", True, GRAY)
        score_label_rect = score_label.get_rect(center=(WINDOW_WIDTH // 2, 380))
        self.screen.blit(score_label, score_label_rect)
        
        # Shapes completed
        shapes_text = self.fonts["large"].render(f"{self.shapes_completed} shapes matched", True, WHITE)
        shapes_rect = shapes_text.get_rect(center=(WINDOW_WIDTH // 2, 460))
        self.screen.blit(shapes_text, shapes_rect)
        
        # High score
        if self.score >= self.high_score and self.score > 0:
            hs_text = self.fonts["large"].render("NEW HIGH SCORE!", True, GREEN)
        else:
            hs_text = self.fonts["medium"].render(f"High Score: {self.high_score}", True, GRAY)
        hs_rect = hs_text.get_rect(center=(WINDOW_WIDTH // 2, 540))
        self.screen.blit(hs_text, hs_rect)
        
        # Play again prompt
        play_text = self.fonts["medium"].render("Press SPACE to play again", True, GREEN)
        play_rect = play_text.get_rect(center=(WINDOW_WIDTH // 2, 640))
        self.screen.blit(play_text, play_rect)
        
        # Quit option
        quit_text = self.fonts["small"].render("Press Q to quit", True, GRAY)
        quit_rect = quit_text.get_rect(center=(WINDOW_WIDTH // 2, 700))
        self.screen.blit(quit_text, quit_rect)
    
    def handle_events(self) -> bool:
        """Handle input events. Returns False to quit."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                elif event.key == pygame.K_q:
                    if self.state in ("title", "game_over"):
                        return False
                elif event.key == pygame.K_SPACE:
                    if self.state == "title":
                        self.start_game()
                    elif self.state == "game_over":
                        self.start_game()
                elif event.key == pygame.K_p:
                    # Capture silhouette for training data
                    if self.state == "playing":
                        self._capture_silhouette()
        
        return True
    
    def run(self):
        """Main game loop."""
        print("Starting game... (ESC to quit)")
        
        running = True
        while running:
            frame_start = time.time()
            
            # Handle events
            running = self.handle_events()
            if not running:
                break
            
            # Update game state
            if self.state == "playing":
                # Update shape timer
                shape_elapsed = time.time() - self.shape_start_time
                self.shape_time_remaining = max(0, SHAPE_TIMEOUT - shape_elapsed)
                
                # Update game timer (accumulated time + current shape time)
                current_game_time = self.game_time_spent + shape_elapsed
                self.game_time_remaining = max(0, GAME_DURATION - current_game_time)
                
                # Check for shape timeout
                if self.shape_time_remaining <= 0:
                    self.complete_shape(success=False)
                # Check for game timeout
                elif self.game_time_remaining <= 0:
                    self.complete_shape(success=False)
            
            # Capture and process frame
            ret, frame_rgb = self.camera.read()
            silhouette = None
            if ret:
                silhouette = self.process_frame(frame_rgb)
            
            # Draw appropriate screen
            if self.state == "title":
                self.draw_title_screen()
            elif self.state == "countdown":
                self.draw_countdown_screen()
            elif self.state == "playing" and silhouette is not None:
                self.draw_game_screen(silhouette)
            elif self.state == "game_over":
                self.draw_game_over_screen()
            
            pygame.display.flip()
            
            # FPS
            frame_time = time.time() - frame_start
            self.update_fps(frame_time)
        
        # Cleanup audio
        self._stop_bgm()
        
        print(f"\nFinal score: {self.score}")
        if self.score > 0:
            print(f"High score: {self.high_score}")


def main():
    print("=" * 50)
    print("SHADOW PUPPETS")
    print("=" * 50)
    print()
    
    game = Game(model_dir="models")
    
    if not game.setup():
        print("Setup failed!")
        return 1
    
    try:
        game.run()
    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        game.cleanup()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
