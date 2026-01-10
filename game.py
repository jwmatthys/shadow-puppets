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
import numpy as np
import pygame
import mediapipe as mp

from camera import Camera
from silhouette import create_processor
from shape_classifier import ShapeClassifier


# Game settings
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 800
CAPTURE_WIDTH = 320
CAPTURE_HEIGHT = 240
ROUND_DURATION = 30  # seconds
COUNTDOWN_DURATION = 5  # seconds before round starts
MATCH_THRESHOLD = 0.25  # minimum match to count as success

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
        self.state = "title"  # title, countdown, playing, round_end, game_over
        self.round_number = 0
        self.score = 0
        self.high_score = 0
        
        # Round state
        self.target_shape = ""
        self.round_start_time = 0
        self.countdown_start_time = 0
        self.time_remaining = ROUND_DURATION
        self.best_match_this_round = 0.0
        self.match_achieved = False
        self.success_sound_played = False  # Track if we played the bell this round
        
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
        
        # Silhouette surface (for scaling)
        self.silhouette_surface = None
        
        # FPS tracking
        self.frame_times = []
        self.fps = 0.0
    
    def setup(self) -> bool:
        """Initialize all components."""
        print("Setting up Shadow Puppets...")
        
        # Camera
        print("  Opening camera...")
        self.camera = Camera(width=CAPTURE_WIDTH, height=CAPTURE_HEIGHT)
        if not self.camera.open():
            print("  ERROR: Could not open camera")
            return False
        print(f"  Camera ready")
        
        # MediaPipe
        print("  Loading MediaPipe...")
        mp_selfie = mp.solutions.selfie_segmentation
        self.segmentation = mp_selfie.SelfieSegmentation(model_selection=1)
        print("  MediaPipe ready")
        
        # Silhouette processor
        self.processor = create_processor("default")
        print("  Silhouette processor ready")
        
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
        
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Shadow Puppets")
        self.clock = pygame.time.Clock()
        
        # Fonts
        self.fonts = {
            "small": pygame.font.Font(None, 32),
            "medium": pygame.font.Font(None, 48),
            "large": pygame.font.Font(None, 72),
            "huge": pygame.font.Font(None, 120),
            "countdown": pygame.font.Font(None, 200),
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
        if self.camera:
            self.camera.close()
        if self.segmentation:
            self.segmentation.close()
        if self.audio_available:
            pygame.mixer.quit()
        pygame.quit()
    
    def start_countdown(self):
        """Start the countdown before a round."""
        self.round_number += 1
        self.target_shape = random.choice(self.classifier.class_names)
        self.countdown_start_time = time.time()
        self.state = "countdown"
        self.success_sound_played = False
        print(f"Round {self.round_number}: Get ready to make a {self.target_shape}!")
    
    def start_round(self):
        """Start the actual round after countdown."""
        self.round_start_time = time.time()
        self.time_remaining = ROUND_DURATION
        self.best_match_this_round = 0.0
        self.match_achieved = False
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
    
    def end_round(self):
        """End the current round."""
        # Stop music and play boom
        self._stop_bgm()
        self._play_boom()
        
        # Award points based on best match
        if self.best_match_this_round >= MATCH_THRESHOLD:
            points = int(self.best_match_this_round * 200)
            self.score += points
            self.match_achieved = True
            print(f"  Match! +{points} points (total: {self.score})")
        else:
            print(f"  No match. Best: {self.best_match_this_round:.0%}")
        
        if self.score > self.high_score:
            self.high_score = self.score
        
        self.state = "round_end"
    
    def process_frame(self, frame_rgb: np.ndarray) -> np.ndarray:
        """Process a frame and update classification."""
        # Get segmentation mask
        results = self.segmentation.process(frame_rgb)
        
        # Create silhouette image
        silhouette = self.processor.create_silhouette_image(
            results.segmentation_mask,
            CAPTURE_WIDTH,
            CAPTURE_HEIGHT
        )
        
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
                
                # Track best match this round (use instant value)
                if self.target_confidence > self.best_match_this_round:
                    self.best_match_this_round = self.target_confidence
                
                # Check for success - play bell sound once when threshold reached (only during playing state)
                if self.state == "playing" and self.target_confidence >= MATCH_THRESHOLD and not self.success_sound_played:
                    self._play_bell()
                    self.success_sound_played = True
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
            "Work together to match the target shape",
            f"You have {ROUND_DURATION} seconds per round",
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
        """Draw the countdown screen before round starts."""
        self.screen.fill(BLACK)
        
        # Calculate countdown
        elapsed = time.time() - self.countdown_start_time
        remaining = COUNTDOWN_DURATION - elapsed
        
        if remaining <= 0:
            # Countdown finished, start the round
            self.start_round()
            return
        
        # Round info
        round_text = self.fonts["large"].render(f"Round {self.round_number}", True, WHITE)
        round_rect = round_text.get_rect(center=(WINDOW_WIDTH // 2, 200))
        self.screen.blit(round_text, round_rect)
        
        # Target shape
        target_text = self.fonts["huge"].render(f"{self.target_shape.upper()}", True, YELLOW)
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
        
        # Top bar - Target shape and score
        pygame.draw.rect(self.screen, BLACK, (0, 0, WINDOW_WIDTH, 70))
        
        # Target shape (left)
        target_text = self.fonts["large"].render(f"Make: {self.target_shape.upper()}", True, YELLOW)
        self.screen.blit(target_text, (20, 15))
        
        # Score (right)
        score_text = self.fonts["medium"].render(f"Score: {self.score}", True, WHITE)
        score_rect = score_text.get_rect(topright=(WINDOW_WIDTH - 20, 8))
        self.screen.blit(score_text, score_rect)
        
        # Round number
        round_text = self.fonts["small"].render(f"Round {self.round_number}", True, GRAY)
        round_rect = round_text.get_rect(topright=(WINDOW_WIDTH - 20, 42))
        self.screen.blit(round_text, round_rect)
        
        # Bottom bar - Match info and timer
        bottom_bar_y = WINDOW_HEIGHT - 100
        pygame.draw.rect(self.screen, BLACK, (0, bottom_bar_y, WINDOW_WIDTH, 100))
        
        # Current best guess (left side) - uses smoothed display values
        if self.display_prediction:
            guess_color = GREEN if self.display_prediction == self.target_shape else WHITE
            guess_text = self.fonts["medium"].render(
                f"Looks like: {self.display_prediction} ({int(self.display_confidence * 100)}%)",
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
        
        # Best this round
        best_text = self.fonts["small"].render(f"Best: {int(self.best_match_this_round * 100)}%", True, GRAY)
        best_rect = best_text.get_rect(center=(WINDOW_WIDTH // 2, bottom_bar_y + 55))
        self.screen.blit(best_text, best_rect)
        
        # Timer bar
        timer_bar_y = bottom_bar_y + 75
        timer_bar_height = 20
        timer_bar_width = WINDOW_WIDTH - 40
        timer_x = 20
        
        # Background
        pygame.draw.rect(self.screen, GRAY, (timer_x, timer_bar_y, timer_bar_width, timer_bar_height))
        
        # Filled portion
        fill_width = int(timer_bar_width * (self.time_remaining / ROUND_DURATION))
        if self.time_remaining > 10:
            timer_color = GREEN
        elif self.time_remaining > 5:
            timer_color = YELLOW
        else:
            timer_color = RED
        pygame.draw.rect(self.screen, timer_color, (timer_x, timer_bar_y, fill_width, timer_bar_height))
        
        # Time text
        time_text = self.fonts["small"].render(f"{int(self.time_remaining)}s", True, WHITE)
        time_rect = time_text.get_rect(midright=(WINDOW_WIDTH - 25, timer_bar_y + timer_bar_height // 2))
        self.screen.blit(time_text, time_rect)
        
        # FPS (small, corner)
        fps_text = self.fonts["small"].render(f"{self.fps:.0f} FPS", True, GRAY)
        self.screen.blit(fps_text, (silhouette_x + 5, silhouette_y + 5))
    
    def draw_round_end_screen(self):
        """Draw the round end screen."""
        self.screen.fill(DARK_GRAY)
        
        # Result
        if self.match_achieved:
            result = self.fonts["huge"].render("MATCHED!", True, GREEN)
            points = int(self.best_match_this_round * 200)
            points_text = self.fonts["large"].render(f"+{points} points", True, YELLOW)
        else:
            result = self.fonts["huge"].render("TIME'S UP!", True, RED)
            points_text = self.fonts["large"].render(f"Best: {int(self.best_match_this_round * 100)}%", True, GRAY)
        
        result_rect = result.get_rect(center=(WINDOW_WIDTH // 2, 280))
        self.screen.blit(result, result_rect)
        
        points_rect = points_text.get_rect(center=(WINDOW_WIDTH // 2, 380))
        self.screen.blit(points_text, points_rect)
        
        # Total score
        score_text = self.fonts["large"].render(f"Total Score: {self.score}", True, WHITE)
        score_rect = score_text.get_rect(center=(WINDOW_WIDTH // 2, 480))
        self.screen.blit(score_text, score_rect)
        
        # Continue prompt
        continue_text = self.fonts["medium"].render("Press SPACE for next round", True, GREEN)
        continue_rect = continue_text.get_rect(center=(WINDOW_WIDTH // 2, 600))
        self.screen.blit(continue_text, continue_rect)
        
        # Quit option
        quit_text = self.fonts["small"].render("Press Q to quit", True, GRAY)
        quit_rect = quit_text.get_rect(center=(WINDOW_WIDTH // 2, 660))
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
                    if self.state in ("title", "round_end"):
                        return False
                elif event.key == pygame.K_SPACE:
                    if self.state == "title":
                        self.score = 0
                        self.round_number = 0
                        self.start_countdown()
                    elif self.state == "round_end":
                        self.start_countdown()
        
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
                # Update timer
                elapsed = time.time() - self.round_start_time
                self.time_remaining = max(0, ROUND_DURATION - elapsed)
                
                if self.time_remaining <= 0:
                    self.end_round()
            
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
            elif self.state == "round_end":
                self.draw_round_end_screen()
            
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
