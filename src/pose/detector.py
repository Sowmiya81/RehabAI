"""
Pose detection module using MediaPipe Pose.

This module provides the PoseDetector class for detecting human pose landmarks
in images and videos using Google's MediaPipe Pose solution.
"""

import cv2
import mediapipe as mp
import numpy as np
from typing import Optional
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PoseDetector:
    """
    A class for detecting human pose landmarks using MediaPipe Pose.
    
    This detector can process both single frames and video files, extracting
    33 pose landmarks (x, y, confidence) for each detected person.
    
    Attributes:
        mp_pose: MediaPipe Pose solution object
        pose: MediaPipe Pose detection object
    """
    
    def __init__(
        self,
        static_image_mode: bool = False,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5
    ) -> None:
        """
        Initialize the PoseDetector with MediaPipe Pose.
        
        Args:
            static_image_mode: If False, treat input as video stream. If True,
                              treat input as static images. Defaults to False.
            model_complexity: Complexity of the pose landmark model (0, 1, or 2).
                             Higher values increase accuracy but decrease speed.
                             Defaults to 1.
            min_detection_confidence: Minimum confidence value for pose detection
                                     to be considered successful. Defaults to 0.5.
            min_tracking_confidence: Minimum confidence value for pose landmarks
                                    to be considered successfully tracked.
                                    Defaults to 0.5.
        """
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        logger.info(
            f"PoseDetector initialized with static_image_mode={static_image_mode}, "
            f"model_complexity={model_complexity}, "
            f"min_detection_confidence={min_detection_confidence}, "
            f"min_tracking_confidence={min_tracking_confidence}"
        )
    
    def process_video(self, video_path: str) -> np.ndarray:
        """
        Process a video file and extract pose landmarks for all frames.
        
        Extracts frames at 30fps and processes each frame with MediaPipe to
        detect 33 pose landmarks. Returns a numpy array containing all landmarks.
        
        Args:
            video_path: Path to the video file to process.
            
        Returns:
            A numpy array of shape (num_frames, 33, 3) where the last dimension
            contains (x, y, confidence) for each of the 33 pose landmarks.
            If no person is detected in any frame, returns an empty array.
            
        Raises:
            FileNotFoundError: If the video file does not exist.
            ValueError: If the video file cannot be opened or is invalid.
        """
        logger.info(f"Processing video: {video_path}")
        
        # Open video file
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            error_msg = f"Failed to open video file: {video_path}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            cap.release()
            error_msg = f"Video file appears to be empty or invalid: {video_path}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        logger.info(f"Video properties: {total_frames} frames at {fps:.2f} fps")
        
        # Calculate frame skip to achieve ~30fps
        frame_skip = max(1, int(fps / 30))
        frames_to_process = total_frames // frame_skip
        
        # Initialize list to store landmarks
        all_landmarks = []
        frame_count = 0
        processed_count = 0
        
        try:
            # Process frames with progress bar
            with tqdm(total=frames_to_process, desc="Processing video") as pbar:
                while True:
                    ret, frame = cap.read()
                    
                    if not ret:
                        break
                    
                    # Skip frames to achieve ~30fps
                    if frame_count % frame_skip == 0:
                        landmarks = self.process_frame(frame)
                        
                        if landmarks is not None:
                            all_landmarks.append(landmarks)
                        else:
                            # If no person detected, add zeros array
                            all_landmarks.append(np.zeros((33, 3), dtype=np.float32))
                            logger.warning(f"No person detected in frame {frame_count}")
                        
                        processed_count += 1
                        pbar.update(1)
                    
                    frame_count += 1
            
            cap.release()
            
            if len(all_landmarks) == 0:
                logger.warning("No frames with pose detections found in video")
                return np.array([]).reshape(0, 33, 3)
            
            # Convert to numpy array
            landmarks_array = np.array(all_landmarks, dtype=np.float32)
            logger.info(
                f"Successfully processed {processed_count} frames. "
                f"Output shape: {landmarks_array.shape}"
            )
            
            return landmarks_array
            
        except Exception as e:
            cap.release()
            error_msg = f"Error processing video: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise
    
    def process_frame(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Process a single frame and extract pose landmarks.
        
        Args:
            frame: Input frame as a numpy array (BGR format from OpenCV).
            
        Returns:
            A numpy array of shape (33, 3) containing (x, y, confidence) for
            each of the 33 pose landmarks. Returns None if no person is detected.
        """
        if frame is None or frame.size == 0:
            logger.warning("Empty or invalid frame provided")
            return None
        
        try:
            # Convert BGR to RGB (MediaPipe requires RGB)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process frame with MediaPipe
            results = self.pose.process(rgb_frame)
            
            # Check if pose was detected
            if results.pose_landmarks is None:
                return None
            
            # Extract landmarks
            landmarks = []
            for landmark in results.pose_landmarks.landmark:
                landmarks.append([
                    landmark.x,
                    landmark.y,
                    landmark.visibility  # MediaPipe uses visibility instead of confidence
                ])
            
            landmarks_array = np.array(landmarks, dtype=np.float32)
            return landmarks_array
            
        except Exception as e:
            logger.error(f"Error processing frame: {str(e)}", exc_info=True)
            return None

