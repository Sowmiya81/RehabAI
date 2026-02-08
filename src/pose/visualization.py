"""
Video visualization module for pose overlay and annotation.

This module provides functions for drawing skeletons, annotating angles,
highlighting form issues, and creating annotated videos from pose detection results.
"""

import cv2
import numpy as np
from typing import Dict, List, Optional
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)

# MediaPipe Pose landmark indices
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26
LEFT_ANKLE = 27
RIGHT_ANKLE = 28

# Confidence threshold
CONFIDENCE_THRESHOLD = 0.5

# Colors (BGR format for OpenCV)
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)
COLOR_BLUE = (255, 0, 0)
COLOR_WHITE = (255, 255, 255)
COLOR_YELLOW = (0, 255, 255)
COLOR_BLACK = (0, 0, 0)

# Drawing parameters
LANDMARK_RADIUS = 5
LINE_THICKNESS = 2
ISSUE_CIRCLE_RADIUS = 15
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.5
FONT_THICKNESS = 1


def draw_skeleton(frame: np.ndarray, keypoints: np.ndarray) -> np.ndarray:
    """
    Draw skeleton connections and landmarks on a frame.
    
    Draws skeleton connections using cv2.line and circles at each landmark position.
    Uses green for normal confidence and red for low confidence (<0.5).
    
    Args:
        frame: BGR image frame as numpy array.
        keypoints: Numpy array of shape (33, 3) containing MediaPipe landmarks.
                  Each landmark is [x, y, confidence/visibility].
    
    Returns:
        Annotated frame with skeleton drawn.
    """
    if frame is None or keypoints is None or keypoints.shape != (33, 3):
        logger.warning("Invalid input for draw_skeleton")
        return frame
    
    annotated_frame = frame.copy()
    height, width = frame.shape[:2]
    
    # Convert normalized coordinates to pixel coordinates
    def to_pixel(landmark):
        """Convert normalized landmark to pixel coordinates."""
        if landmark[2] < CONFIDENCE_THRESHOLD:
            return None
        x = int(landmark[0] * width)
        y = int(landmark[1] * height)
        return (x, y)
    
    # Define skeleton connections
    connections = [
        # Shoulders
        (LEFT_SHOULDER, RIGHT_SHOULDER),
        # Hips
        (LEFT_HIP, RIGHT_HIP),
        # Left side: shoulder -> hip -> knee -> ankle
        (LEFT_SHOULDER, LEFT_HIP),
        (LEFT_HIP, LEFT_KNEE),
        (LEFT_KNEE, LEFT_ANKLE),
        # Right side: shoulder -> hip -> knee -> ankle
        (RIGHT_SHOULDER, RIGHT_HIP),
        (RIGHT_HIP, RIGHT_KNEE),
        (RIGHT_KNEE, RIGHT_ANKLE),
    ]
    
    # Draw connections
    for start_idx, end_idx in connections:
        start_landmark = keypoints[start_idx]
        end_landmark = keypoints[end_idx]
        
        # Check confidence for both points
        start_conf = start_landmark[2] >= CONFIDENCE_THRESHOLD
        end_conf = end_landmark[2] >= CONFIDENCE_THRESHOLD
        
        if start_conf and end_conf:
            start_pos = to_pixel(start_landmark)
            end_pos = to_pixel(end_landmark)
            
            if start_pos and end_pos:
                # Use green for normal confidence
                cv2.line(annotated_frame, start_pos, end_pos, COLOR_GREEN, LINE_THICKNESS)
        elif start_landmark[2] > 0 or end_landmark[2] > 0:
            # At least one point has some detection, but low confidence
            start_pos = to_pixel(start_landmark) if start_landmark[2] >= CONFIDENCE_THRESHOLD else None
            end_pos = to_pixel(end_landmark) if end_landmark[2] >= CONFIDENCE_THRESHOLD else None
            
            if start_pos and end_pos:
                # Use red for low confidence
                cv2.line(annotated_frame, start_pos, end_pos, COLOR_RED, LINE_THICKNESS)
    
    # Draw circles at each landmark position
    for i, landmark in enumerate(keypoints):
        if landmark[2] < CONFIDENCE_THRESHOLD:
            continue
        
        x = int(landmark[0] * width)
        y = int(landmark[1] * height)
        
        # Choose color based on confidence
        if landmark[2] >= CONFIDENCE_THRESHOLD:
            color = COLOR_GREEN
        else:
            color = COLOR_RED
        
        cv2.circle(annotated_frame, (x, y), LANDMARK_RADIUS, color, -1)
    
    return annotated_frame


def annotate_angles(frame: np.ndarray, keypoints: np.ndarray, angles: Dict[str, float]) -> np.ndarray:
    """
    Add text overlays showing key angles near joints.
    
    Formats angles as "Knee: 105°" and places them at joint positions.
    
    Args:
        frame: BGR image frame as numpy array.
        keypoints: Numpy array of shape (33, 3) containing MediaPipe landmarks.
        angles: Dictionary with joint angles, e.g.:
                {'left_knee_angle': 105.0, 'right_knee_angle': 110.0, ...}
    
    Returns:
        Annotated frame with angle text overlays.
    """
    if frame is None or keypoints is None or angles is None:
        logger.warning("Invalid input for annotate_angles")
        return frame
    
    annotated_frame = frame.copy()
    height, width = frame.shape[:2]
    
    # Convert normalized coordinates to pixel coordinates
    def to_pixel(landmark):
        """Convert normalized landmark to pixel coordinates."""
        if landmark[2] < CONFIDENCE_THRESHOLD:
            return None
        x = int(landmark[0] * width)
        y = int(landmark[1] * height)
        return (x, y)
    
    # Annotate left knee angle
    if 'left_knee_angle' in angles and keypoints[LEFT_KNEE][2] >= CONFIDENCE_THRESHOLD:
        knee_pos = to_pixel(keypoints[LEFT_KNEE])
        if knee_pos:
            text = f"L Knee: {angles['left_knee_angle']:.0f}°"
            # Position text above the knee
            text_pos = (knee_pos[0] - 40, knee_pos[1] - 20)
            cv2.putText(annotated_frame, text, text_pos, FONT, FONT_SCALE, COLOR_WHITE, FONT_THICKNESS)
            cv2.putText(annotated_frame, text, text_pos, FONT, FONT_SCALE, COLOR_BLACK, FONT_THICKNESS + 1)
    
    # Annotate right knee angle
    if 'right_knee_angle' in angles and keypoints[RIGHT_KNEE][2] >= CONFIDENCE_THRESHOLD:
        knee_pos = to_pixel(keypoints[RIGHT_KNEE])
        if knee_pos:
            text = f"R Knee: {angles['right_knee_angle']:.0f}°"
            # Position text above the knee
            text_pos = (knee_pos[0] - 40, knee_pos[1] - 20)
            cv2.putText(annotated_frame, text, text_pos, FONT, FONT_SCALE, COLOR_WHITE, FONT_THICKNESS)
            cv2.putText(annotated_frame, text, text_pos, FONT, FONT_SCALE, COLOR_BLACK, FONT_THICKNESS + 1)
    
    # Annotate left hip angle
    if 'left_hip_angle' in angles and keypoints[LEFT_HIP][2] >= CONFIDENCE_THRESHOLD:
        hip_pos = to_pixel(keypoints[LEFT_HIP])
        if hip_pos:
            text = f"L Hip: {angles['left_hip_angle']:.0f}°"
            # Position text to the left of the hip
            text_pos = (hip_pos[0] - 50, hip_pos[1])
            cv2.putText(annotated_frame, text, text_pos, FONT, FONT_SCALE, COLOR_WHITE, FONT_THICKNESS)
            cv2.putText(annotated_frame, text, text_pos, FONT, FONT_SCALE, COLOR_BLACK, FONT_THICKNESS + 1)
    
    # Annotate right hip angle
    if 'right_hip_angle' in angles and keypoints[RIGHT_HIP][2] >= CONFIDENCE_THRESHOLD:
        hip_pos = to_pixel(keypoints[RIGHT_HIP])
        if hip_pos:
            text = f"R Hip: {angles['right_hip_angle']:.0f}°"
            # Position text to the right of the hip
            text_pos = (hip_pos[0] + 10, hip_pos[1])
            cv2.putText(annotated_frame, text, text_pos, FONT, FONT_SCALE, COLOR_WHITE, FONT_THICKNESS)
            cv2.putText(annotated_frame, text, text_pos, FONT, FONT_SCALE, COLOR_BLACK, FONT_THICKNESS + 1)
    
    # Annotate spine angle
    if 'spine_angle' in angles:
        # Use midpoint between hips for spine angle position
        left_hip = keypoints[LEFT_HIP]
        right_hip = keypoints[RIGHT_HIP]
        if left_hip[2] >= CONFIDENCE_THRESHOLD and right_hip[2] >= CONFIDENCE_THRESHOLD:
            mid_x = int((left_hip[0] + right_hip[0]) / 2 * width)
            mid_y = int((left_hip[1] + right_hip[1]) / 2 * height)
            text = f"Spine: {angles['spine_angle']:.0f}°"
            # Position text above the hip midpoint
            text_pos = (mid_x - 40, mid_y - 40)
            cv2.putText(annotated_frame, text, text_pos, FONT, FONT_SCALE, COLOR_WHITE, FONT_THICKNESS)
            cv2.putText(annotated_frame, text, text_pos, FONT, FONT_SCALE, COLOR_BLACK, FONT_THICKNESS + 1)
    
    return annotated_frame


def highlight_issue(
    frame: np.ndarray,
    keypoints: np.ndarray,
    issue_type: str,
    side: str
) -> np.ndarray:
    """
    Highlight specific form issues on the frame.
    
    Draws visual indicators (circles, lines, text) to highlight detected issues.
    
    Args:
        frame: BGR image frame as numpy array.
        keypoints: Numpy array of shape (33, 3) containing MediaPipe landmarks.
        issue_type: Type of issue to highlight: "knee_valgus", "forward_lean", etc.
        side: Side affected: "left", "right", or "both".
    
    Returns:
        Annotated frame with issue highlighted.
    """
    if frame is None or keypoints is None:
        logger.warning("Invalid input for highlight_issue")
        return frame
    
    annotated_frame = frame.copy()
    height, width = frame.shape[:2]
    
    # Convert normalized coordinates to pixel coordinates
    def to_pixel(landmark):
        """Convert normalized landmark to pixel coordinates."""
        if landmark[2] < CONFIDENCE_THRESHOLD:
            return None
        x = int(landmark[0] * width)
        y = int(landmark[1] * height)
        return (x, y)
    
    if issue_type == "knee_valgus":
        # Draw red circle around affected knee
        if side == "left" and keypoints[LEFT_KNEE][2] >= CONFIDENCE_THRESHOLD:
            knee_pos = to_pixel(keypoints[LEFT_KNEE])
            if knee_pos:
                cv2.circle(annotated_frame, knee_pos, ISSUE_CIRCLE_RADIUS, COLOR_RED, 3)
                text = "Knee Valgus (L)"
                text_pos = (knee_pos[0] - 60, knee_pos[1] - ISSUE_CIRCLE_RADIUS - 10)
                cv2.putText(annotated_frame, text, text_pos, FONT, FONT_SCALE + 0.1, COLOR_RED, FONT_THICKNESS + 1)
        
        elif side == "right" and keypoints[RIGHT_KNEE][2] >= CONFIDENCE_THRESHOLD:
            knee_pos = to_pixel(keypoints[RIGHT_KNEE])
            if knee_pos:
                cv2.circle(annotated_frame, knee_pos, ISSUE_CIRCLE_RADIUS, COLOR_RED, 3)
                text = "Knee Valgus (R)"
                text_pos = (knee_pos[0] - 60, knee_pos[1] - ISSUE_CIRCLE_RADIUS - 10)
                cv2.putText(annotated_frame, text, text_pos, FONT, FONT_SCALE + 0.1, COLOR_RED, FONT_THICKNESS + 1)
        
        elif side == "both":
            # Highlight both knees
            if keypoints[LEFT_KNEE][2] >= CONFIDENCE_THRESHOLD:
                knee_pos = to_pixel(keypoints[LEFT_KNEE])
                if knee_pos:
                    cv2.circle(annotated_frame, knee_pos, ISSUE_CIRCLE_RADIUS, COLOR_RED, 3)
            if keypoints[RIGHT_KNEE][2] >= CONFIDENCE_THRESHOLD:
                knee_pos = to_pixel(keypoints[RIGHT_KNEE])
                if knee_pos:
                    cv2.circle(annotated_frame, knee_pos, ISSUE_CIRCLE_RADIUS, COLOR_RED, 3)
            text = "Knee Valgus (Both)"
            text_pos = (width // 2 - 80, 30)
            cv2.putText(annotated_frame, text, text_pos, FONT, FONT_SCALE + 0.1, COLOR_RED, FONT_THICKNESS + 1)
    
    elif issue_type == "forward_lean":
        # Highlight spine in red (draw line from hip to shoulder)
        left_hip = keypoints[LEFT_HIP]
        right_hip = keypoints[RIGHT_HIP]
        left_shoulder = keypoints[LEFT_SHOULDER]
        right_shoulder = keypoints[RIGHT_SHOULDER]
        
        if (left_hip[2] >= CONFIDENCE_THRESHOLD and right_hip[2] >= CONFIDENCE_THRESHOLD and
            left_shoulder[2] >= CONFIDENCE_THRESHOLD and right_shoulder[2] >= CONFIDENCE_THRESHOLD):
            
            hip_mid = to_pixel(np.array([
                (left_hip[0] + right_hip[0]) / 2,
                (left_hip[1] + right_hip[1]) / 2,
                1.0
            ]))
            shoulder_mid = to_pixel(np.array([
                (left_shoulder[0] + right_shoulder[0]) / 2,
                (left_shoulder[1] + right_shoulder[1]) / 2,
                1.0
            ]))
            
            if hip_mid and shoulder_mid:
                # Draw red line for spine
                cv2.line(annotated_frame, hip_mid, shoulder_mid, COLOR_RED, 4)
                text = "Forward Lean"
                text_pos = (shoulder_mid[0] - 50, shoulder_mid[1] - 20)
                cv2.putText(annotated_frame, text, text_pos, FONT, FONT_SCALE + 0.1, COLOR_RED, FONT_THICKNESS + 1)
    
    elif issue_type == "asymmetry":
        # Highlight both sides in yellow to indicate asymmetry
        text = "Asymmetry Detected"
        text_pos = (width // 2 - 80, 30)
        cv2.putText(annotated_frame, text, text_pos, FONT, FONT_SCALE + 0.1, COLOR_YELLOW, FONT_THICKNESS + 1)
        
        # Draw yellow circles on both knees
        if keypoints[LEFT_KNEE][2] >= CONFIDENCE_THRESHOLD:
            knee_pos = to_pixel(keypoints[LEFT_KNEE])
            if knee_pos:
                cv2.circle(annotated_frame, knee_pos, ISSUE_CIRCLE_RADIUS, COLOR_YELLOW, 3)
        if keypoints[RIGHT_KNEE][2] >= CONFIDENCE_THRESHOLD:
            knee_pos = to_pixel(keypoints[RIGHT_KNEE])
            if knee_pos:
                cv2.circle(annotated_frame, knee_pos, ISSUE_CIRCLE_RADIUS, COLOR_YELLOW, 3)
    
    elif issue_type == "limited_depth":
        # Highlight both knees to indicate limited depth
        text = "Limited Depth"
        text_pos = (width // 2 - 60, 50)
        cv2.putText(annotated_frame, text, text_pos, FONT, FONT_SCALE + 0.1, COLOR_YELLOW, FONT_THICKNESS + 1)
    
    return annotated_frame


def create_annotated_video(
    video_path: str,
    keypoints_sequence: np.ndarray,
    issues: List[Dict],
    output_path: str,
    fps: Optional[int] = None
) -> str:
    """
    Create an annotated video with skeleton, angles, and issue highlights.
    
    Loads the original video and processes each frame to add:
    - Skeleton overlay
    - Angle annotations
    - Issue highlights (if they occur in that frame)
    
    Args:
        video_path: Path to input video file.
        keypoints_sequence: Numpy array of shape (num_frames, 33, 3) containing
                          MediaPipe landmarks for each frame.
        issues: List of issue dictionaries from analyze_squat_form.
        output_path: Path where annotated video will be saved.
        fps: Optional FPS override. If None, uses original video FPS.
    
    Returns:
        Path to the created annotated video file.
    
    Raises:
        ValueError: If video cannot be opened or has invalid properties.
    """
    logger.info(f"Creating annotated video from {video_path}")
    
    # Open input video
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        error_msg = f"Failed to open video file: {video_path}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    # Get video properties
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if fps is None:
        fps = int(original_fps)
    
    logger.info(f"Video properties: {width}x{height}, {fps} fps, {total_frames} frames")
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        cap.release()
        error_msg = f"Failed to create output video: {output_path}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    # Import extract_joint_angles for angle calculation
    from .biomechanics import extract_joint_angles
    
    # Create frame-to-issues mapping for efficient lookup
    frame_issues = {}
    for issue in issues:
        for frame_idx in issue.get('frames', []):
            if frame_idx not in frame_issues:
                frame_issues[frame_idx] = []
            frame_issues[frame_idx].append(issue)
    
    num_keypoints_frames = len(keypoints_sequence)
    frame_count = 0
    
    try:
        # Process frames with progress bar
        with tqdm(total=min(total_frames, num_keypoints_frames), desc="Annotating video") as pbar:
            while frame_count < min(total_frames, num_keypoints_frames):
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                # Get keypoints for this frame
                if frame_count < num_keypoints_frames:
                    keypoints = keypoints_sequence[frame_count]
                else:
                    # If we run out of keypoints, use last frame's keypoints
                    keypoints = keypoints_sequence[-1]
                
                # Draw skeleton
                annotated_frame = draw_skeleton(frame, keypoints)
                
                # Extract and annotate angles
                angles = extract_joint_angles(keypoints)
                if angles:
                    annotated_frame = annotate_angles(annotated_frame, keypoints, angles)
                
                # Highlight issues for this frame
                if frame_count in frame_issues:
                    for issue in frame_issues[frame_count]:
                        annotated_frame = highlight_issue(
                            annotated_frame,
                            keypoints,
                            issue.get('type', ''),
                            issue.get('side', 'both')
                        )
                
                # Write frame to output video
                out.write(annotated_frame)
                
                frame_count += 1
                pbar.update(1)
        
        cap.release()
        out.release()
        
        logger.info(f"Successfully created annotated video: {output_path}")
        return output_path
        
    except Exception as e:
        cap.release()
        out.release()
        error_msg = f"Error creating annotated video: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise

