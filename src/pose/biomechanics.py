"""
Biomechanics analysis module for exercise form evaluation.

This module provides functions for analyzing human movement patterns,
calculating joint angles, and detecting form issues in exercises like squats.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
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

# Confidence threshold for landmark detection
CONFIDENCE_THRESHOLD = 0.5


def calculate_angle(
    point1: np.ndarray,
    point2: np.ndarray,
    point3: np.ndarray
) -> float:
    """
    Calculate the angle at point2 between vectors (point1->point2) and (point3->point2).
    
    Uses vector math: angle = arccos(dot(v1, v2) / (norm(v1) * norm(v2)))
    
    Args:
        point1: First point as numpy array [x, y] or [x, y, confidence]
        point2: Vertex point (angle is calculated here) as numpy array
        point3: Third point as numpy array
    
    Returns:
        Angle in degrees (0-180). Returns 0.0 if vectors are invalid (zero length).
    
    Raises:
        ValueError: If input points are invalid or have wrong dimensions.
    """
    try:
        # Extract x, y coordinates (ignore confidence if present)
        p1 = np.array([point1[0], point1[1]])
        p2 = np.array([point2[0], point2[1]])
        p3 = np.array([point3[0], point3[1]])
        
        # Calculate vectors from point2
        v1 = p1 - p2  # Vector from point2 to point1
        v2 = p3 - p2  # Vector from point2 to point3
        
        # Calculate magnitudes
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        
        # Handle edge cases: zero vectors
        if norm_v1 < 1e-6 or norm_v2 < 1e-6:
            logger.warning("Zero vector detected in angle calculation")
            return 0.0
        
        # Calculate dot product
        dot_product = np.dot(v1, v2)
        
        # Calculate angle in radians, then convert to degrees
        cos_angle = dot_product / (norm_v1 * norm_v2)
        
        # Clamp to [-1, 1] to avoid numerical errors
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        
        angle_rad = np.arccos(cos_angle)
        angle_deg = np.degrees(angle_rad)
        
        return angle_deg
        
    except (IndexError, ValueError, TypeError) as e:
        logger.error(f"Error calculating angle: {e}")
        raise ValueError(f"Invalid points for angle calculation: {e}")


def extract_joint_angles(keypoints: np.ndarray) -> Optional[Dict[str, float]]:
    """
    Extract joint angles from MediaPipe pose landmarks for a single frame.
    
    Calculates knee angles, hip angles, and spine angle from MediaPipe landmarks.
    Returns None if landmarks have low confidence (<0.5).
    
    Args:
        keypoints: Numpy array of shape (33, 3) containing MediaPipe landmarks.
                  Each landmark is [x, y, confidence/visibility].
    
    Returns:
        Dictionary with joint angles in degrees:
        {
            'left_knee_angle': float,
            'right_knee_angle': float,
            'left_hip_angle': float,
            'right_hip_angle': float,
            'spine_angle': float
        }
        Returns None if confidence is too low or landmarks are missing.
    """
    if keypoints is None or keypoints.shape != (33, 3):
        logger.warning("Invalid keypoints shape for joint angle extraction")
        return None
    
    try:
        # Extract required landmarks
        landmarks = {
            'left_shoulder': keypoints[LEFT_SHOULDER],
            'right_shoulder': keypoints[RIGHT_SHOULDER],
            'left_hip': keypoints[LEFT_HIP],
            'right_hip': keypoints[RIGHT_HIP],
            'left_knee': keypoints[LEFT_KNEE],
            'right_knee': keypoints[RIGHT_KNEE],
            'left_ankle': keypoints[LEFT_ANKLE],
            'right_ankle': keypoints[RIGHT_ANKLE],
        }
        
        # Check confidence for all required landmarks
        for name, landmark in landmarks.items():
            if landmark[2] < CONFIDENCE_THRESHOLD:
                logger.debug(f"Low confidence for {name}: {landmark[2]}")
                return None
        
        angles = {}
        
        # Calculate left knee angle (hip-knee-ankle)
        angles['left_knee_angle'] = calculate_angle(
            landmarks['left_hip'],
            landmarks['left_knee'],
            landmarks['left_ankle']
        )
        
        # Calculate right knee angle (hip-knee-ankle)
        angles['right_knee_angle'] = calculate_angle(
            landmarks['right_hip'],
            landmarks['right_knee'],
            landmarks['right_ankle']
        )
        
        # Calculate left hip angle (shoulder-hip-knee)
        angles['left_hip_angle'] = calculate_angle(
            landmarks['left_shoulder'],
            landmarks['left_hip'],
            landmarks['left_knee']
        )
        
        # Calculate right hip angle (shoulder-hip-knee)
        angles['right_hip_angle'] = calculate_angle(
            landmarks['right_shoulder'],
            landmarks['right_hip'],
            landmarks['right_knee']
        )
        
        # Calculate spine angle (torso from vertical)
        # Use average of left and right shoulder-hip vectors
        # Vertical line is (hip_x, hip_y) -> (hip_x, hip_y - 1)
        left_hip = landmarks['left_hip']
        right_hip = landmarks['right_hip']
        avg_hip = np.array([(left_hip[0] + right_hip[0]) / 2, (left_hip[1] + right_hip[1]) / 2])
        
        left_shoulder = landmarks['left_shoulder']
        right_shoulder = landmarks['right_shoulder']
        avg_shoulder = np.array([(left_shoulder[0] + right_shoulder[0]) / 2, 
                                 (left_shoulder[1] + right_shoulder[1]) / 2])
        
        # Vertical line point above hip
        vertical_point = np.array([avg_hip[0], avg_hip[1] - 1.0])
        
        # Calculate angle between vertical and shoulder-hip line
        angles['spine_angle'] = calculate_angle(
            vertical_point,
            avg_hip,
            avg_shoulder
        )
        
        return angles
        
    except (IndexError, KeyError) as e:
        logger.error(f"Error extracting joint angles: {e}")
        return None


def analyze_squat_form(
    keypoints_sequence: np.ndarray,
    fps: int = 30
) -> Dict:
    """
    Analyze squat form from a sequence of pose keypoints.
    
    Detects form issues including knee valgus, forward lean, asymmetry,
    and limited depth. Returns comprehensive analysis with severity scores.
    
    Args:
        keypoints_sequence: Numpy array of shape (num_frames, 33, 3) containing
                           MediaPipe landmarks for each frame.
        fps: Frames per second of the video. Defaults to 30.
    
    Returns:
        Dictionary containing analysis results:
        {
            "exercise": "squat",
            "duration_sec": float,
            "issues": [
                {
                    "type": str,
                    "severity": str,  # "mild", "moderate", "severe"
                    "side": str,  # "left", "right", or "both"
                    "magnitude_degrees": float,
                    "frames": List[int],
                    "timestamps_sec": List[float],
                    "description": str
                }
            ],
            "metrics": {
                "knee_flexion_rom": {"left": float, "right": float},
                "hip_flexion_rom": {"left": float, "right": float}
            }
        }
    """
    if keypoints_sequence is None or len(keypoints_sequence) == 0:
        logger.warning("Empty keypoints sequence provided")
        return {
            "exercise": "squat",
            "duration_sec": 0.0,
            "issues": [],
            "metrics": {}
        }
    
    num_frames = len(keypoints_sequence)
    duration_sec = num_frames / fps
    
    # Extract joint angles for all frames
    all_angles = []
    valid_frames = []
    
    for frame_idx in range(num_frames):
        angles = extract_joint_angles(keypoints_sequence[frame_idx])
        if angles is not None:
            all_angles.append(angles)
            valid_frames.append(frame_idx)
        else:
            all_angles.append(None)
    
    if len(valid_frames) == 0:
        logger.warning("No valid frames with sufficient confidence")
        return {
            "exercise": "squat",
            "duration_sec": duration_sec,
            "issues": [],
            "metrics": {}
        }
    
    issues = []
    
    # Track knee and hip angles for ROM calculation
    left_knee_angles = []
    right_knee_angles = []
    left_hip_angles = []
    right_hip_angles = []
    spine_angles = []
    
    # Track knee valgus (knee collapses inward)
    left_knee_valgus_frames = []
    right_knee_valgus_frames = []
    
    # Track forward lean
    forward_lean_frames = []
    
    # Track asymmetry
    knee_asymmetry_frames = []
    hip_asymmetry_frames = []
    
    # Process each valid frame
    for frame_idx in valid_frames:
        angles = all_angles[frame_idx]
        if angles is None:
            continue
        
        timestamp = frame_idx / fps
        
        # Collect angles for ROM calculation
        left_knee_angles.append(angles['left_knee_angle'])
        right_knee_angles.append(angles['right_knee_angle'])
        left_hip_angles.append(angles['left_hip_angle'])
        right_hip_angles.append(angles['right_hip_angle'])
        spine_angles.append(angles['spine_angle'])
        
        # Get keypoints for this frame
        keypoints = keypoints_sequence[frame_idx]
        
        # 1. Detect knee valgus (knee x-position < midpoint(hip_x, ankle_x) by >0.02)
        # For left side: if left_knee_x < midpoint(left_hip_x, left_ankle_x) - 0.02
        left_hip_x = keypoints[LEFT_HIP][0]
        left_ankle_x = keypoints[LEFT_ANKLE][0]
        left_knee_x = keypoints[LEFT_KNEE][0]
        left_midpoint = (left_hip_x + left_ankle_x) / 2
        left_valgus_offset = left_midpoint - left_knee_x
        
        if left_valgus_offset > 0.02:
            left_knee_valgus_frames.append({
                'frame': frame_idx,
                'timestamp': timestamp,
                'offset': left_valgus_offset,
                'angle': angles['left_knee_angle']
            })
        
        # For right side: if right_knee_x > midpoint(right_hip_x, right_ankle_x) + 0.02
        right_hip_x = keypoints[RIGHT_HIP][0]
        right_ankle_x = keypoints[RIGHT_ANKLE][0]
        right_knee_x = keypoints[RIGHT_KNEE][0]
        right_midpoint = (right_hip_x + right_ankle_x) / 2
        right_valgus_offset = right_knee_x - right_midpoint
        
        if right_valgus_offset > 0.02:
            right_knee_valgus_frames.append({
                'frame': frame_idx,
                'timestamp': timestamp,
                'offset': right_valgus_offset,
                'angle': angles['right_knee_angle']
            })
        
        # 2. Detect forward lean (spine_angle > 60 degrees)
        if angles['spine_angle'] > 60:
            forward_lean_frames.append({
                'frame': frame_idx,
                'timestamp': timestamp,
                'angle': angles['spine_angle']
            })
        
        # 3. Detect asymmetry (abs(left_angle - right_angle) > 10 degrees)
        knee_diff = abs(angles['left_knee_angle'] - angles['right_knee_angle'])
        if knee_diff > 10:
            knee_asymmetry_frames.append({
                'frame': frame_idx,
                'timestamp': timestamp,
                'difference': knee_diff
            })
        
        hip_diff = abs(angles['left_hip_angle'] - angles['right_hip_angle'])
        if hip_diff > 10:
            hip_asymmetry_frames.append({
                'frame': frame_idx,
                'timestamp': timestamp,
                'difference': hip_diff
            })
    
    # Create issue reports
    
    # Knee valgus - Left
    if left_knee_valgus_frames:
        avg_offset = np.mean([f['offset'] for f in left_knee_valgus_frames])
        # Convert offset to approximate degrees (rough conversion: 0.02 ≈ 10 degrees)
        magnitude_deg = (avg_offset / 0.02) * 10
        
        if magnitude_deg < 15:
            severity = "mild"
        elif magnitude_deg < 25:
            severity = "moderate"
        else:
            severity = "severe"
        
        issues.append({
            "type": "knee_valgus",
            "severity": severity,
            "side": "left",
            "magnitude_degrees": round(magnitude_deg, 1),
            "frames": [f['frame'] for f in left_knee_valgus_frames],
            "timestamps_sec": [round(f['timestamp'], 2) for f in left_knee_valgus_frames],
            "description": f"Left knee collapses inward during squat. "
                          f"Average deviation: {magnitude_deg:.1f} degrees."
        })
    
    # Knee valgus - Right
    if right_knee_valgus_frames:
        avg_offset = np.mean([f['offset'] for f in right_knee_valgus_frames])
        magnitude_deg = (avg_offset / 0.02) * 10
        
        if magnitude_deg < 15:
            severity = "mild"
        elif magnitude_deg < 25:
            severity = "moderate"
        else:
            severity = "severe"
        
        issues.append({
            "type": "knee_valgus",
            "severity": severity,
            "side": "right",
            "magnitude_degrees": round(magnitude_deg, 1),
            "frames": [f['frame'] for f in right_knee_valgus_frames],
            "timestamps_sec": [round(f['timestamp'], 2) for f in right_knee_valgus_frames],
            "description": f"Right knee collapses inward during squat. "
                          f"Average deviation: {magnitude_deg:.1f} degrees."
        })
    
    # Forward lean
    if forward_lean_frames:
        avg_angle = np.mean([f['angle'] for f in forward_lean_frames])
        
        if avg_angle < 70:
            severity = "mild"
        elif avg_angle < 80:
            severity = "moderate"
        else:
            severity = "severe"
        
        issues.append({
            "type": "forward_lean",
            "severity": severity,
            "side": "both",
            "magnitude_degrees": round(avg_angle, 1),
            "frames": [f['frame'] for f in forward_lean_frames],
            "timestamps_sec": [round(f['timestamp'], 2) for f in forward_lean_frames],
            "description": f"Excessive forward lean detected. "
                          f"Average spine angle: {avg_angle:.1f} degrees (target: <60°)."
        })
    
    # Knee asymmetry
    if knee_asymmetry_frames:
        avg_diff = np.mean([f['difference'] for f in knee_asymmetry_frames])
        
        if avg_diff < 15:
            severity = "mild"
        elif avg_diff < 20:
            severity = "moderate"
        else:
            severity = "severe"
        
        issues.append({
            "type": "asymmetry",
            "severity": severity,
            "side": "both",
            "magnitude_degrees": round(avg_diff, 1),
            "frames": [f['frame'] for f in knee_asymmetry_frames],
            "timestamps_sec": [round(f['timestamp'], 2) for f in knee_asymmetry_frames],
            "description": f"Knee angle asymmetry detected. "
                          f"Average difference: {avg_diff:.1f} degrees between left and right."
        })
    
    # Hip asymmetry
    if hip_asymmetry_frames:
        avg_diff = np.mean([f['difference'] for f in hip_asymmetry_frames])
        
        if avg_diff < 15:
            severity = "mild"
        elif avg_diff < 20:
            severity = "moderate"
        else:
            severity = "severe"
        
        issues.append({
            "type": "asymmetry",
            "severity": severity,
            "side": "both",
            "magnitude_degrees": round(avg_diff, 1),
            "frames": [f['frame'] for f in hip_asymmetry_frames],
            "timestamps_sec": [round(f['timestamp'], 2) for f in hip_asymmetry_frames],
            "description": f"Hip angle asymmetry detected. "
                          f"Average difference: {avg_diff:.1f} degrees between left and right."
        })
    
    # Limited depth: if max(knee_angle) < 90 degrees
    # Note: This checks if the maximum knee angle in the sequence is < 90
    # For proper depth, knee should flex below 90 degrees (min angle < 90)
    if left_knee_angles and right_knee_angles:
        max_left_knee = max(left_knee_angles)
        max_right_knee = max(right_knee_angles)
        
        # Check if max knee angle < 90 (as per requirement)
        # This would indicate the knee never reached a flexed position
        if max_left_knee < 90 or max_right_knee < 90:
            # Use the smaller of the two max angles
            min_max_angle = min(max_left_knee, max_right_knee)
            
            if min_max_angle > 80:
                severity = "mild"
            elif min_max_angle > 70:
                severity = "moderate"
            else:
                severity = "severe"
            
            issues.append({
                "type": "limited_depth",
                "severity": severity,
                "side": "both",
                "magnitude_degrees": round(min_max_angle, 1),
                "frames": [],
                "timestamps_sec": [],
                "description": f"Insufficient squat depth detected. Maximum knee angle: {min_max_angle:.1f}° "
                              f"(target: knee should flex below 90° for full depth)."
            })
    
    # Calculate Range of Motion (ROM) metrics
    metrics = {}
    
    if left_knee_angles and right_knee_angles:
        # ROM = max angle - min angle (larger ROM = better)
        left_knee_rom = max(left_knee_angles) - min(left_knee_angles)
        right_knee_rom = max(right_knee_angles) - min(right_knee_angles)
        
        metrics["knee_flexion_rom"] = {
            "left": round(left_knee_rom, 1),
            "right": round(right_knee_rom, 1)
        }
    
    if left_hip_angles and right_hip_angles:
        left_hip_rom = max(left_hip_angles) - min(left_hip_angles)
        right_hip_rom = max(right_hip_angles) - min(right_hip_angles)
        
        metrics["hip_flexion_rom"] = {
            "left": round(left_hip_rom, 1),
            "right": round(right_hip_rom, 1)
        }
    
    return {
        "exercise": "squat",
        "duration_sec": round(duration_sec, 2),
        "issues": issues,
        "metrics": metrics
    }

