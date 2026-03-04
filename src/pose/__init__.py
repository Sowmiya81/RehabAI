"""Pose detection module."""

from .detector import PoseDetector
from .biomechanics import (
    calculate_angle,
    extract_joint_angles,
    analyze_squat_form,
    analyze_pullup_form,
    analyze_pushup_form,
    analyze_situp_form,
    analyze_exercise_form
)
from .visualization import (
    draw_skeleton,
    annotate_angles,
    highlight_issue,
    create_annotated_video
)

__all__ = [
    "PoseDetector",
    "calculate_angle",
    "extract_joint_angles",
    "analyze_squat_form",
    "analyze_pullup_form",
    "analyze_pushup_form",
    "analyze_situp_form",
    "analyze_exercise_form",
    "draw_skeleton",
    "annotate_angles",
    "highlight_issue",
    "create_annotated_video"
]

