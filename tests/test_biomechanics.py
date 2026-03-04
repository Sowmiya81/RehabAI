# -*- coding: utf-8 -*-
"""
Tests for biomechanics analysis functions.

Tests all exercise analysis functions (squat, pull-up, push-up, situp)
and the dispatcher with synthetic keypoint data.
"""
import numpy as np
import importlib.util
from pathlib import Path

# Load biomechanics module directly to avoid cv2 dependency from detector.py
biomech_path = Path(__file__).parent.parent / "src" / "pose" / "biomechanics.py"
spec = importlib.util.spec_from_file_location("biomechanics", str(biomech_path))
biomechanics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(biomechanics)

calculate_angle = biomechanics.calculate_angle
extract_joint_angles = biomechanics.extract_joint_angles
analyze_squat_form = biomechanics.analyze_squat_form
analyze_pullup_form = biomechanics.analyze_pullup_form
analyze_pushup_form = biomechanics.analyze_pushup_form
analyze_situp_form = biomechanics.analyze_situp_form
analyze_exercise_form = biomechanics.analyze_exercise_form

# -----------------------------------------------------------
# Test angle calculation
# -----------------------------------------------------------
p1 = np.array([0, 1, 0])
p2 = np.array([0, 0, 0])
p3 = np.array([1, 0, 0])
angle = calculate_angle(p1, p2, p3)
print(f"Angle (should be 90): {angle:.1f} degrees")
assert abs(angle - 90.0) < 1.0, f"Expected ~90, got {angle}"

# -----------------------------------------------------------
# Test extract_joint_angles (now includes elbow angles)
# -----------------------------------------------------------
keypoints = np.random.rand(33, 3).astype(np.float32)
keypoints[:, 2] = 0.9  # Set confidence high
angles = extract_joint_angles(keypoints)
assert angles is not None, "Expected angles dict, got None"
expected_keys = {
    'left_knee_angle', 'right_knee_angle',
    'left_hip_angle', 'right_hip_angle',
    'left_elbow_angle', 'right_elbow_angle',
    'spine_angle',
}
assert expected_keys.issubset(angles.keys()), f"Missing keys: {expected_keys - angles.keys()}"
print(f"PASS - Extracted angles: {sorted(angles.keys())}")

# -----------------------------------------------------------
# Helper: create synthetic keypoint sequences
# -----------------------------------------------------------
def make_keypoints_sequence(n_frames=60, seed=42):
    """Generate a synthetic (n_frames, 33, 3) array with high confidence."""
    rng = np.random.RandomState(seed)
    seq = rng.rand(n_frames, 33, 3).astype(np.float32)
    seq[:, :, 2] = 0.9  # high confidence
    return seq


# -----------------------------------------------------------
# Test analyze_squat_form
# -----------------------------------------------------------
seq = make_keypoints_sequence()
result = analyze_squat_form(seq, fps=30)
assert result['exercise'] == 'squat'
assert 'duration_sec' in result
assert isinstance(result['issues'], list)
assert isinstance(result['metrics'], dict)
print(f"PASS - analyze_squat_form: {len(result['issues'])} issues, metrics={list(result['metrics'].keys())}")

# -----------------------------------------------------------
# Test analyze_pullup_form
# -----------------------------------------------------------
result = analyze_pullup_form(seq, fps=30)
assert result['exercise'] == 'pull-up'
assert 'duration_sec' in result
assert isinstance(result['issues'], list)
assert isinstance(result['metrics'], dict)
print(f"PASS - analyze_pullup_form: {len(result['issues'])} issues, metrics={list(result['metrics'].keys())}")

# -----------------------------------------------------------
# Test analyze_pushup_form
# -----------------------------------------------------------
result = analyze_pushup_form(seq, fps=30)
assert result['exercise'] == 'push-up'
assert 'duration_sec' in result
assert isinstance(result['issues'], list)
assert isinstance(result['metrics'], dict)
print(f"PASS - analyze_pushup_form: {len(result['issues'])} issues, metrics={list(result['metrics'].keys())}")

# -----------------------------------------------------------
# Test analyze_situp_form
# -----------------------------------------------------------
result = analyze_situp_form(seq, fps=30)
assert result['exercise'] == 'situp'
assert 'duration_sec' in result
assert isinstance(result['issues'], list)
assert isinstance(result['metrics'], dict)
print(f"PASS - analyze_situp_form: {len(result['issues'])} issues, metrics={list(result['metrics'].keys())}")

# -----------------------------------------------------------
# Test analyze_exercise_form dispatcher
# -----------------------------------------------------------
for ex_type in ['squat', 'pull-up', 'push-up', 'situp']:
    result = analyze_exercise_form(seq, exercise_type=ex_type, fps=30)
    assert result['exercise'] == ex_type, f"Expected {ex_type}, got {result['exercise']}"
print("PASS - analyze_exercise_form dispatcher works for all exercise types")

# -----------------------------------------------------------
# Test edge cases: empty / None inputs
# -----------------------------------------------------------
for fn, name in [
    (analyze_squat_form, 'squat'),
    (analyze_pullup_form, 'pull-up'),
    (analyze_pushup_form, 'push-up'),
    (analyze_situp_form, 'situp'),
]:
    empty = fn(np.array([]).reshape(0, 33, 3))
    assert empty['issues'] == []
    none_result = fn(None)
    assert none_result['issues'] == []
print("PASS - All analysis functions handle empty/None input gracefully")

print("\nAll tests passed!")
