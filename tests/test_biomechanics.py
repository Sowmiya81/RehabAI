# tests/test_analyzer.py
import numpy as np
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.pose.biomechanics import calculate_angle, extract_joint_angles

# Test angle calculation
p1 = np.array([0, 1, 0])
p2 = np.array([0, 0, 0])
p3 = np.array([1, 0, 0])
angle = calculate_angle(p1, p2, p3)
print(f"Angle (should be 90): {angle}°")

# Test with synthetic keypoints
keypoints = np.random.rand(33, 3)
keypoints[:, 2] = 0.9  # Set confidence high
angles = extract_joint_angles(keypoints)
print(f"✅ Extracted angles: {angles.keys()}")
