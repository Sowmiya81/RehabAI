# Create test file: tests/test_detector.py
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.pose.detector import PoseDetector

detector = PoseDetector()
print("✅ PoseDetector initialized successfully!")