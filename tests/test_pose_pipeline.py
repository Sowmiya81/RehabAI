"""
Test script for the complete pose detection pipeline.

This script demonstrates the full workflow:
1. Pose detection from video
2. Biomechanics analysis
3. Video annotation with issues highlighted
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.pose import PoseDetector, analyze_squat_form, create_annotated_video
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def print_issues(analysis: dict) -> None:
    """
    Print detected issues in a readable format.
    
    Args:
        analysis: Analysis dictionary from analyze_squat_form
    """
    print("\n" + "=" * 70)
    print("SQUAT FORM ANALYSIS RESULTS")
    print("=" * 70)
    print(f"Exercise: {analysis.get('exercise', 'unknown').upper()}")
    print(f"Duration: {analysis.get('duration_sec', 0):.2f} seconds")
    
    # Print metrics
    metrics = analysis.get('metrics', {})
    if metrics:
        print("\n📊 Range of Motion (ROM):")
        if 'knee_flexion_rom' in metrics:
            knee_rom = metrics['knee_flexion_rom']
            print(f"  - Knee Flexion: Left={knee_rom.get('left', 0):.1f}°, Right={knee_rom.get('right', 0):.1f}°")
        if 'hip_flexion_rom' in metrics:
            hip_rom = metrics['hip_flexion_rom']
            print(f"  - Hip Flexion: Left={hip_rom.get('left', 0):.1f}°, Right={hip_rom.get('right', 0):.1f}°")
    
    # Print issues
    issues = analysis.get('issues', [])
    if not issues:
        print("\n✅ No form issues detected! Great form!")
    else:
        print(f"\n⚠️  Detected {len(issues)} form issue(s):")
        print("-" * 70)
        
        for i, issue in enumerate(issues, 1):
            issue_type = issue.get('type', 'unknown')
            severity = issue.get('severity', 'unknown')
            side = issue.get('side', 'both')
            magnitude = issue.get('magnitude_degrees', 0)
            description = issue.get('description', 'No description')
            frames = issue.get('frames', [])
            timestamps = issue.get('timestamps_sec', [])
            
            # Severity emoji
            severity_emoji = {
                'mild': '🟡',
                'moderate': '🟠',
                'severe': '🔴'
            }.get(severity, '⚪')
            
            print(f"\n{i}. {severity_emoji} {issue_type.replace('_', ' ').title()} ({severity.upper()})")
            print(f"   Side: {side.title()}")
            print(f"   Magnitude: {magnitude:.1f}°")
            print(f"   Description: {description}")
            
            if frames:
                print(f"   Occurrences: {len(frames)} frame(s)")
                if timestamps:
                    print(f"   Time range: {min(timestamps):.2f}s - {max(timestamps):.2f}s")
    
    print("\n" + "=" * 70)


def main():
    """Main function to run the pose detection pipeline."""
    print("=" * 70)
    print("POSE DETECTION PIPELINE TEST")
    print("=" * 70)
    
    # Define paths
    project_root = Path(__file__).parent.parent
    test_video_path = project_root / "data" / "videos" / "test" / "my_squat.mp4"
    output_dir = project_root / "outputs" / "annotated_videos"
    output_video_path = output_dir / "test_output1.mp4"
    
    # Check if test video exists
    if not test_video_path.exists():
        print(f"\n❌ Test video not found at: {test_video_path}")
        print("\n📝 Instructions:")
        print("   1. Place a test video file named 'squat_18.mp4' in the following directory:")
        print(f"      {test_video_path.parent}")
        print("   2. The video should contain a person performing squats")
        print("   3. Supported formats: .mp4, .avi, .mov")
        print("\n   Example command to add a video:")
        print(f"   cp /path/to/your/video.mp4 {test_video_path}")
        return 1
    
    print(f"\n✅ Test video found: {test_video_path}")
    
    try:
        # Step 1: Initialize PoseDetector
        print("\n[1/5] Initializing PoseDetector...")
        detector = PoseDetector()
        print("   ✅ PoseDetector initialized successfully")
        
        # Step 2: Process video to get keypoints
        print("\n[2/5] Processing video to extract pose landmarks...")
        print("   This may take a while depending on video length...")
        keypoints = detector.process_video(str(test_video_path))
        
        if keypoints is None or len(keypoints) == 0:
            print("   ❌ No pose landmarks detected in video")
            return 1
        
        num_frames = len(keypoints)
        print(f"   ✅ Successfully processed {num_frames} frames")
        print(f"   Keypoints shape: {keypoints.shape}")
        
        # Step 3: Analyze squat form
        print("\n[3/5] Analyzing squat form...")
        analysis = analyze_squat_form(keypoints, fps=30)
        print("   ✅ Analysis complete")
        
        # Step 4: Print detected issues
        print("\n[4/5] Displaying analysis results...")
        print_issues(analysis)
        
        # Step 5: Create annotated video
        print("\n[5/5] Creating annotated video...")
        print("   This may take a while...")
        
        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)
        
        issues_list = analysis.get('issues', [])
        output_path = create_annotated_video(
            str(test_video_path),
            keypoints,
            issues_list,
            str(output_video_path),
            fps=30
        )
        
        print(f"\n✅ Successfully created annotated video!")
        print(f"   Output path: {output_path}")
        print(f"   File size: {Path(output_path).stat().st_size / (1024*1024):.2f} MB")
        
        print("\n" + "=" * 70)
        print("🎉 Pipeline completed successfully!")
        print("=" * 70)
        
        return 0
        
    except FileNotFoundError as e:
        print(f"\n❌ File not found: {e}")
        return 1
    except ValueError as e:
        print(f"\n❌ ValueError: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        logger.exception("Pipeline error:")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

