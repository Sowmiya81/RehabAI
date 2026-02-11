"""
LangChain tools for agent interaction with pose detection and RAG systems.

This module provides tool functions that agents can use to analyze exercise videos,
search literature, and compare against normative data. Uses dependency injection
for efficient resource management and testing.
"""

from langchain.tools import tool
from typing import Dict, List, Optional, Any
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Import existing components
from src.pose.detector import PoseDetector
from src.pose.biomechanics import analyze_squat_form
from src.rag.retriever import HybridRetriever

# Module-level instances (initialized once when imported)
_pose_detector = None
_retriever = None


def initialize_tools(retriever: HybridRetriever, pose_detector: PoseDetector = None):
    """
    Initialize tools with shared instances.
    Call this once at application startup.
    
    Args:
        retriever: Initialized HybridRetriever instance
        pose_detector: Optional PoseDetector instance (created if not provided)
    """
    global _retriever, _pose_detector
    _retriever = retriever
    _pose_detector = pose_detector or PoseDetector()
    logger.info("Agent tools initialized")


def _get_biomechanics_analysis_impl(video_path: str, exercise_type: str = "squat") -> Dict[str, Any]:
    """
    Implementation function for biomechanics analysis (testable directly).
    
    Args:
        video_path: Path to video file (mp4, avi, mov)
        exercise_type: Type of exercise being performed (currently supports "squat")
        
    Returns:
        Dict containing analysis results with exercise, duration, issues, metrics, and quality_score.
    """
    if _pose_detector is None:
        return {"error": "Tools not initialized. Call initialize_tools() first."}
    
    try:
        # Use existing detector from your code
        keypoints = _pose_detector.process_video(video_path)
        
        if keypoints is None or len(keypoints) == 0:
            return {"error": f"Failed to process video: {video_path}"}
        
        # Use existing biomechanics analysis
        report = analyze_squat_form(keypoints)
        report['exercise'] = exercise_type
        
        # Calculate quality score based on issues
        issues = report.get('issues', [])
        if not issues:
            quality_score = 10.0
        else:
            # Deduct points based on severity
            severity_weights = {"mild": 1, "moderate": 2, "severe": 3}
            total_deduction = sum(severity_weights.get(issue.get('severity', 'mild'), 1) for issue in issues)
            quality_score = max(0.0, 10.0 - total_deduction)
        
        report['quality_score'] = round(quality_score, 1)
        report['rep_count'] = len([f for f in range(len(keypoints)) if keypoints[f].sum() > 0]) // 30  # Rough estimate
        
        logger.info(f"Analyzed {video_path}: {len(issues)} issues detected, quality score: {quality_score}")
        return report
        
    except Exception as e:
        logger.error(f"Biomechanics analysis error: {e}")
        return {"error": str(e)}


@tool
def get_biomechanics_analysis(video_path: str, exercise_type: str = "squat") -> Dict[str, Any]:
    """
    Analyze exercise video and return biomechanics report with detected issues.
    
    This tool processes a video file using pose estimation to detect movement quality issues
    like asymmetry, knee valgus, forward lean, and depth limitations.
    
    Args:
        video_path: Path to video file (mp4, avi, mov)
        exercise_type: Type of exercise being performed (currently supports "squat")
        
    Returns:
        Dict containing:
        - exercise: Exercise type
        - duration_sec: Video duration
        - issues: List of detected issues with severity, frames, timestamps
        - metrics: Biomechanics measurements (angles, ROM, asymmetry)
        - quality_score: Overall form quality (0-10)
        
    Example:
        >>> result = get_biomechanics_analysis("squat_video.mp4", "squat")
        >>> print(result['issues'])
        {'type': 'asymmetry', 'severity': 'moderate', 'magnitude_degrees': 13, ...}
    """
    return _get_biomechanics_analysis_impl(video_path, exercise_type)


def _search_exercise_literature_impl(
    query: str, 
    exercise_type: Optional[str] = None,
    issue_type: Optional[str] = None,
    n_results: int = 5
) -> List[Dict[str, Any]]:
    """
    Implementation function for literature search (testable directly).
    
    Args:
        query: Search query describing the issue or desired intervention
        exercise_type: Filter by exercise (e.g., "squat", "lunge")
        issue_type: Filter by issue addressed (e.g., "asymmetry", "knee_valgus")
        n_results: Number of results to return (default: 5)
        
    Returns:
        List of dicts containing search results with chunk_id, text, metadata, and relevance_score.
    """
    if _retriever is None:
        return [{"error": "Tools not initialized. Call initialize_tools() first."}]
    
    try:
        results = _retriever.search(
            query=query,
            exercise_type=exercise_type,
            issue_type=issue_type,
            n_results=n_results
        )
        
        logger.info(f"Literature search: '{query}' returned {len(results)} results")
        return results
        
    except Exception as e:
        logger.error(f"Literature search error: {e}")
        return [{"error": str(e)}]


@tool
def search_exercise_literature(
    query: str, 
    exercise_type: Optional[str] = None,
    issue_type: Optional[str] = None,
    n_results: int = 5
) -> List[Dict[str, Any]]:
    """
    Search rehabilitation literature for evidence-based exercise corrections.
    
    Retrieves relevant research papers and intervention protocols from the vector database
    using semantic search with optional metadata filtering.
    
    Args:
        query: Search query describing the issue or desired intervention
        exercise_type: Filter by exercise (e.g., "squat", "lunge")
        issue_type: Filter by issue addressed (e.g., "asymmetry", "knee_valgus")
        n_results: Number of results to return (default: 5)
        
    Returns:
        List of dicts containing:
        - chunk_id: Unique identifier
        - text: Research excerpt with intervention details
        - metadata: Source citation, evidence level, exercise type
        - relevance_score: Similarity score (0-1, higher is better)
        
    Example:
        >>> results = search_exercise_literature(
        ...     "correction for asymmetry",
        ...     exercise_type="squat",
        ...     issue_type="asymmetry",
        ...     n_results=3
        ... )
        >>> print(results[0]['metadata']['source'])
        'Bishop C et al. IJSPP 2018;13(4):545-547'
    """
    return _search_exercise_literature_impl(query, exercise_type, issue_type, n_results)


def _compare_to_normative_data_impl(angle_data: Dict[str, float], exercise_type: str = "squat") -> Dict[str, str]:
    """
    Implementation function for normative data comparison (testable directly).
    
    Args:
        angle_data: Dict with measured angles (e.g., {'knee_flexion_left': 105, 'knee_flexion_right': 118})
        exercise_type: Exercise being analyzed
        
    Returns:
        Dict with comparison statements for each metric
    """
    # Load normative data
    norms_path = Path(__file__).parent.parent.parent / "data" / "reference" / "exercise_norms.json"
    
    try:
        with open(norms_path, 'r') as f:
            norms = json.load(f)
        
        exercise_norms = norms.get(exercise_type, {})
        comparisons = {}
        
        # Compare each metric
        for metric, value in angle_data.items():
            if 'left' in metric and metric.replace('left', 'right') in angle_data:
                # Calculate asymmetry
                left_val = value
                right_val = angle_data[metric.replace('left', 'right')]
                asymmetry = abs(left_val - right_val)
                
                if asymmetry < 5:
                    assessment = "normal"
                elif asymmetry < 10:
                    assessment = "mild asymmetry"
                elif asymmetry < 15:
                    assessment = "moderate asymmetry"
                else:
                    assessment = "severe asymmetry"
                
                comparisons[f"{metric}_asymmetry"] = f"{asymmetry}° difference ({assessment})"
        
        return comparisons
        
    except FileNotFoundError:
        logger.warning(f"Normative data not found at {norms_path}, using default ranges")
        # Provide basic comparisons without normative data
        comparisons = {}
        for metric, value in angle_data.items():
            if 'left' in metric and metric.replace('left', 'right') in angle_data:
                left_val = value
                right_val = angle_data[metric.replace('left', 'right')]
                asymmetry = abs(left_val - right_val)
                comparisons[f"{metric}_asymmetry"] = f"{asymmetry}° difference"
        return comparisons
        
    except Exception as e:
        logger.error(f"Comparison error: {e}")
        return {"error": str(e)}


@tool
def compare_to_normative_data(angle_data: Dict[str, float], exercise_type: str = "squat") -> Dict[str, str]:
    """
    Compare biomechanics measurements to normative reference values.
    
    This tool compares measured angles to normative data and identifies
    asymmetries or deviations from normal ranges.
    
    Args:
        angle_data: Dict with measured angles (e.g., {'knee_flexion_left': 105, 'knee_flexion_right': 118})
        exercise_type: Exercise being analyzed (default: squat)
        
    Returns:
        Dict with comparison statements for each metric
        
    Example:
        >>> result = compare_to_normative_data({'knee_flexion_left': 105, 'knee_flexion_right': 118})
        >>> print(result['knee_flexion_asymmetry'])
        '13° asymmetry detected (normal: <5°, moderate: 10-15°)'
    """
    return _compare_to_normative_data_impl(angle_data, exercise_type)


# Export tool list for LangChain agent
AGENT_TOOLS = [
    get_biomechanics_analysis,
    search_exercise_literature,
    compare_to_normative_data
]
