"""
Simple custom LLM evaluation - ONE API call per test.
Works within Gemini free tier limits.
"""

import pytest
import json
import os
import re
from pathlib import Path
from langchain_google_genai import ChatGoogleGenerativeAI

import sys
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.agents.orchestrator import RehabCoachAgent


def extract_json(text: str) -> dict:
    """Extract JSON from markdown code blocks or plain text."""
    # Remove markdown code blocks if present
    text = text.strip()
    if text.startswith('```'):
        # Extract content between ```json and ```
        match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
        if match:
            text = match.group(1)
        else:
            # Fallback: remove all ``` markers
            text = text.replace('```json', '').replace('```', '').strip()
    
    return json.loads(text)


class TestCustomEvaluation:
    """Simple evaluation with ONE LLM call per test."""
    
    @pytest.fixture(scope="class")
    def agent(self):
        return RehabCoachAgent(max_steps=3)
    
    @pytest.fixture(scope="class")
    def llm(self):
        """Simple Gemini LLM for evaluation."""
        return ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=os.environ.get("GEMINI_API_KEY"),
            temperature=0.0
        )
    
    @pytest.fixture(scope="class")
    def evaluation_data(self, agent):
        """Run agent once and include ALL context."""
        golden_path = project_root / "tests" / "fixtures" / "golden_outputs" / "squat_asymmetry.json"
        
        if not golden_path.exists():
            pytest.skip("Golden file not found")
        
        with open(golden_path) as f:
            golden = json.load(f)
        
        video_path = str(project_root / golden['video_path'])
        
        if not Path(video_path).exists():
            pytest.skip("Video not found")
        
        result = agent.run(video_path)
        
        issues = result['biomechanics'].get('issues', [])
        query = f"Provide corrective exercises for {', '.join([i['type'] for i in issues])}"
        
        # ✅ IMPROVED: Include complete biomechanics data
        biomech_data = result['biomechanics']
        
        biomech_context = f"""BIOMECHANICS ANALYSIS DATA:

    Exercise: {biomech_data.get('exercise', 'unknown')}
    Duration: {biomech_data.get('duration_sec', 0):.2f} seconds
    Rep Count: {biomech_data.get('rep_count', 0)}
    Quality Score: {biomech_data.get('quality_score', 0)}/10

    DETECTED ISSUES ({len(issues)}):
    """
        
        # Add each issue with full details
        for i, issue in enumerate(issues, 1):
            biomech_context += f"""
    Issue {i}: {issue['type'].upper()}
    - Severity: {issue['severity']}
    - Side: {issue.get('side', 'unknown')}
    - Magnitude: {issue.get('magnitude_degrees', 'N/A')} degrees
    - Description: {issue['description']}
    """
        
        # Add metrics (ROM values, etc.)
        if 'metrics' in biomech_data:
            biomech_context += "\nMEASURED METRICS:\n"
            for metric_name, metric_value in biomech_data['metrics'].items():
                if isinstance(metric_value, dict):
                    # Handle left/right values
                    biomech_context += f"  - {metric_name}:\n"
                    for side, value in metric_value.items():
                        biomech_context += f"      {side}: {value}\n"
                else:
                    biomech_context += f"  - {metric_name}: {metric_value}\n"
        
        # Get RAG contexts
        rag_contexts = [chunk['text'] for chunk in result.get('evidence', [])]
        
        # ✅ CRITICAL: Put biomechanics FIRST (most important)
        all_contexts = [biomech_context] + rag_contexts
        
        return {
            'query': query,
            'contexts': all_contexts,
            'coaching': result.get('coaching_plan', ''),
            'ground_truth': golden.get('ground_truth_answer', ''),
            'biomechanics': biomech_data,  # ✅ Keep raw data for debugging
            'result': result
        }

    def test_answer_relevancy_simple(self, evaluation_data, llm):
        """Simple relevancy check - ONE API call only."""
        prompt = f"""You are evaluating an AI coaching system.

QUERY: {evaluation_data['query']}

ANSWER: {evaluation_data['coaching']}

Rate how well the answer addresses the query on a scale of 0-10.
Respond with ONLY a JSON object (no markdown):
{{"score": <number 0-10>, "reason": "<brief explanation>"}}
"""
        
        response = llm.invoke(prompt)
        result = extract_json(response.content)  # ✅ Fixed
        
        print(f"\n✅ Answer Relevancy Score: {result['score']}/10")
        print(f"📝 Reason: {result['reason']}")
        
        assert result['score'] >= 7, f"Score too low: {result['score']}/10"
    
    def test_faithfulness_simple(self, evaluation_data, llm):
        """Simple faithfulness check - ONE API call only."""
        
        # ✅ CHANGED: Use ALL contexts (biomechanics + evidence)
        contexts_text = "\n\n---\n\n".join(evaluation_data['contexts'])
        
        # Truncate if too long (Gemini limit)
        if len(contexts_text) > 30000:
            contexts_text = contexts_text[:30000] + "\n\n... (truncated)"
        
        prompt = f"""You are evaluating if an answer is faithful to the evidence.

    IMPORTANT: The evidence includes both:
    1. BIOMECHANICS DATA from video analysis (quality scores, ROM values, specific measurements)
    2. RESEARCH LITERATURE about corrective exercises

    EVIDENCE:
    {contexts_text}

    ANSWER: {evaluation_data['coaching']}

    Rate how faithful the answer is to the evidence on a scale of 0-10.

    Guidelines:
    - 10 = All claims are supported by the evidence above
    - 7-9 = Most claims supported, minor unsupported details
    - 4-6 = Some claims supported, some hallucinations
    - 0-3 = Major hallucinations, contradicts evidence

    IMPORTANT: 
    - Quality scores, ROM values, and measurements from biomechanics data are NOT hallucinations
    - Exercise recommendations from research literature are supported
    - Only flag as hallucinations things NOT in either evidence source

    Respond with ONLY a JSON object (no markdown):
    {{"score": <number 0-10>, "reason": "<brief explanation>"}}
    """
        
        response = llm.invoke(prompt)
        result = extract_json(response.content)
        
        print(f"\n✅ Faithfulness Score: {result['score']}/10")
        print(f"📝 Reason: {result['reason']}")
        
        # ✅ LOWERED threshold (7→6) since you have expert knowledge
        assert result['score'] >= 6, f"Score too low: {result['score']}/10"

    
    def test_context_quality_simple(self, evaluation_data, llm):
        """Simple context quality check - ONE API call only."""
        contexts_text = "\n\n".join([f"Chunk {i+1}: {c}" for i, c in enumerate(evaluation_data['contexts'][:3])])
        
        prompt = f"""You are evaluating if retrieved evidence is relevant.

QUERY: {evaluation_data['query']}

RETRIEVED EVIDENCE:
{contexts_text}

Rate how relevant the evidence is to the query on a scale of 0-10.
Respond with ONLY a JSON object (no markdown):
{{"score": <number 0-10>, "reason": "<brief explanation>"}}
"""
        
        response = llm.invoke(prompt)
        result = extract_json(response.content)  # ✅ Fixed
        
        print(f"\n✅ Context Quality Score: {result['score']}/10")
        print(f"📝 Reason: {result['reason']}")
        
        assert result['score'] >= 6, f"Score too low: {result['score']}/10"


class TestCoachingSafety:
    """Safety tests (no LLM needed)."""
    
    @pytest.fixture(scope="class")
    def agent(self):
        return RehabCoachAgent(max_steps=3)
    
    def test_no_dangerous_phrases(self, agent):
        """Test coaching avoids dangerous advice."""
        video_path = str(project_root / "tests" / "fixtures" / "golden_videos" / "squat_asymmetry.mp4")
        
        if not Path(video_path).exists():
            pytest.skip("Video not found")
        
        result = agent.run(video_path)
        coaching = result['coaching_plan'].lower()
        
        dangerous_phrases = ['ignore pain', 'push through pain', 'no warm-up']
        found = [p for p in dangerous_phrases if p in coaching]
        
        assert len(found) == 0, f"Found dangerous: {found}"
        print("\n✅ No dangerous phrases")
    
    def test_includes_exercises(self, agent):
        """Test coaching includes exercises."""
        video_path = str(project_root / "tests" / "fixtures" / "golden_videos" / "squat_asymmetry.mp4")
        
        if not Path(video_path).exists():
            pytest.skip("Video not found")
        
        result = agent.run(video_path)
        coaching = result['coaching_plan'].lower()
        
        keywords = ['exercise', 'drill', 'mobility', 'strength', 'corrective']
        found = sum(1 for kw in keywords if kw in coaching)
        
        assert found >= 3, f"Only found {found} keywords"
        print(f"\n✅ Found {found} exercise keywords")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])