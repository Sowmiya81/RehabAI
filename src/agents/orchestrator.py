import google.genai as genai
import json
import logging
import os
import sys
from pathlib import Path
from typing import TypedDict, Annotated, Optional
from langgraph.graph import StateGraph, END
import operator
from contextlib import nullcontext

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv(project_root / ".env") 

from src.agents.tools import (
    _get_biomechanics_analysis_impl,
    _search_exercise_literature_impl,
    _compare_to_normative_data_impl
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:%(name)s:%(message)s'
)
logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    """State flowing through agent graph."""
    messages: Annotated[list[str], operator.add]
    video_path: str
    user_context: Optional[dict]
    
    current_step: int
    max_steps: int
    
    biomechanics: Optional[dict]
    search_queries: Optional[list]
    evidence: list
    coaching_plan: Optional[str]
    
    next_action: Optional[str]
    agent_finished: bool


class RehabCoachAgent:
    """LangGraph agent with full debug logging."""
    
    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        temperature: float = 0.4,
        max_steps: int = 3
    ):
        self.model = model
        self.temperature = temperature
        self.max_steps = max_steps
        
        # Initialize Gemini client
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY not set. Run: export GEMINI_API_KEY='your-key'"
            )
        
        self.client = genai.Client(api_key=api_key)
        
        logger.info(f"Agent initialized: {model}, max_steps={max_steps}")
        
        # Initialize tools (CRITICAL - must do before using biomechanics/RAG)
        logger.info("Initializing tools (CV + RAG)...")
        try:
            from src.rag.embeddings import EmbeddingGenerator
            from src.rag.vector_store import VectorStore
            from src.rag.retriever import HybridRetriever
            from src.agents.tools import initialize_tools
            
            # Create embedder
            embedder = EmbeddingGenerator()
            logger.info("  ✓ Embedder created")
            
            # Create vector store
            self.vector_store = VectorStore(persist_directory="./data/vector_db")
            self.vector_store.create_collection(collection_name="rehab_literature")
            logger.info("  ✓ Vector store created")

            if self.vector_store.count() == 0:
                logger.info("Vector DB empty — running ingestion pipeline...")
                from src.rag.ingest import ingest_literature
                ingested = ingest_literature(self.vector_store, embedder)
                logger.info(f"Ingestion complete: {ingested} documents added")
            
            # Create retriever with embedder and vector store
            retriever = HybridRetriever(embedder=embedder, vector_store=self.vector_store)
            logger.info("  ✓ Retriever created")
            
            # Initialize tools with retriever
            initialize_tools(retriever=retriever)
            
            logger.info("✅ Tools initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Tools initialization failed: {e}", exc_info=True)
            logger.error("Biomechanics and RAG may not work properly")
            logger.error("")
            logger.error("Common causes:")
            logger.error("  1. ChromaDB collection 'rehab_literature' doesn't exist")
            logger.error("  2. RAG data hasn't been ingested yet")
            logger.error("  3. Sentence-transformers model not downloaded")
            logger.error("")
            logger.error("To fix:")
            logger.error("  python scripts/ingest_literature.py")

        
        # Build graph
        self.graph = self._build_graph()
        logger.info("LangGraph compiled successfully")
    
    def _build_graph(self) -> StateGraph:
        """Build LangGraph state machine."""
        
        graph = StateGraph(AgentState)
        
        # Add nodes
        graph.add_node("reason", self._reasoning_node)
        graph.add_node("analyze_video", self._analyze_video_node)
        graph.add_node("search_literature", self._search_literature_node)
        graph.add_node("generate_coaching", self._generate_coaching_node)
        
        # Set entry point
        graph.set_entry_point("reason")
        
        # Add conditional edges from reasoning
        graph.add_conditional_edges(
            "reason",
            self._route_next_action,
            {
                "analyze_video": "analyze_video",
                "search_literature": "search_literature",
                "generate_coaching": "generate_coaching",
                "finish": END
            }
        )
        
        # Add edges back to reasoning
        graph.add_edge("analyze_video", "reason")
        graph.add_edge("search_literature", "reason")
        graph.add_edge("generate_coaching", END)
        
        return graph.compile()
    
    def _reasoning_node(self, state: AgentState) -> dict:
        """Agent decides next action."""
        
        current_step = state.get('current_step', 0) + 1
        
        logger.info("="*70)
        logger.info(f"REASONING NODE - Step {current_step}/{self.max_steps}")
        logger.info("="*70)
        
            
        if current_step > self.max_steps:
            logger.warning(f"Max steps reached, forcing completion")
            return {
                "current_step": current_step,
                "next_action": "generate_coaching",
                "agent_finished": True,
                "messages": ["Max steps reached"]
            }
        
        has_biomechanics = state.get('biomechanics') is not None
        has_evidence = len(state.get('evidence', [])) > 0
        has_coaching = state.get('coaching_plan') is not None
        
        logger.info(f"Current state:")
        logger.info(f"  - Biomechanics analyzed: {has_biomechanics}")
        logger.info(f"  - Evidence collected: {has_evidence} ({len(state.get('evidence', []))} chunks)")
        logger.info(f"  - Coaching generated: {has_coaching}")
        
        prompt = f"""
You are a fitness coaching AI agent. Analyze current state and decide next action.

CURRENT STATE:
Step: {current_step}/{self.max_steps}
Video analyzed: {has_biomechanics}
Evidence collected: {len(state.get('evidence', []))} studies
Coaching generated: {has_coaching}

AVAILABLE ACTIONS:
1. analyze_video - Get biomechanics from video (use if not done)
2. search_literature - Search evidence (use if biomechanics done but no evidence)
3. generate_coaching - Create coaching plan (use if have biomechanics and evidence)
4. finish - Done (use if coaching generated)

GUIDELINES:
- Be decisive, do not loop unnecessarily
- If step 3/3, MUST choose generate_coaching or finish

Output ONLY this JSON:
{{
"reasoning": "one sentence explaining decision",
"action": "action_name",
"finished": true/false
}}
"""
        
        try:
            logger.info("Calling Gemini for reasoning...")
            
            # FIXED: Correct API syntax for google-genai v1.60+
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config={
                    "temperature": self.temperature,
                    "max_output_tokens": 2048,
                }
            )
            response_text = response.text.strip()
            
            logger.info(f"Gemini response: {response_text[:200]}")
            
            # Parse JSON
            if "```json" in response_text:
                json_str = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                json_str = response_text.split("```").split("```").strip()[1]
            else:
                json_str = response_text
            
            decision = json.loads(json_str)
            
            logger.info(f"Decision: {decision['action']}")
            logger.info(f"Reasoning: {decision['reasoning']}")
            logger.info("="*70)
            
            return {
                "current_step": current_step,
                "next_action": decision['action'],
                "agent_finished": decision.get('finished', False),
                "messages": [f"Step {current_step}: {decision['reasoning']}"]
            }
            
        except Exception as e:
            logger.error(f"Reasoning error: {e}", exc_info=True)
            logger.warning("Using fallback logic")
            
            if not has_biomechanics:
                action = "analyze_video"
            elif not has_evidence:
                action = "search_literature"
            else:
                action = "generate_coaching"
            
            logger.info(f"Fallback action: {action}")
            logger.info("="*70)
            
            return {
                "current_step": current_step,
                "next_action": action,
                "agent_finished": False,
                "messages": [f"Step {current_step}: Fallback to {action}"]
            }
    
    def _analyze_video_node(self, state: AgentState) -> dict:
        """Execute CV analysis."""
        
        video_path = state['video_path']
        
        logger.info("="*70)
        logger.info("ANALYZE VIDEO NODE")
        logger.info("="*70)
        
        logger.info(f"Analyzing: {video_path}")
        
        try:
            biomechanics = _get_biomechanics_analysis_impl(video_path, "squat")
            
            logger.info("FULL BIOMECHANICS OUTPUT:")
            logger.info(json.dumps(biomechanics, indent=2, default=str))
            
            # Check for initialization error
            if "error" in biomechanics:
                logger.error(f"Biomechanics error: {biomechanics['error']}")
                logger.error("Tools may not be initialized properly")
            
            issues = biomechanics.get('issues', [])
            metrics = biomechanics.get('metrics', {})
            
            logger.info(f"Issues found: {len(issues)}")
            logger.info(f"Metrics: {list(metrics.keys())}")
            
            # Check asymmetry
            if 'asymmetry_score' in metrics:
                logger.info(f"Asymmetry score: {metrics['asymmetry_score']}")
                if len(issues) == 0 and metrics['asymmetry_score'] > 0:
                    logger.warning("⚠️ ASYMMETRY DETECTED BUT NO ISSUES!")
                    logger.warning("Check biomechanics.py thresholds")
            
            # Generate queries
            queries = []
            if issues:
                for issue in issues[:2]:
                    queries.append(f"{issue.get('type', 'movement')} correction squat")
            else:
                queries = ["squat form optimization", "squat mobility"]
            
            logger.info(f"Generated queries: {queries}")
            logger.info("="*70)
            
            return {
                "biomechanics": biomechanics,
                "search_queries": queries,
                "messages": [f"Analyzed video: {len(issues)} issues"]
            }
            
        except Exception as e:
            logger.error(f"Video analysis error: {e}", exc_info=True)
            logger.error("="*70)
            return {
                "biomechanics": {"error": str(e), "issues": []},
                "search_queries": ["squat general"],
                "messages": [f"Analysis failed: {e}"]
            }
    
    def _search_literature_node(self, state: AgentState) -> dict:
        """Execute RAG search."""
        
        queries = state.get('search_queries', []) or ["squat correction"]
        
        logger.info("="*70)
        logger.info("SEARCH LITERATURE NODE")
        logger.info("="*70)
        
        logger.info(f"Queries: {queries}")
        
        all_evidence = []
        
        for i, query in enumerate(queries[:2], 1):
            try:
                logger.info(f"Query {i}: '{query}'")
                
                results = _search_exercise_literature_impl(
                    query=query,
                    exercise_type="squat",
                    issue_type=None,
                    n_results=3
                )
                
                logger.info(f"  Retrieved {len(results)} results")
                
                for j, r in enumerate(results, 1):
                    chunk_id = r.get('chunk_id', 'None')
                    score = r.get('relevance_score', 0)
                    logger.info(f"    [{j}] chunk_id={chunk_id}, score={score:.3f}")
                
                all_evidence.extend(results)
                
            except Exception as e:
                logger.error(f"Search error: {e}", exc_info=True)
        
            # Deduplicate
            seen = set()
            unique = []
            for doc in all_evidence:
                cid = doc.get('chunk_id', '')
                if cid and cid not in seen:
                    seen.add(cid)
                    unique.append(doc)
            
            # ADDED: Check for invalid chunks
            invalid_chunks = []
            valid_chunks = []
            
            for doc in unique:
                if doc.get('chunk_id') is None or doc.get('chunk_id') == '':
                    invalid_chunks.append(doc)
                else:
                    valid_chunks.append(doc)
            
            if invalid_chunks:
                logger.error(f"⚠️ Found {len(invalid_chunks)} chunks with chunk_id=None")
                logger.error("This means ChromaDB collection is empty or not loaded!")
                logger.error("Run: python scripts/ingest_literature.py (or similar)")
            
            unique = valid_chunks
            
            logger.info(f"Total: {len(all_evidence)} → Unique: {len(unique)}")
            
            if len(unique) == 0:
                logger.error("⚠️ NO VALID EVIDENCE! RAG system needs data")
            
            logger.info("="*70)
            
            return {
                "evidence": unique,
                "messages": [f"Retrieved {len(unique)} chunks"]
            }
    
    def _generate_coaching_node(self, state: AgentState) -> dict:
        """Generate coaching plan."""
        
        logger.info("="*70)
        logger.info("GENERATE COACHING NODE")
        logger.info("="*70)
        
        biomechanics = state.get('biomechanics', {})
        evidence = state.get('evidence', [])
            
        prompt = f"""
You are an expert fitness coach. Create a coaching plan.

BIOMECHANICS:
{json.dumps(biomechanics, indent=2, default=str)}

EVIDENCE:
{self._format_evidence(evidence)}

Create a brief coaching plan in Markdown with:
- Movement Analysis
- Corrective Exercises
- Evidence citations

Keep it concise.
"""
        
        try:
            # FIXED: Correct API syntax
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config={
                    "temperature": self.temperature,
                    "max_output_tokens": 2048,
                }
            )
            
            plan = response.text
            logger.info(f"Generated {len(plan)} chars")
            logger.info("="*70)
            
            return {
                "coaching_plan": plan,
                "agent_finished": True,
                "messages": ["Coaching generated"]
            }
            
        except Exception as e:
            logger.error(f"Generation error: {e}", exc_info=True)
            logger.error("="*70)
            return {
                "coaching_plan": f"Error: {e}",
                "agent_finished": True,
                "messages": [f"Failed: {e}"]
            }
    
    def _route_next_action(self, state: AgentState) -> str:
        """Route based on next_action."""
        
        if state.get('agent_finished'):
            logger.info("Routing to: FINISH")
            return "finish"
        
        action = state.get('next_action', 'finish')
        
        logger.info(f"Routing to: {action}")
        
        if action in ["analyze_video", "search_literature", "generate_coaching"]:
            return action
        
        return "finish"
    
    def _format_evidence(self, evidence: list) -> str:
        """Format evidence for prompt."""
        
        if not evidence:
            return "No evidence available"
        
        formatted = []
        for i, doc in enumerate(evidence[:3], 1):
            formatted.append(f"[{i}] {doc.get('text', '')[:200]}...")
        
        return "\n".join(formatted)
    
    def run(self, video_path: str, user_context: dict = None) -> dict:
        """Run agent."""
        
        initial_state = {
            "messages": [],
            "video_path": video_path,
            "user_context": user_context or {},
            "current_step": 0,
            "max_steps": self.max_steps,
            "biomechanics": None,
            "search_queries": None,
            "evidence": [],
            "coaching_plan": None,
            "next_action": None,
            "agent_finished": False
        }
        
        logger.info(f"Starting agent run: {video_path}")
        
        try:
            final_state = self.graph.invoke(initial_state)
            
            result = {
                "biomechanics": final_state.get('biomechanics'),
                "evidence": final_state.get('evidence'),
                "coaching_plan": final_state.get('coaching_plan'),
                "agent_trace": final_state.get('messages', []),
                "total_steps": final_state.get('current_step', 0)
            }
            
            logger.info(f"✅ Agent completed in {result['total_steps']} steps")
            
            return result
            
        except Exception as e:
            logger.error(f"Agent run failed: {e}", exc_info=True)
            raise


def create_agent(max_steps: int = 3) -> RehabCoachAgent:
    """Factory function."""
    return RehabCoachAgent(max_steps=max_steps)


if __name__ == "__main__":
    agent = create_agent()
    
    test_video = "data/videos/test/my_squat.mp4"
    
    print("\n" + "="*70)
    print("Running agent on test video...")
    print("="*70)
    
    result = agent.run(test_video)
    
    print("\n" + "="*70)
    print("AGENT TRACE:")
    print("="*70)
    for msg in result['agent_trace']:
        print(f"  - {msg}")
    
    print("\n" + "="*70)
    print("SUMMARY:")
    print("="*70)
    print(f"Completed in: {result['total_steps']} steps")
    
    bio = result.get('biomechanics')
    if bio:
        print(f"Issues found: {len(bio.get('issues', []))}")
    else:
        print("Biomechanics: None")
    
    print(f"Evidence retrieved: {len(result.get('evidence', []))} chunks")
    
    plan = result.get('coaching_plan')
    if plan:
        print("\n" + "="*70)
        print("COACHING PLAN PREVIEW:")
        print("="*70)
        print(plan)
