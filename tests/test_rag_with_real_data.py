"""
Test script for RAG pipeline with real detected pose analysis issues.

This script simulates the complete workflow from detected pose issues → evidence retrieval,
testing the integration between pose analysis output and RAG system components.

Usage:
    python tests/test_rag_with_real_data.py
"""

import sys
from pathlib import Path
import logging
from typing import List, Dict, Any

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import RAG components
from src.rag.embeddings import EmbeddingGenerator
from src.rag.vector_store import VectorStore
from src.rag.retriever import HybridRetriever

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_vector_database(vector_db_path: Path) -> bool:
    """
    Check if vector database exists and is accessible.
    
    Args:
        vector_db_path: Path to the vector database directory.
    
    Returns:
        True if database exists and is accessible, False otherwise.
    """
    if not vector_db_path.exists():
        logger.error(f"Vector database directory not found: {vector_db_path}")
        return False
    
    if not vector_db_path.is_dir():
        logger.error(f"Vector database path is not a directory: {vector_db_path}")
        return False
    
    # Check if ChromaDB files exist
    chroma_files = list(vector_db_path.glob("*.sqlite3"))
    if not chroma_files:
        logger.error(f"No ChromaDB files found in: {vector_db_path}")
        return False
    
    logger.info(f"Vector database found at: {vector_db_path}")
    return True


def initialize_retriever() -> HybridRetriever:
    """
    Initialize the RAG retriever components.
    
    Returns:
        Initialized HybridRetriever instance.
    
    Raises:
        RuntimeError: If vector database is not accessible.
        Exception: If initialization fails.
    """
    # Define paths
    vector_db_path = project_root / "data" / "vector_db"
    
    # Check if vector database exists
    if not check_vector_database(vector_db_path):
        print("\n" + "=" * 70)
        print("VECTOR DATABASE NOT FOUND")
        print("=" * 70)
        print(f"\nVector database not found at: {vector_db_path}")
        print("\nTo fix this issue:")
        print("1. Run the RAG setup script:")
        print("   python scripts/setup_rag.py")
        print("\n2. Ensure the literature data exists:")
        print(f"   {project_root / 'data' / 'literature' / 'rehab_papers_chunked.json'}")
        print("\n3. Then run this test script again.")
        print("=" * 70)
        raise RuntimeError("Vector database not initialized")
    
    try:
        # Initialize components
        logger.info("Initializing EmbeddingGenerator...")
        embedder = EmbeddingGenerator(model_name='all-MiniLM-L6-v2')
        
        logger.info("Initializing VectorStore...")
        vector_store = VectorStore(persist_directory=str(vector_db_path))
        
        # Create/get collection
        logger.info("Getting collection...")
        vector_store.create_collection(collection_name='rehab_literature')
        
        # Initialize retriever
        logger.info("Initializing HybridRetriever...")
        retriever = HybridRetriever(embedder=embedder, vector_store=vector_store)
        
        logger.info("RAG system initialized successfully")
        return retriever
        
    except Exception as e:
        logger.error(f"Failed to initialize RAG system: {e}")
        raise


def format_issue_description(issue: Dict[str, Any]) -> str:
    """
    Format issue information for display.
    
    Args:
        issue: Issue dictionary with pose analysis data.
    
    Returns:
        Formatted string describing the issue.
    """
    frames = issue.get('frames_affected', [])
    timestamps = issue.get('timestamps_sec', [])
    
    # Format frame range
    if frames:
        if len(frames) == 1:
            frame_str = f"frame {frames[0]}"
        else:
            frame_str = f"frames {frames[0]}-{frames[-1]}"
    else:
        frame_str = "no frames specified"
    
    # Format timestamp range
    if timestamps:
        if len(timestamps) == 1:
            time_str = f"at {timestamps[0]}s"
        else:
            time_str = f"at {timestamps[0]:.2f}-{timestamps[-1]:.2f}s"
    else:
        time_str = "no timestamps specified"
    
    return f"{issue['description']} (affects {frame_str}, {time_str})"


def format_search_results(results: List[Dict[str, Any]], max_chars: int = 250) -> None:
    """
    Format and print search results.
    
    Args:
        results: List of search result dictionaries.
        max_chars: Maximum characters to show from text content.
    """
    if not results:
        print("   No results found.")
        return
    
    for i, result in enumerate(results, 1):
        chunk_id = result.get('chunk_id', 'unknown')
        relevance = result.get('relevance_score', 0.0)
        metadata = result.get('metadata', {})
        text = result.get('text', '')
        
        # Extract citation information
        source = metadata.get('source', 'Unknown source')
        evidence_level = metadata.get('evidence_level', 'Not specified')
        
        print(f"\n{i}. [{chunk_id}] (Relevance: {relevance:.2f})")
        print(f"   Source: {source}")
        print(f"   Evidence Level: {evidence_level}")
        print()
        print("   Recommendation:")
        if len(text) <= max_chars:
            print(f"   {text}")
        else:
            print(f"   {text[:max_chars]}...")
        print()


def test_rag_with_real_issues():
    """
    Test the RAG pipeline with real detected pose analysis issues.
    
    This function simulates the complete workflow:
    1. Define test issues matching pose analyzer output format
    2. Generate search queries for each issue
    3. Retrieve evidence using the RAG system
    4. Format and display results
    """
    print("🎯 TESTING RAG PIPELINE WITH REAL DETECTED ISSUES")
    print("=" * 70)
    
    # Test data matching biomechanics output format
    test_issues = [
        {
            "type": "asymmetry",
            "severity": "moderate",
            "side": "left",
            "magnitude_degrees": 13,
            "description": "Left side achieves 13° less knee flexion than right",
            "frames_affected": [45, 46, 47, 48],
            "timestamps_sec": [1.5, 1.53, 1.56, 1.6]
        },
        {
            "type": "depth_issue",
            "severity": "mild",
            "side": "bilateral",
            "magnitude_degrees": 8,
            "description": "Squat depth is 8° shallower than optimal range",
            "frames_affected": [23, 24, 25],
            "timestamps_sec": [0.75, 0.78, 0.81]
        },
        {
            "type": "valgus",
            "severity": "moderate",
            "side": "right",
            "magnitude_degrees": 12,
            "description": "Right knee exhibits 12° valgus collapse during descent",
            "frames_affected": [67, 68, 69],
            "timestamps_sec": [2.1, 2.13, 2.16]
        }
    ]
    
    try:
        # Initialize retriever
        print("\n📡 Initializing RAG system...")
        retriever = initialize_retriever()
        
        # Process each issue
        for i, issue in enumerate(test_issues, 1):
            print(f"\n🔍 Issue #{i}: {issue['type']} ({issue['severity']} severity)")
            print(f"   Description: {format_issue_description(issue)}")
            
            # Generate search query
            query = f"correction exercises for {issue['type']}"
            print(f"\n🔍 Search Query: \"{query}\"")
            
            # Search for evidence
            print(f"\n📚 Top 3 Evidence-Based Corrections:")
            print("-" * 35)
            
            try:
                results = retriever.search(
                    query=query,
                    issue_type=issue['type'],
                    n_results=3
                )
                format_search_results(results)
                
            except Exception as e:
                logger.error(f"Error searching for issue {issue['type']}: {e}")
                print(f"   Error: Failed to retrieve evidence for {issue['type']}")
                continue
        
        # Show how this feeds into agents
        print("\n💡 How this feeds into agents:")
        print("-" * 35)
        print("- Movement Agent will receive: test_issues dict")
        print("- RAG Agent will receive: search queries from Movement Agent")
        print("- Coach Agent will receive: retrieved evidence + movement analysis")
        print("- Final output: Personalized coaching with source citations")
        
        print("\n" + "=" * 70)
        print("✅ RAG PIPELINE TEST COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        
        return True
        
    except RuntimeError as e:
        # Vector database not found - already handled in initialize_retriever
        return False
        
    except Exception as e:
        logger.error(f"Unexpected error during testing: {e}")
        print(f"\n❌ Test failed with error: {e}")
        print("\nPlease check:")
        print("1. All dependencies are installed: pip install -r requirements.txt")
        print("2. Vector database is initialized: python scripts/setup_rag.py")
        print("3. Literature data exists in data/literature/")
        return False


def main():
    """
    Main function to run the RAG pipeline test.
    
    Returns:
        0 if successful, 1 if failed.
    """
    try:
        success = test_rag_with_real_issues()
        return 0 if success else 1
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Test interrupted by user")
        return 1
        
    except Exception as e:
        logger.exception("Unexpected error in main:")
        print(f"\n❌ Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
