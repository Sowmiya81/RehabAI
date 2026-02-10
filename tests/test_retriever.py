# tests/test_retriever.py

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.rag.embeddings import EmbeddingGenerator
from src.rag.vector_store import VectorStore
from src.rag.retriever import HybridRetriever

# Initialize components
print("Initializing EmbeddingGenerator...")
embedder = EmbeddingGenerator()

print("Initializing VectorStore...")
store = VectorStore(persist_directory='./data/vector_db')

print("Creating/getting collection...")
store.create_collection(collection_name='rehab_literature')

print("Creating HybridRetriever...")
retriever = HybridRetriever(embedder, store)

# Check if collection has documents
count = store.count()
print(f"\nCollection contains {count} documents")

if count == 0:
    print("\nWarning: Collection is empty!")
    print("Please run 'python scripts/setup_rag.py' first to populate the database.")
    sys.exit(1)

# Test search
print("\n" + "=" * 70)
print("Testing search...")
print("=" * 70)

results = retriever.search(
    query="how to fix asymmetry in squats",
    n_results=3
)

print(f"\nRetrieved {len(results)} results")
for i, result in enumerate(results, 1):
    print(f"\n{i}. [{result['chunk_id']}] (score: {result['relevance_score']:.2f})")
    print(f"   {result['text'][:150]}...")
    if 'source' in result['metadata']:
        print(f"   Source: {result['metadata']['source']}")

# Test search with filters
print("\n" + "=" * 70)
print("Testing search with filters...")
print("=" * 70)

filtered_results = retriever.search(
    query="exercises for knee valgus",
    issue_type="knee_valgus",
    n_results=3
)

print(f"\nRetrieved {len(filtered_results)} filtered results")
for i, result in enumerate(filtered_results, 1):
    print(f"\n{i}. [{result['chunk_id']}] (score: {result['relevance_score']:.2f})")
    print(f"   {result['text'][:150]}...")
    if 'source' in result['metadata']:
        print(f"   Source: {result['metadata']['source']}")

print("\n" + "=" * 70)
print("All tests completed!")
print("=" * 70)
