# tests/test_vector_store.py
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.rag.vector_store import VectorStore
import numpy as np

store = VectorStore()
collection = store.create_collection()
print(f"✅ Collection created: {collection.name}")

# Test add
test_embeddings = np.random.rand(2, 384)
store.add_documents(
    texts=["test doc 1", "test doc 2"],
    embeddings=test_embeddings,
    metadatas=[{"test": "1"}, {"test": "2"}],
    ids=["test_1", "test_2"]
)
print(f"✅ Documents added: {store.count()}")
