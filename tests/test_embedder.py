# tests/test_embedder.py
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.rag.embeddings import EmbeddingGenerator

embedder = EmbeddingGenerator()
test_texts = ["hip strengthening exercises", "knee valgus correction"]
embeddings = embedder.embed_chunks(test_texts)
print(f"✅ Generated embeddings shape: {embeddings.shape}")
print(f"Expected: (2, 384)")
