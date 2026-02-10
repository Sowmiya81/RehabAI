"""
Script to initialize the RAG system by loading literature and creating vector database.

This script:
1. Loads literature corpus from JSON file
2. Generates embeddings for all chunks
3. Stores them in ChromaDB vector database
4. Tests retrieval with a sample query
"""

import json
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Try to import required packages and provide helpful error messages
try:
    from src.rag import EmbeddingGenerator, VectorStore
except ImportError as e:
    print("=" * 70)
    print("MISSING DEPENDENCIES ERROR")
    print("=" * 70)
    print(f"\nError: {e}")
    print("\nThe required packages are not installed in your Python environment.")
    print("\nTo fix this issue:")
    print("\n1. Install the missing packages:")
    print("   pip install sentence-transformers chromadb numpy")
    print("\n2. Or install all requirements from requirements.txt:")
    print(f"   pip install -r {project_root / 'requirements.txt'}")
    print("\n3. If you're using a virtual environment, make sure it's activated:")
    print("   source venv/bin/activate  # On macOS/Linux")
    print("   venv\\Scripts\\activate     # On Windows")
    print("\n4. Then run this script again:")
    print("   python scripts/setup_rag.py")
    print("=" * 70)
    sys.exit(1)

import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_literature(file_path: Path):
    """
    Load literature corpus from JSON file.
    
    Args:
        file_path: Path to the JSON file containing literature chunks.
    
    Returns:
        Tuple of (texts, metadatas, chunk_ids) lists.
    
    Raises:
        FileNotFoundError: If file doesn't exist.
        ValueError: If file format is invalid.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Literature file not found: {file_path}")
    
    logger.info(f"Loading literature from: {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            raise ValueError("JSON file must contain a list of chunks")
        
        texts = []
        metadatas = []
        chunk_ids = []
        
        for entry in data:
            if 'chunk_id' not in entry or 'text' not in entry:
                logger.warning(f"Skipping invalid entry: missing chunk_id or text")
                continue
            
            chunk_ids.append(entry['chunk_id'])
            texts.append(entry['text'])
            metadatas.append(entry.get('metadata', {}))
        
        logger.info(f"Loaded {len(texts)} chunks from literature file")
        return texts, metadatas, chunk_ids
        
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format: {e}")
    except Exception as e:
        raise Exception(f"Error loading literature file: {e}")


def main():
    """Main function to initialize the RAG system."""
    print("=" * 70)
    print("RAG SYSTEM INITIALIZATION")
    print("=" * 70)
    
    # Define paths
    project_root = Path(__file__).parent.parent
    literature_path = project_root / "data" / "literature" / "rehab_papers_chunked.json"
    vector_db_path = project_root / "data" / "vector_db"
    
    try:
        # Step 1: Load literature corpus
        print("\n[1/7] Loading literature corpus...")
        texts, metadatas, chunk_ids = load_literature(literature_path)
        print(f"Loaded {len(texts)} chunks")
        
        if len(texts) == 0:
            print("No chunks found in literature file")
            return 1
        
        # Step 2: Initialize EmbeddingGenerator
        print("\n[2/7] Initializing EmbeddingGenerator...")
        embedder = EmbeddingGenerator(model_name='all-MiniLM-L6-v2')
        print(f"EmbeddingGenerator initialized")
        print(f"Embedding dimension: {embedder.embedding_dim}")
        
        # Step 3: Generate embeddings for all chunks
        print("\n[3/7] Generating embeddings for all chunks...")
        print("   This may take a while depending on the number of chunks...")
        embeddings = embedder.embed_chunks(texts, batch_size=32)
        print(f"Generated embeddings: shape {embeddings.shape}")
        
        # Step 4: Initialize VectorStore
        print("\n[4/7] Initializing VectorStore...")
        store = VectorStore(persist_directory=str(vector_db_path))
        print(f"VectorStore initialized")
        print(f"Database location: {vector_db_path.absolute()}")
        
        # Step 5: Create collection
        print("\n[5/7] Creating collection...")
        store.create_collection(collection_name='rehab_literature')
        print(f"Collection 'rehab_literature' created/ready")
        
        # Step 6: Add all documents to collection
        print("\n[6/7] Adding documents to vector database...")
        num_added = store.add_documents(
            texts=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=chunk_ids,
            batch_size=100
        )
        print(f"Added {num_added} documents to collection")
        
        # Step 7: Print statistics
        print("\n[7/7] System Statistics:")
        print("-" * 70)
        total_count = store.count()
        print(f"   Total chunks in database: {total_count}")
        print(f"   Embedding dimensions: {embedder.embedding_dim}")
        print(f"   Database location: {vector_db_path.absolute()}")
        print(f"   Collection name: rehab_literature")
        
        # Print sample metadata
        if metadatas:
            print("\nSample metadata (first chunk):")
            sample_meta = metadatas[0]
            for key, value in sample_meta.items():
                print(f"     - {key}: {value}")
        
        # Test retrieval with sample query
        print("\n" + "=" * 70)
        print("TESTING RETRIEVAL")
        print("=" * 70)
        
        test_query = "exercises for asymmetry correction"
        print(f"\nQuery: '{test_query}'")
        print("\nGenerating query embedding...")
        
        query_embedding = embedder.embed_query(test_query)
        print(f"Query embedding generated: shape {query_embedding.shape}")
        
        print("\nSearching database for top 3 results...")
        results = store.search(query_embedding, n_results=3)
        
        if not results['ids']:
            print("No results found")
        else:
            print(f"\nFound {len(results['ids'])} results:")
            print("-" * 70)
            
            for i, (chunk_id, doc, metadata, distance) in enumerate(
                zip(
                    results['ids'],
                    results['documents'],
                    results['metadatas'],
                    results['distances']
                ),
                1
            ):
                print(f"\n   Result {i}:")
                print(f"   ID: {chunk_id}")
                print(f"   Distance: {distance:.4f} (lower = more similar)")
                print(f"   Metadata: {metadata}")
                print(f"   Text preview: {doc[:150]}...")
        
        print("\n" + "=" * 70)
        print("RAG SYSTEM INITIALIZATION COMPLETE!")
        print("=" * 70)
        print(f"\nThe vector database is ready at: {vector_db_path.absolute()}")
        print("You can now use the RAG system for retrieval-augmented generation.")
        
        return 0
        
    except FileNotFoundError as e:
        print(f"\nFile not found: {e}")
        print("\nInstructions:")
        print(f"1. Ensure the literature file exists at: {literature_path}")
        print("2. The file should be a JSON array of objects with:")
        print("      - chunk_id: unique identifier")
        print("      - text: the text content")
        print("      - metadata: dictionary with metadata")
        return 1
        
    except ValueError as e:
        print(f"\nValueError: {e}")
        return 1
        
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        logger.exception("Setup error:")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

