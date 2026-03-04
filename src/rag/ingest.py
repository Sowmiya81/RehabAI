import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def ingest_literature(store, embedder, literature_path: str = None):
    """Build ChromaDB collection from JSON corpus. Called at startup if DB is empty."""
    
    if literature_path is None:
        literature_path = Path(__file__).parent.parent.parent / "data" / "literature" / "rehab_papers_chunked.json"
    
    literature_path = Path(literature_path)
    
    if not literature_path.exists():
        logger.error(f"Literature file not found: {literature_path}")
        return 0
    
    logger.info(f"Starting ingestion from {literature_path}")
    
    with open(literature_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    texts, metadatas, chunk_ids = [], [], []
    for entry in data:
        if 'chunk_id' not in entry or 'text' not in entry:
            continue
        chunk_ids.append(entry['chunk_id'])
        texts.append(entry['text'])
        metadatas.append(entry.get('metadata', {}))
    
    logger.info(f"Loaded {len(texts)} chunks, generating embeddings...")
    embeddings = embedder.embed_chunks(texts, batch_size=32)
    
    store.create_collection(collection_name='rehab_literature')
    num_added = store.add_documents(
        texts=texts,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=chunk_ids,
        batch_size=100
    )
    
    logger.info(f"Ingestion complete: {num_added} documents added")
    return num_added