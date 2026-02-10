"""
Vector store module using ChromaDB for persistent vector storage and retrieval.

This module provides the VectorStore class for managing document embeddings,
enabling efficient similarity search and retrieval for RAG applications.
"""

import chromadb
from chromadb.config import Settings
import numpy as np
from typing import List, Dict, Optional
from tqdm import tqdm
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class VectorStore:
    """
    A wrapper class for ChromaDB vector storage and retrieval.
    
    This class provides a convenient interface for storing document embeddings
    and performing similarity searches. It uses persistent storage to maintain
    data across sessions.
    
    Attributes:
        client: ChromaDB persistent client
        persist_directory: Directory where ChromaDB data is stored
        collection: Current collection being used
    """
    
    def __init__(self, persist_directory: str = './data/vector_db'):
        """
        Initialize the VectorStore with a ChromaDB persistent client.
        
        Creates a persistent ChromaDB client that stores data on disk,
        allowing data to persist across sessions. Uses cosine similarity
        as the default distance metric.
        
        Args:
            persist_directory: Directory path where ChromaDB will store data.
                             Defaults to './data/vector_db'.
                             Directory will be created if it doesn't exist.
        """
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initializing ChromaDB client at: {self.persist_directory}")
        
        try:
            self.client = chromadb.PersistentClient(
                path=str(self.persist_directory),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            logger.info("ChromaDB client initialized successfully")
            self.collection = None
        except Exception as e:
            error_msg = f"Failed to initialize ChromaDB client: {str(e)}"
            logger.error(error_msg)
            raise Exception(error_msg) from e
    
    def create_collection(self, collection_name: str = 'rehab_literature'):
        """
        Create or get an existing collection.
        
        Creates a new collection if it doesn't exist, or returns the existing
        one. Configures the collection to use cosine similarity for distance
        calculations.
        
        Args:
            collection_name: Name of the collection to create or get.
                           Defaults to 'rehab_literature'.
        
        Returns:
            ChromaDB collection object.
        """
        logger.info(f"Creating/getting collection: {collection_name}")
        
        try:
            # Get or create collection with cosine similarity
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            
            logger.info(f"Collection '{collection_name}' ready")
            return self.collection
            
        except Exception as e:
            error_msg = f"Failed to create/get collection '{collection_name}': {str(e)}"
            logger.error(error_msg)
            raise Exception(error_msg) from e
    
    def add_documents(
        self,
        texts: List[str],
        embeddings: np.ndarray,
        metadatas: List[dict],
        ids: List[str],
        batch_size: int = 100
    ) -> int:
        """
        Add documents to the collection in batches.
        
        Adds documents with their embeddings, metadata, and IDs to the collection.
        Processes documents in batches for efficiency and shows progress.
        
        Note: List values in metadata are automatically converted to comma-separated
        strings (e.g., ["squat", "lunge"] becomes "squat, lunge") since ChromaDB
        only supports str, int, float, bool, or None for metadata values.
        
        Args:
            texts: List of text strings (documents) to add.
            embeddings: Numpy array of embeddings with shape (num_docs, embedding_dim).
            metadatas: List of metadata dictionaries, one per document.
                      List values will be converted to comma-separated strings.
            ids: List of unique IDs for each document.
            batch_size: Number of documents to add per batch. Defaults to 100.
        
        Returns:
            Number of documents successfully added.
        
        Raises:
            ValueError: If inputs are invalid or mismatched in length.
            RuntimeError: If collection is not initialized.
        """
        if self.collection is None:
            error_msg = "Collection not initialized. Call create_collection() first."
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        # Validate inputs
        num_docs = len(texts)
        if num_docs == 0:
            error_msg = "Cannot add empty list of documents"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        if len(embeddings) != num_docs:
            error_msg = f"Mismatch: {num_docs} texts but {len(embeddings)} embeddings"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        if len(metadatas) != num_docs:
            error_msg = f"Mismatch: {num_docs} texts but {len(metadatas)} metadata entries"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        if len(ids) != num_docs:
            error_msg = f"Mismatch: {num_docs} texts but {len(ids)} IDs"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        logger.info(f"Adding {num_docs} documents to collection in batches of {batch_size}")
        
        # Convert embeddings to list format (ChromaDB expects list of lists)
        embeddings_list = embeddings.tolist()
        
        # Helper function to normalize metadata values for ChromaDB
        def normalize_metadata(metadata_dict):
            """
            Convert metadata values to ChromaDB-compatible types.
            Lists are converted to comma-separated strings.
            """
            normalized = {}
            for key, value in metadata_dict.items():
                if isinstance(value, list):
                    # Convert list to comma-separated string
                    normalized[key] = ", ".join(str(item) for item in value)
                elif isinstance(value, (str, int, float, bool)) or value is None:
                    # Already compatible types
                    normalized[key] = value
                else:
                    # Convert other types to string
                    normalized[key] = str(value)
            return normalized
        
        # Normalize all metadata
        normalized_metadatas = [normalize_metadata(meta) for meta in metadatas]
        
        try:
            # Add documents in batches with progress bar
            num_batches = (num_docs + batch_size - 1) // batch_size
            total_added = 0
            
            with tqdm(total=num_docs, desc="Adding documents") as pbar:
                for i in range(0, num_docs, batch_size):
                    batch_end = min(i + batch_size, num_docs)
                    
                    batch_texts = texts[i:batch_end]
                    batch_embeddings = embeddings_list[i:batch_end]
                    batch_metadatas = normalized_metadatas[i:batch_end]
                    batch_ids = ids[i:batch_end]
                    
                    # Add batch to collection
                    self.collection.add(
                        embeddings=batch_embeddings,
                        documents=batch_texts,
                        metadatas=batch_metadatas,
                        ids=batch_ids
                    )
                    
                    batch_size_actual = batch_end - i
                    total_added += batch_size_actual
                    pbar.update(batch_size_actual)
            
            logger.info(f"Successfully added {total_added} documents to collection")
            return total_added
            
        except Exception as e:
            error_msg = f"Error adding documents: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise
    
    def search(
        self,
        query_embedding: np.ndarray,
        n_results: int = 5,
        where_filter: Optional[dict] = None
    ) -> Dict[str, List]:
        """
        Search the collection using a query embedding.
        
        Performs similarity search using cosine distance. Returns the most
        similar documents along with their metadata and distances.
        
        Args:
            query_embedding: Query embedding as numpy array with shape (embedding_dim,).
            n_results: Number of results to return. Defaults to 5.
            where_filter: Optional metadata filter dictionary.
                        Example: {"issue_addressed": "asymmetry"}
                        Only documents matching the filter will be returned.
        
        Returns:
            Dictionary with keys:
            - 'ids': List of chunk IDs
            - 'documents': List of text chunks
            - 'metadatas': List of metadata dictionaries
            - 'distances': List of cosine distances (lower = more similar)
            
            Returns empty lists if no results found.
        
        Raises:
            RuntimeError: If collection is not initialized.
            ValueError: If query_embedding is invalid.
        """
        if self.collection is None:
            error_msg = "Collection not initialized. Call create_collection() first."
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        if query_embedding is None or query_embedding.size == 0:
            error_msg = "Invalid query embedding"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Convert embedding to list format
        query_embedding_list = query_embedding.tolist()
        
        try:
            logger.debug(f"Searching collection with n_results={n_results}, filter={where_filter}")
            
            # Perform search
            results = self.collection.query(
                query_embeddings=[query_embedding_list],
                n_results=n_results,
                where=where_filter
            )
            
            # Handle empty results
            if not results['ids'] or len(results['ids'][0]) == 0:
                logger.info("No results found for query")
                return {
                    'ids': [],
                    'documents': [],
                    'metadatas': [],
                    'distances': []
                }
            
            # ChromaDB returns results as lists of lists (for multiple queries)
            # Since we're querying with a single embedding, extract first element
            num_results = len(results['ids'][0])
            
            return {
                'ids': results['ids'][0],
                'documents': results['documents'][0],
                'metadatas': results['metadatas'][0] if results['metadatas'] else [{}] * num_results,
                'distances': results['distances'][0] if results['distances'] else []
            }
            
        except Exception as e:
            error_msg = f"Error searching collection: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise
    
    def count(self) -> int:
        """
        Get the total number of documents in the collection.
        
        Returns:
            Number of documents in the collection.
        
        Raises:
            RuntimeError: If collection is not initialized.
        """
        if self.collection is None:
            error_msg = "Collection not initialized. Call create_collection() first."
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        try:
            count = self.collection.count()
            logger.debug(f"Collection contains {count} documents")
            return count
        except Exception as e:
            error_msg = f"Error counting documents: {str(e)}"
            logger.error(error_msg)
            raise

