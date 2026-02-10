"""
Retriever module for hybrid embedding and vector search.

This module provides the HybridRetriever class that combines embedding generation
with vector store search, providing a convenient interface for RAG retrieval operations.
"""

from typing import List, Dict, Optional
from src.rag.embeddings import EmbeddingGenerator
from src.rag.vector_store import VectorStore
import logging

logger = logging.getLogger(__name__)


class HybridRetriever:
    """
    A hybrid retriever that combines embedding generation and vector search.
    
    This class provides a high-level interface for retrieving relevant documents
    from a vector store using natural language queries. It handles embedding
    generation, metadata filtering, and result formatting.
    
    Attributes:
        embedder: EmbeddingGenerator instance for creating query embeddings
        vector_store: VectorStore instance for searching the vector database
    """
    
    def __init__(
        self,
        embedder: EmbeddingGenerator,
        vector_store: VectorStore
    ):
        """
        Initialize the HybridRetriever with embedding generator and vector store.
        
        Args:
            embedder: EmbeddingGenerator instance for generating query embeddings.
            vector_store: VectorStore instance for searching the vector database.
        
        Raises:
            ValueError: If embedder or vector_store is None.
        """
        if embedder is None:
            raise ValueError("EmbeddingGenerator cannot be None")
        if vector_store is None:
            raise ValueError("VectorStore cannot be None")
        
        self.embedder = embedder
        self.vector_store = vector_store
        
        logger.info("HybridRetriever initialized")
    
    def _format_results(self, raw_results: Dict[str, List]) -> List[Dict]:
        """
        Convert ChromaDB raw results format to clean dictionary format.
        
        Converts distance to relevance score (1 - distance) and formats
        results as a list of dictionaries with chunk_id, text, metadata,
        and relevance_score.
        
        Args:
            raw_results: Dictionary from VectorStore.search() with keys:
                        'ids', 'documents', 'metadatas', 'distances'
        
        Returns:
            List of formatted result dictionaries, sorted by relevance_score
            (highest first). Each dict contains:
            - chunk_id: str
            - text: str
            - metadata: dict
            - relevance_score: float (0-1, higher is more relevant)
        """
        if not raw_results or not raw_results.get('ids'):
            return []
        
        formatted_results = []
        
        for i, chunk_id in enumerate(raw_results['ids']):
            # Get corresponding values
            text = raw_results['documents'][i] if i < len(raw_results['documents']) else ""
            metadata = raw_results['metadatas'][i] if i < len(raw_results['metadatas']) else {}
            distance = raw_results['distances'][i] if i < len(raw_results['distances']) else 1.0
            
            # Convert distance to relevance score (lower distance = higher relevance)
            # Distance is typically 0-2 for cosine distance, normalize to 0-1
            # For cosine: distance = 1 - cosine_similarity, so relevance = cosine_similarity
            relevance_score = max(0.0, min(1.0, 1.0 - distance))
            
            formatted_results.append({
                "chunk_id": chunk_id,
                "text": text,
                "metadata": metadata,
                "relevance_score": round(relevance_score, 4)
            })
        
        # Sort by relevance_score (highest first)
        formatted_results.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        return formatted_results
    
    def search(
        self,
        query: str,
        exercise_type: Optional[str] = None,
        issue_type: Optional[str] = None,
        n_results: int = 5
    ) -> List[Dict]:
        """
        Search the vector store with a natural language query.
        
        Embeds the query string, applies optional metadata filters, and returns
        formatted results sorted by relevance score.
        
        Args:
            query: Natural language query string.
            exercise_type: Optional filter for exercise_type metadata field.
                          Filters using "$contains" operator to match comma-separated values.
            issue_type: Optional filter for issue_addressed metadata field.
                       Filters using "$contains" operator to match comma-separated values.
            n_results: Number of results to return. Defaults to 5.
        
        Returns:
            List of result dictionaries, sorted by relevance_score (highest first).
            Each dict contains:
            - chunk_id: str
            - text: str
            - metadata: dict
            - relevance_score: float (0-1, higher is more relevant)
        
        Raises:
            ValueError: If query is empty or invalid.
            RuntimeError: If vector store collection is not initialized.
        """
        if not query or not query.strip():
            error_msg = "Query cannot be empty"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        logger.info(f"Searching with query: '{query[:50]}...' (n_results={n_results})")
        
        try:
            # Generate query embedding
            query_embedding = self.embedder.embed_query(query)
            logger.debug(f"Query embedding generated: shape {query_embedding.shape}")
            
            # Build metadata filter
            where_filter = None
            if exercise_type or issue_type:
                where_filter = {}
                
                if exercise_type:
                    # Use $contains to match values in comma-separated strings
                    where_filter["exercise_type"] = {"$contains": exercise_type}
                    logger.debug(f"Filtering by exercise_type: {exercise_type}")
                
                if issue_type:
                    # Use $contains to match values in comma-separated strings
                    where_filter["issue_addressed"] = {"$contains": issue_type}
                    logger.debug(f"Filtering by issue_addressed: {issue_type}")
            
            # Search vector store
            raw_results = self.vector_store.search(
                query_embedding=query_embedding,
                n_results=n_results,
                where_filter=where_filter
            )
            
            # Format results
            formatted_results = self._format_results(raw_results)
            
            logger.info(f"Found {len(formatted_results)} results")
            return formatted_results
            
        except ValueError as e:
            logger.error(f"ValueError in search: {e}")
            raise
        except Exception as e:
            error_msg = f"Error during search: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e
    
    def batch_search(
        self,
        queries: List[str],
        exercise_type: Optional[str] = None,
        issue_type: Optional[str] = None,
        n_results: int = 5
    ) -> Dict[str, List[Dict]]:
        """
        Execute multiple searches in batch.
        
        Useful for agents that generate multiple queries or when searching
        with different query variations. Each query is processed independently.
        
        Args:
            queries: List of query strings to search.
            exercise_type: Optional filter for exercise_type metadata field.
            issue_type: Optional filter for issue_addressed metadata field.
            n_results: Number of results to return per query. Defaults to 5.
        
        Returns:
            Dictionary mapping each query string to its list of results.
            Format: {query: [result_dict, ...], ...}
        
        Raises:
            ValueError: If queries list is empty or contains invalid queries.
        """
        if not queries:
            error_msg = "Queries list cannot be empty"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        if not isinstance(queries, list):
            error_msg = "Queries must be a list"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        logger.info(f"Executing batch search for {len(queries)} queries")
        
        results_dict = {}
        
        for query in queries:
            try:
                results = self.search(
                    query=query,
                    exercise_type=exercise_type,
                    issue_type=issue_type,
                    n_results=n_results
                )
                results_dict[query] = results
            except Exception as e:
                logger.warning(f"Error processing query '{query}': {e}")
                # Store empty results for failed queries
                results_dict[query] = []
        
        logger.info(f"Batch search completed: {len(results_dict)} queries processed")
        return results_dict

