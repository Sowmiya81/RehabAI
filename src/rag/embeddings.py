"""
Embedding generation module using sentence-transformers.

This module provides the EmbeddingGenerator class for generating text embeddings
using pre-trained transformer models, suitable for RAG (Retrieval-Augmented Generation)
applications.
"""

from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """
    A class for generating text embeddings using sentence-transformers.
    
    This class provides efficient batch processing for generating embeddings
    from text chunks and single queries. The model is loaded once and cached
    for reuse across multiple embedding operations.
    
    Attributes:
        model: The loaded SentenceTransformer model
        model_name: Name of the model being used
        embedding_dim: Dimension of the generated embeddings
    """
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize the EmbeddingGenerator with a sentence-transformers model.
        
        The model is loaded once during initialization and cached for reuse.
        The default model 'all-MiniLM-L6-v2' provides a good balance between
        speed and quality, generating 384-dimensional embeddings.
        
        Args:
            model_name: Name of the sentence-transformers model to use.
                       Defaults to 'all-MiniLM-L6-v2' (384-dim, fast, good quality).
        
        Raises:
            Exception: If the model cannot be loaded.
        """
        self.model_name = model_name
        logger.info(f"Loading sentence-transformers model: {model_name}")
        
        try:
            self.model = SentenceTransformer(model_name)
            # Get embedding dimension from model
            # Create a dummy embedding to determine dimension
            test_embedding = self.model.encode("test", convert_to_numpy=True)
            self.embedding_dim = test_embedding.shape[0]
            
            logger.info(
                f"Successfully loaded model '{model_name}' "
                f"with embedding dimension: {self.embedding_dim}"
            )
        except Exception as e:
            error_msg = f"Failed to load model '{model_name}': {str(e)}"
            logger.error(error_msg)
            raise Exception(error_msg) from e
    
    def embed_chunks(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Generate embeddings for a list of text chunks in batches.
        
        Processes texts in batches for memory efficiency, especially useful
        when dealing with large numbers of text chunks. Shows progress with
        a tqdm progress bar.
        
        Args:
            texts: List of text strings to embed.
            batch_size: Number of texts to process in each batch. Defaults to 32.
                       Larger batches are faster but use more memory.
        
        Returns:
            Numpy array of shape (num_texts, embedding_dim) containing embeddings
            for each input text. For default model, shape is (num_texts, 384).
        
        Raises:
            ValueError: If texts list is empty or None.
        """
        if not texts:
            error_msg = "Cannot embed empty list of texts"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        if texts is None:
            error_msg = "Texts cannot be None"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Filter out empty strings
        non_empty_texts = [text for text in texts if text and text.strip()]
        if not non_empty_texts:
            error_msg = "All texts are empty after filtering"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        logger.info(f"Generating embeddings for {len(non_empty_texts)} text chunks")
        
        try:
            # Generate embeddings in batches with progress bar
            embeddings = self.model.encode(
                non_empty_texts,
                batch_size=batch_size,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=False  # Keep raw embeddings
            )
            
            logger.info(
                f"Successfully generated embeddings: shape {embeddings.shape}"
            )
            
            return embeddings
            
        except Exception as e:
            error_msg = f"Error generating embeddings: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        Generate embedding for a single query string.
        
        This method is optimized for embedding single queries, typically used
        for retrieval operations in RAG systems.
        
        Args:
            query: Single query string to embed.
        
        Returns:
            Numpy array of shape (embedding_dim,) containing the embedding.
            For default model, shape is (384,).
        
        Raises:
            ValueError: If query is empty or None.
        """
        if not query or not query.strip():
            error_msg = "Query cannot be empty"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        try:
            embedding = self.model.encode(
                query,
                convert_to_numpy=True,
                normalize_embeddings=False,
                show_progress_bar=False
            )
            
            return embedding
            
        except Exception as e:
            error_msg = f"Error embedding query: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise

