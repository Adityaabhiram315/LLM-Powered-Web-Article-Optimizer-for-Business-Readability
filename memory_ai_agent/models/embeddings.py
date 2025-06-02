import numpy as np
import logging
from typing import Union
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variable
_embedding_dimension = 384

def get_embeddings(text: str) -> np.ndarray:
    """Generate embeddings for text using either a model or fallback hash-based method.

    Args:
        text: Input text to embed

    Returns:
        A NumPy ndarray representing the embedding vector.
    """
    if not text:
        return np.zeros(_embedding_dimension)

    try:
        # Try to use sentence-transformers if available
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer('all-MiniLM-L6-v2')
            embedding = model.encode(text)

            # Ensure result is a NumPy array
            if not isinstance(embedding, np.ndarray):
                embedding = np.array(embedding)

            return embedding
        except ImportError:
            # Fall back to simple hashing method
            logger.warning("sentence-transformers not found, using hash-based embedding.")
            return _simple_hash_embedding(text)
    except Exception as e:
        logger.error(f"Error generating embeddings: {str(e)}")
        return _simple_hash_embedding(text)

def _simple_hash_embedding(text: str) -> np.ndarray:
    """Fallback embedding generator using text hashing."""
    result = np.zeros(_embedding_dimension)
    chunks = [text[i:i+10] for i in range(0, len(text), 10)]

    for i, chunk in enumerate(chunks):
        hash_val = hashlib.md5(chunk.encode()).hexdigest()
        for j in range(min(32, _embedding_dimension // 12)):
            val = int(hash_val[j:j+2], 16)
            idx = (i * 32 + j) % _embedding_dimension
            result[idx] = (val / 255.0) * 2 - 1  # Scale to [-1, 1]

    norm = np.linalg.norm(result)
    return result / norm if norm > 0 else result

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors."""
    if np.all(a == 0) or np.all(b == 0):
        return 0.0
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
