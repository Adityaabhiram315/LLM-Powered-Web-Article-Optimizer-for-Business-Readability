import chromadb
from chromadb.config import Settings
import numpy as np
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime
import os
import shutil
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorDatabase:
    """Vector database implementation using ChromaDB"""
    
    def __init__(self, db_path: str = "./data/chroma_db"):
        """Initialize ChromaDB
        
        Args:
            db_path: Path to store the ChromaDB files
        """
        try:            # Create base directory for database if it doesn't exist
            db_path = os.path.abspath(db_path)
            db_dir = os.path.dirname(db_path) if '.' in os.path.basename(db_path) else db_path
            
            if not os.path.exists(db_dir):
                os.makedirs(db_dir, exist_ok=True)
            
            # Remove existing database if it's invalid
            if os.path.exists(db_path) and not os.path.isdir(db_path):
                try:
                    os.remove(db_path)
                except Exception as e:
                    logger.warning(f"Could not remove invalid database file: {e}")
            
            # Initialize ChromaDB client
            self.client = chromadb.PersistentClient(path=db_path)
            
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name="conversations",
                metadata={"description": "AI Agent conversation memory"}
            )
            
        except Exception as e:
            logger.error(f"Error initializing ChromaDB: {str(e)}")
            raise
        
    def add_item(self, vector: np.ndarray, metadata: Dict[str, Any], save_immediately: bool = True) -> str:
        """Add or update an item in the database
        
        Args:
            vector: The embedding vector
            metadata: Associated metadata
            save_immediately: Not used (ChromaDB saves automatically)
        """
        try:
            # Check if content already exists
            content = f"{metadata.get('user_input', '')}|{metadata.get('ai_response', '')}"
            doc_id = str(hash(content))
            
            # Try to get existing item
            existing = self.collection.get(ids=[doc_id])
            if existing and existing.get('ids'):
                # Update existing item
                self.collection.update(
                    ids=[doc_id],
                    embeddings=[vector.tolist()],
                    documents=[content],
                    metadatas=[{**metadata, 'updated_at': datetime.now().isoformat()}]
                )
                logger.info(f"Updated existing item: {doc_id}")
            else:
                # Add new item
                self.collection.add(
                    ids=[doc_id],
                    embeddings=[vector.tolist()],
                    documents=[content],
                    metadatas=[{**metadata, 'created_at': datetime.now().isoformat()}]
                )
                logger.info(f"Added new item: {doc_id}")
            
            return doc_id
            
        except Exception as e:
            logger.error(f"Error adding/updating item: {str(e)}")
            raise
        
    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar vectors
        
        Args:
            query_vector: Query embedding
            top_k: Number of results to return
        """
        try:
            results = self.collection.query(
                query_embeddings=[query_vector.tolist()],
                n_results=top_k,
                include=['metadatas', 'distances', 'documents']
            )
            
            formatted_results = []
            
            # Check if we have valid results
            if results and isinstance(results, dict) and results.get('ids'):
                ids = results['ids'][0] if results['ids'] else []
                metadatas = results.get('metadatas', [])
                documents = results.get('documents', [])
                distances = results.get('distances', [])
                
                # Get first items if they exist
                if metadatas: metadatas = metadatas[0]
                if documents: documents = documents[0]
                if distances: distances = distances[0]
                
                # Iterate through results
                for i in range(len(ids)):
                    result = {
                        'id': ids[i],
                        'metadata': {},
                        'content': "",
                        'similarity': 0.0
                    }
                    
                    # Add optional fields if available
                    if metadatas and i < len(metadatas):
                        result['metadata'] = metadatas[i]
                    if documents and i < len(documents):
                        result['content'] = documents[i]
                    if distances is not None and i < len(distances):
                        try:
                            dist = distances[i]
                            if isinstance(dist, (int, float)):
                                result['similarity'] = 1 - float(dist)
                            elif isinstance(dist, list) and len(dist) > 0:
                                result['similarity'] = 1 - float(dist[0])
                        except Exception:
                            result['similarity'] = 0.0

                    formatted_results.append(result)
                    
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error searching: {str(e)}")
            return []
    
    def get_all_items(self) -> List[Dict[str, Any]]:
        """Get all items in the database"""
        try:
            results = self.collection.get(
                include=['metadatas', 'documents']
            )
            
            items = []
            
            # Check if we have valid results
            if results and isinstance(results, dict) and results.get('ids'):
                ids = results['ids']
                metadatas = results.get('metadatas', [])
                documents = results.get('documents', [])
                
                # Iterate through results
                for i in range(len(ids)):
                    item = {
                        'id': ids[i],
                        'metadata': {},
                        'content': ""
                    }
                    
                    # Add optional fields if available
                    if metadatas and i < len(metadatas):
                        item['metadata'] = metadatas[i]
                    if documents and i < len(documents):
                        item['content'] = documents[i]
                        
                    items.append(item)
            
            return items
            
        except Exception as e:
            logger.error(f"Error getting all items: {str(e)}")
            return []
        
    def delete_item(self, item_id: str) -> bool:
        """Delete an item from the database"""
        try:
            self.collection.delete(ids=[item_id])
            return True
        except Exception as e:
            logger.error(f"Error deleting item: {str(e)}")
            return False
            
    def clear(self) -> None:
        """Clear the entire database"""
        try:
            self.client.delete_collection("conversations")
            self.collection = self.client.create_collection(
                name="conversations",
                metadata={"description": "AI Agent conversation memory"}
            )
        except Exception as e:
            logger.error(f"Error clearing database: {str(e)}")
            raise