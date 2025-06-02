import json
import os
import re
from datetime import datetime
import logging
import time
import numpy as np
from typing import List, Dict, Any, Optional, Tuple

from utils.vectordb import VectorDatabase
from models.embeddings import get_embeddings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Flag to control whether to save DB after every addition
# Set to False for faster operation when adding many items
SAVE_IMMEDIATELY = False 

class MemoryVectorDB:
    """Memory system using vector database for storage and retrieval"""
    
    def __init__(self, 
                 memory_file: str = "memory.json", 
                 vector_db_path: str = "vector_db.pkl", 
                 memory_limit: int = 100):
        """Initialize memory system with vector database
        
        Args:
            memory_file: File to store user info (legacy)
            vector_db_path: Path to vector database file
            memory_limit: Maximum number of conversations to remember
        """
        self.memory_file = memory_file
        self.memory_limit = memory_limit
        self.vectordb = VectorDatabase(db_path=vector_db_path)
        
        # Load user info from legacy file (keep this for compatibility)
        self.user_info = self._load_user_info()
        
        # Import any existing conversations from legacy format
        self._import_legacy_conversations()
        
    def _load_user_info(self) -> Dict[str, Any]:
        """Load user info from legacy file"""
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, "r") as f:
                    data = json.load(f)
                    return data.get("user_info", {})
            except json.JSONDecodeError:
                return {}
        return {}
    
    def _save_user_info(self) -> None:
        """Save user info to legacy file"""
        with open(self.memory_file, "w") as f:
            json.dump({"user_info": self.user_info}, f, indent=2)
    
    def _import_legacy_conversations(self) -> None:
        """Import conversations from legacy JSON format"""
        if not os.path.exists(self.memory_file):
            return
            
        try:
            with open(self.memory_file, "r") as f:
                data = json.load(f)
                conversations = data.get("conversations", [])
                
            # Skip import if no conversations or if we already have vectors
            if not conversations or len(self.vectordb.get_all_items()) > 0:
                return
                
            logger.info(f"Importing {len(conversations)} conversations from legacy format")
            
            for conv in conversations:
                user_input = conv.get("user_input", "")
                ai_response = conv.get("ai_response", "")
                timestamp = conv.get("timestamp", datetime.now().isoformat())
                
                # Add to vector database
                try:
                    combined_text = f"User: {user_input}\nAI: {ai_response}"
                    vector = get_embeddings(combined_text)
                    
                    self.vectordb.add_item(
                        vector=vector,
                        metadata={
                            "user_input": user_input,
                            "ai_response": ai_response,
                            "timestamp": timestamp,
                            "thread_id": "default"
                        }
                    )
                except Exception as e:
                    logger.error(f"Error importing conversation: {str(e)}")
                    
            logger.info("Legacy conversation import complete")
        except Exception as e:
            logger.error(f"Error importing legacy conversations: {str(e)}")
    def add_conversation(self, 
                         user_input: str, 
                         ai_response: str, 
                         thread_id: str = "default",
                         save_immediately: bool = False) -> None:
        """Add a conversation to memory
        
        Args:
            user_input: User's input
            ai_response: AI's response
            thread_id: Thread identifier
            save_immediately: Whether to save immediately or batch with other operations
        """
        try:
            # Generate embedding for the conversation
            combined_text = f"User: {user_input}\nAI: {ai_response}"
            vector = get_embeddings(combined_text)
            
            # Add to vector database
            self.vectordb.add_item(
                vector=vector,
                metadata={
                    "user_input": user_input,
                    "ai_response": ai_response,
                    "timestamp": datetime.now().isoformat(),
                    "thread_id": thread_id
                },
                save_immediately=save_immediately
            )
            
            # Cleanup old conversations if exceeding limit (only when saving immediately)
            if save_immediately and len(self.vectordb.get_all_items()) > self.memory_limit:
                self._cleanup_old_conversations()
        except Exception as e:
            logger.error(f"Error adding conversation to memory: {str(e)}")
    
    def _cleanup_old_conversations(self) -> None:
        """Cleanup old conversations if exceeding limit"""
        all_items = self.vectordb.get_all_items()
        
        if len(all_items) <= self.memory_limit:
            return
            
        # Sort by timestamp
        all_items.sort(key=lambda x: x["metadata"].get("timestamp", ""), reverse=True)
        
        # Keep only the most recent ones
        items_to_keep = all_items[:self.memory_limit]
        items_to_delete = all_items[self.memory_limit:]
        
        # Delete old items
        for item in items_to_delete:
            self.vectordb.delete_item(item["id"])
            
        logger.info(f"Cleaned up {len(items_to_delete)} old conversations")
    
    def add_user_info(self, key: str, value: Any) -> None:
        """Add or update user information
        
        Args:
            key: Information key
            value: Information value
        """
        self.user_info[key] = value
        self._save_user_info()
    
    def get_user_info(self) -> Dict[str, Any]:
        """Get user information"""
        return self.user_info
    
    def get_conversation_history(self, thread_id: str = "default", limit: int = 10) -> List[Dict[str, Any]]:
        """Get conversation history for a thread
        
        Args:
            thread_id: Thread identifier
            limit: Maximum number of conversations to return
            
        Returns:
            List of conversation entries
        """
        all_items = self.vectordb.get_all_items()
        
        # Filter by thread_id
        thread_items = [
            item for item in all_items 
            if item["metadata"].get("thread_id") == thread_id
        ]
        
        # Sort by timestamp (newest first)
        thread_items.sort(
            key=lambda x: x["metadata"].get("timestamp", ""), 
            reverse=True
        )
        
        # Convert to the expected format
        conversations = []
        for item in thread_items[:limit]:
            metadata = item["metadata"]
            conversations.append({
                "timestamp": metadata.get("timestamp", ""),
                "user_input": metadata.get("user_input", ""),
                "ai_response": metadata.get("ai_response", "")
            })
            
        return conversations
    
    def get_formatted_history(self, thread_id: str = "default", limit: int = 5) -> str:
        """Get formatted conversation history for context
        
        Args:
            thread_id: Thread identifier
            limit: Maximum number of conversations to include
            
        Returns:
            Formatted conversation history
        """
        conversations = self.get_conversation_history(thread_id, limit)
        
        if not conversations:
            return ""
        
        formatted = "Previous conversations:\n"
        
        # Show newest conversations last (chronological order)
        for conv in reversed(conversations):
            timestamp = datetime.fromisoformat(conv["timestamp"]).strftime("%Y-%m-%d %H:%M")
            formatted += f"[{timestamp}] User: {conv['user_input']}\n"
            formatted += f"[{timestamp}] AI: {conv['ai_response']}\n\n"
            
        return formatted
    
    def find_relevant_memories_semantic(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Find memories relevant to the query using semantic search
        
        Args:
            query: The user query
            top_k: Number of results to return
            
        Returns:
            List of relevant conversation entries
        """
        try:
            # Generate embedding for the query
            query_vector = get_embeddings(query)
            
            # Search vector database
            results = self.vectordb.search(query_vector, top_k=top_k)
            
            # Convert to the expected format
            relevant_memories = []
            for result in results:
                metadata = result["metadata"]
                relevant_memories.append({
                    "timestamp": metadata.get("timestamp", ""),
                    "content": f"User: {metadata.get('user_input', '')}\nAI: {metadata.get('ai_response', '')}",
                    "relevance": result["similarity"]
                })
                
            return relevant_memories
        except Exception as e:
            logger.error(f"Error finding relevant memories: {str(e)}")
            return []
    
    def find_relevant_memories_keywords(self, query: str, threshold: float = 0.3) -> List[Dict[str, Any]]:
        """Find memories relevant to the query using keyword matching
        
        Args:
            query: The user query
            threshold: Minimum relevance score threshold
            
        Returns:
            List of relevant conversation entries
        """
        all_items = self.vectordb.get_all_items()
        
        if not all_items:
            return []
        
        # Extract keywords from the query
        stop_words = {"a", "an", "the", "is", "are", "was", "were", "be", "been", 
                      "being", "to", "of", "and", "in", "that", "have", "with", 
                      "for", "not", "on", "at", "this", "but", "by", "from"}
        
        # Simple tokenization and filtering
        query_lower = query.lower()
        query_words = re.findall(r'\b\w+\b', query_lower)
        keywords = [word for word in query_words if word not in stop_words and len(word) > 2]
        
        # Check relevance with memory items
        relevant_memories = []
        
        for item in all_items:
            metadata = item["metadata"]
            user_input = metadata.get("user_input", "")
            ai_response = metadata.get("ai_response", "")
            
            # Check relevance in both user input and AI response
            combined_text = (user_input + " " + ai_response).lower()
            
            # Count keyword matches
            match_count = sum(1 for keyword in keywords if keyword in combined_text)
            
            # Calculate relevance score
            relevance_score = match_count / len(keywords) if keywords else 0
            
            if relevance_score >= threshold:
                relevant_memory = {
                    "timestamp": metadata.get("timestamp", ""),
                    "content": f"User: {user_input}\nAI: {ai_response}",
                    "relevance": relevance_score
                }
                relevant_memories.append(relevant_memory)
        
        # Sort by relevance score, highest first
        relevant_memories.sort(key=lambda x: x["relevance"], reverse=True)
        
        return relevant_memories[:3]  # Return top 3 most relevant
    
    def get_relevant_context(self, query: str, use_semantic: bool = True) -> str:
        """Get relevant context from memory for the query
        
        Args:
            query: The user query
            use_semantic: Whether to use semantic search
            
        Returns:
            Formatted relevant context
        """
        # Try semantic search first, fall back to keyword search
        if use_semantic:
            try:
                relevant_memories = self.find_relevant_memories_semantic(query)
                
                # If semantic search fails or returns nothing, fall back to keyword search
                if not relevant_memories:
                    relevant_memories = self.find_relevant_memories_keywords(query)
            except Exception:
                relevant_memories = self.find_relevant_memories_keywords(query)
        else:
            relevant_memories = self.find_relevant_memories_keywords(query)
        
        if not relevant_memories:
            return ""
            
        context = "Relevant information from memory:\n\n"
        
        for memory in relevant_memories:
            timestamp = datetime.fromisoformat(memory["timestamp"]).strftime("%Y-%m-%d %H:%M")
            context += f"[{timestamp}]\n{memory['content']}\n\n"
            
        return context
    
    def list_threads(self) -> Dict[str, Dict[str, Any]]:
        """List all conversation threads
        
        Returns:
            Dictionary of thread_id -> thread info
        """
        all_items = self.vectordb.get_all_items()
        
        threads = {}
        for item in all_items:
            metadata = item["metadata"]
            thread_id = metadata.get("thread_id", "default")
            
            if thread_id not in threads:
                # Create new thread entry
                threads[thread_id] = {
                    "name": thread_id,
                    "count": 1,
                    "last_updated": metadata.get("timestamp", "")
                }
            else:
                # Update existing thread
                threads[thread_id]["count"] += 1
                
                # Update last_updated if this conversation is newer
                current_ts = threads[thread_id]["last_updated"]
                new_ts = metadata.get("timestamp", "")
                
                if new_ts > current_ts:
                    threads[thread_id]["last_updated"] = new_ts
        
        return threads
    
    def clear_memory(self) -> None:
        """Clear all conversation memory"""
        self.vectordb.clear()
        logger.info("Cleared all conversation memory")

# Use this as the new Memory class
Memory = MemoryVectorDB