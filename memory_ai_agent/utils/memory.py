import json
import os
import re
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

class Memory:
    def __init__(self, memory_file: str = "memory.json", memory_limit: int = 10):
        """Initialize memory system.
        
        Args:
            memory_file: File to store memory
            memory_limit: Maximum number of conversations to remember
        """
        self.memory_file = memory_file
        self.memory_limit = memory_limit
        self.memory = self._load_memory()
        
    def _load_memory(self) -> Dict[str, Any]:
        """Load memory from file."""
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, "r") as f:
                    return json.load(f)
            except json.JSONDecodeError:
                return {"conversations": [], "user_info": {}}
        return {"conversations": [], "user_info": {}}
    
    def _save_memory(self) -> None:
        """Save memory to file."""
        with open(self.memory_file, "w") as f:
            json.dump(self.memory, f, indent=2)
            
    def add_conversation(self, user_input: str, ai_response: str) -> None:
        """Add a conversation to memory.
        
        Args:
            user_input: User's input
            ai_response: AI's response
        """
        conversation = {
            "timestamp": datetime.now().isoformat(),
            "user_input": user_input,
            "ai_response": ai_response
        }
        
        self.memory["conversations"].append(conversation)
        
        # Limit the number of conversations
        if len(self.memory["conversations"]) > self.memory_limit:
            self.memory["conversations"] = self.memory["conversations"][-self.memory_limit:]
            
        self._save_memory()
        
    def add_user_info(self, key: str, value: Any) -> None:
        """Add or update user information.
        
        Args:
            key: Information key
            value: Information value
        """
        self.memory["user_info"][key] = value
        self._save_memory()
        
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get conversation history."""
        return self.memory["conversations"]
    
    def get_user_info(self) -> Dict[str, Any]:
        """Get user information."""
        return self.memory["user_info"]
    
    def get_formatted_history(self) -> str:
        """Get formatted conversation history for context."""
        if not self.memory["conversations"]:
            return ""
        
        formatted = "Previous conversations:\n"
        for i, conv in enumerate(self.memory["conversations"][-5:]):  # Last 5 conversations
            timestamp = datetime.fromisoformat(conv['timestamp']).strftime("%Y-%m-%d %H:%M")
            formatted += f"[{timestamp}] User: {conv['user_input']}\n"
            formatted += f"[{timestamp}] AI: {conv['ai_response']}\n\n"
            
        return formatted
    
    def find_relevant_memories(self, query: str, threshold: float = 0.3) -> List[Dict[str, Any]]:
        """Find memories relevant to the query using simple keyword matching.
        
        Args:
            query: The user query
            threshold: Minimum relevance score threshold
            
        Returns:
            List of relevant conversation entries
        """
        if not self.memory["conversations"]:
            return []
            
        # Extract keywords from the query (simple implementation)
        # Remove common words and punctuation
        stop_words = {"a", "an", "the", "is", "are", "was", "were", "be", "been", 
                      "being", "to", "of", "and", "in", "that", "have", "with", 
                      "for", "not", "on", "at", "this", "but", "by", "from"}
        
        # Simple tokenization and filtering
        query_lower = query.lower()
        query_words = re.findall(r'\b\w+\b', query_lower)
        keywords = [word for word in query_words if word not in stop_words and len(word) > 2]
        
        # Check relevance with memory items
        relevant_memories = []
        
        for conv in self.memory["conversations"]:
            # Check relevance in both user input and AI response
            combined_text = (conv["user_input"] + " " + conv["ai_response"]).lower()
            
            # Count keyword matches
            match_count = sum(1 for keyword in keywords if keyword in combined_text)
            
            # Calculate relevance score
            relevance_score = match_count / len(keywords) if keywords else 0
            
            if relevance_score >= threshold:
                relevant_memory = {
                    "timestamp": conv["timestamp"],
                    "content": f"User: {conv['user_input']}\nAI: {conv['ai_response']}",
                    "relevance": relevance_score
                }
                relevant_memories.append(relevant_memory)
                
        # Sort by relevance score, highest first
        relevant_memories.sort(key=lambda x: x["relevance"], reverse=True)
        
        return relevant_memories[:3]  # Return top 3 most relevant
        
    def get_relevant_context(self, query: str) -> str:
        """Get relevant context from memory for the query.
        
        Args:
            query: The user query
            
        Returns:
            Formatted relevant context
        """
        relevant_memories = self.find_relevant_memories(query)
        
        if not relevant_memories:
            return ""
            
        context = "Relevant information from memory:\n\n"
        
        for memory in relevant_memories:
            timestamp = datetime.fromisoformat(memory["timestamp"]).strftime("%Y-%m-%d %H:%M")
            context += f"[{timestamp}]\n{memory['content']}\n\n"
            
        return context