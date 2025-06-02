import requests
import json
import time
import re
import logging
from typing import Dict, Any, List, Optional, Tuple

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMInterface:
    def __init__(
        self, 
        api_key: Optional[str] = None, 
        site_url: Optional[str] = "http://localhost", 
        site_name: Optional[str] = "Memory AI Agent",
        default_model: Optional[str] = "gpt-3.5-turbo",
        available_models: Optional[Dict[str, str]] = None
    ):
        """Initialize LLM interface.
        
        Args:
            api_key: OpenRouter API key (optional)
            site_url: Site URL for OpenRouter
            site_name: Site name for OpenRouter
            default_model: Default model to use
            available_models: Available models
        """
        self.api_key = api_key or "demo"  # Use demo key as fallback
        self.site_url = site_url
        self.site_name = site_name
        self.default_model = default_model
        
        # Default models if none provided
        if available_models is None:
            self.available_models = {
                "gpt-3.5-turbo": "openai/gpt-3.5-turbo",
                "gpt-4": "openai/gpt-4",
                "claude": "anthropic/claude-instant-v1",
                "llama": "meta-llama/llama-2-70b-chat"
            }
        else:
            self.available_models = available_models
            
        # Validate API connection
        if self.api_key != "demo":
            self._test_connection()
        
    def _test_connection(self):
        """Test connection to API."""
        try:
            # Simple request to check if API key works
            response = requests.post(
                url="https://openrouter.ai/api/v1/auth/test",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
            )
            if response.status_code != 200:
                logger.warning(f"API key test failed with status {response.status_code}")
        except Exception as e:
            logger.warning(f"Could not connect to API: {str(e)}")
            # Continue anyway - will try again on actual calls
    
    def check_knowledge(
        self,
        user_input: str,
        system_prompt: str,
        conversation_history: str,
        model: Optional[str] = None
    ) -> Tuple[bool, str, float]:
        """Check if the model needs to search for information.
        
        Args:
            user_input: User's input
            system_prompt: System prompt for knowledge checking
            conversation_history: Conversation history
            model: Model to use, defaults to default_model
            
        Returns:
            Tuple of (needs_search, search_query, time_taken)
        """
        if not model:
            model = self.default_model

        # Ensure model is not None and is a string
        model_key = model if model is not None else self.default_model
        model_id = self.available_models.get(model_key, self.available_models[self.default_model])
        
        # Prepare messages
        messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_input
            }
        ]
        
        # Call API
        start_time = time.time()
        try:
            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": self.site_url,
                    "X-Title": self.site_name,
                },
                data=json.dumps({
                    "model": model_id,
                    "messages": messages
                })
            )
            
            response_json = response.json()
            end_time = time.time()
            
            if 'choices' in response_json and len(response_json['choices']) > 0:
                result = response_json['choices'][0]['message']['content'].strip()
                
                # Parse the response to determine if we need to search
                if result.lower().startswith("search:"):
                    search_query = result[7:].strip()
                    return True, search_query, end_time - start_time
                    
                # Another format might be "SEARCH: query"
                match = re.match(r"SEARCH:\s*(.*)", result, re.IGNORECASE)
                if match:
                    return True, match.group(1), end_time - start_time
                    
                # If no search is needed
                return False, "", end_time - start_time
            else:
                # If there's an error, default to not searching
                return False, "", end_time - start_time
                
        except Exception as e:
            end_time = time.time()
            return False, "", end_time - start_time
            
    def generate_response(
        self, 
        user_input: str, 
        system_prompt: str,
        conversation_history: str,
        search_results: Optional[str] = None,
        model: Optional[str] = None
    ) -> tuple[str, str, float]:
        """Generate response from LLM.
        
        Args:
            user_input: User's input
            system_prompt: System prompt
            conversation_history: Conversation history
            search_results: Search results, if any
            model: Model to use, defaults to default_model
            
        Returns:
            Tuple of response text, model name used, and time taken
        """
        if not model:
            model = self.default_model
        # Ensure model is a string and not None
        model_key: str = model if model is not None else self.default_model
        model_id = self.available_models.get(model_key, self.available_models[self.default_model])
        model_name = model_key
        
        # Prepare content with context
        content = user_input
        if search_results:
            content = f"Search results for your query:\n\n{search_results}\n\nUser question: {user_input}"
            
        if conversation_history:
            system_prompt = f"{system_prompt}\n\nConversation history:\n{conversation_history}"
        
        # Prepare messages
        messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": content
            }
        ]
        
        # Call API
        start_time = time.time()
        try:
            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": self.site_url,
                    "X-Title": self.site_name,
                },
                data=json.dumps({
                    "model": model_id,
                    "messages": messages
                })
            )
            
            response_json = response.json()
            end_time = time.time()
            
            if 'choices' in response_json and len(response_json['choices']) > 0:
                result = response_json['choices'][0]['message']['content']
                return result, model_name, end_time - start_time
            else:
                error_msg = response_json.get('error', {}).get('message', 'Unknown error')
                return f"Error: {error_msg}", model_name, end_time - start_time
                
        except Exception as e:
            end_time = time.time()
            return f"Error generating response: {str(e)}", model_name, end_time - start_time