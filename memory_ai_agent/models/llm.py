import requests
import json
import time
import re
import logging
from typing import Dict, Any, List, Optional, Tuple
import openai
from openai import OpenAI

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
        self.api_key = api_key
        self.site_url = site_url
        self.site_name = site_name
        self.default_model = default_model
        self.available_models = available_models or {
            "default": default_model,
            # Adding alternative models for rate limit handling
            "alternative_1": "nousresearch/deephermes-3-mistral-24b-preview:free",
            "alternative_2": "qwen/qwen3-1.7b:free"
        }
        
        # Alternative models to use when rate limits are hit
        self.alternative_models = {
            "primary": "nousresearch/deephermes-3-mistral-24b-preview:free",
            "secondary": "qwen/qwen3-1.7b:free"
        }
        
        # Map from model name to alternative model ID
        self.model_alternatives = {}
        for model_id, model_name in self.alternative_models.items():
            for original_name in self.available_models.keys():
                if original_name not in self.model_alternatives:
                    self.model_alternatives[original_name] = []
                self.model_alternatives[original_name].append(model_id)
        
        if api_key:
            self.client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=api_key,
                default_headers={
                    "HTTP-Referer": site_url,
                    "X-Title": site_name
                }
            )
            self._test_connection()
        else:
            self.client = None
            logger.warning("No API key provided. LLM functionality will be limited.")
    
    def _test_connection(self):
        """Test connection to API."""
        try:
            # Simple test to verify connection
            response = self.client.chat.completions.create(
                model=self.available_models.get(self.default_model, self.default_model),
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=5
            )
            logger.info(f"Successfully connected to LLM API with model: {self.default_model}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to LLM API: {str(e)}")
            return False    def _handle_rate_limit(self, current_model_id: str) -> str:
        """
        Handle rate limit errors by switching to alternative models.
        
        Args:
            current_model_id: The current model ID that hit a rate limit
            
        Returns:
            A new model ID to use
        """
        logger.warning(f"Rate limit exceeded for model {current_model_id}. Attempting to switch models.")
        
        # Use these alternative models when rate limit is exceeded
        primary_alternative = "nousresearch/deephermes-3-mistral-24b-preview:free"
        secondary_alternative = "qwen/qwen3-1.7b:free"
        
        # Determine which alternative to use
        if current_model_id == primary_alternative:
            # If the primary alternative is already rate limited, use the secondary
            logger.info(f"Primary alternative {primary_alternative} is rate limited, switching to {secondary_alternative}")
            return secondary_alternative
        
        # Try the primary alternative first for any other model
        logger.info(f"Switching from {current_model_id} to alternative model {primary_alternative}")
        return primary_alternative
              def check_knowledge(
        self,
        user_input: str,
        system_prompt: str,
        conversation_history: str,
        model: Optional[str] = None
    ) -> Tuple[bool, str, float]:
        """
        Check if the AI has knowledge about a user query
        
        Returns a tuple of:
        - Boolean indicating if AI has knowledge
        - Explanation string
        - Confidence score (0-1)
        """
        if not self.client:
            return False, "No API connection available", 0.0
            
        model_id = self.available_models.get(model or self.default_model, self.default_model)
        
        try:
            response = self.client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Conversation history:\n{conversation_history}\n\nUser query: {user_input}"}
                ],
                temperature=0.1,
                max_tokens=150,
                stop=None,
                headers={
                    "HTTP-Referer": self.site_url,
                    "X-Title": self.site_name
                }
            )
            
            # Parse the response
            response_text = response.choices[0].message.content.strip()
            
            # The expected format is: YES/NO: explanation
            parts = response_text.split(":", 1)
            if len(parts) < 2:
                return False, "Unable to determine", 0.5
                
            decision = parts[0].strip().upper()
            explanation = parts[1].strip() if len(parts) > 1 else ""
            
            has_knowledge = decision == "YES"
            confidence = 0.9 if has_knowledge else 0.1
            
            return has_knowledge, explanation, confidence
            
        except openai.RateLimitError:
            # Handle rate limit by switching models
            new_model = self._handle_rate_limit(model_id)
            # Recursive call with new model
            return self.check_knowledge(user_input, system_prompt, conversation_history, new_model)
            
        except Exception as e:
            logger.error(f"Error during knowledge check: {str(e)}")
            return False, f"Error: {str(e)}", 0.0
              def generate_response(
        self, 
        user_input: str, 
        system_prompt: str,
        conversation_history: str,
        search_results: Optional[str] = None,
        model: Optional[str] = None
    ) -> Tuple[str, str, float]:
        """Generate response from LLM."""
        max_retries = 3
        current_retry = 0
        current_model = model or self.default_model
        
        while current_retry < max_retries:
            try:
                model_id = self.available_models.get(current_model, self.default_model)
                
                # Prepare the messages
                messages = [
                    {"role": "system", "content": system_prompt}
                ]
                
                # Add conversation history if provided
                if conversation_history:
                    history_messages = json.loads(conversation_history)
                    messages.extend(history_messages)
                
                # Add search results if available
                if search_results:
                    messages.append({"role": "system", "content": f"Search results:\n{search_results}"})
                
                # Add the user's current input
                messages.append({"role": "user", "content": user_input})
                
                # Prepare the API request
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "HTTP-Referer": self.site_url,
                    "X-Title": self.site_name,
                    "Content-Type": "application/json"
                }
                
                data = {
                    "model": model_id,
                    "messages": messages,
                    "temperature": 0.7,
                    "max_tokens": 1024
                }
                
                # Time the API call
                start_time = time.time()
                response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data)
                
                if response.status_code == 429:  # Rate limit exceeded
                    current_model = self._handle_rate_limit(model_id)
                    current_retry += 1
                    logger.info(f"Retrying with model {current_model} (attempt {current_retry}/{max_retries})")
                    continue
                
                response_data = response.json()
                elapsed_time = time.time() - start_time
                
                if response.status_code == 200:
                    assistant_message = response_data["choices"][0]["message"]["content"]
                    model_used = response_data.get("model", model_id).split(":")[0]
                    return (assistant_message, model_used, elapsed_time)
                else:
                    logger.error(f"Error from API: {response_data}")
                    error_message = response_data.get("error", {}).get("message", "Unknown error")
                    return (f"Error generating response: {error_message}", str(model_id), elapsed_time)
            
            except Exception as e:
                error_message = str(e).lower()
                
                if "rate limit" in error_message:
                    current_model = self._handle_rate_limit(model_id)
                    current_retry += 1
                    logger.info(f"Retrying with model {current_model} (attempt {current_retry}/{max_retries})")
                else:
                    logger.error(f"Error generating response: {str(e)}")
                    return (f"Error: {str(e)}", str(current_model), 0.0)
        
        # If we've exhausted all retries
        logger.error("Failed to generate response after multiple retries with different models")
        return ("I'm currently experiencing high demand. Please try again in a few moments.", str(current_model), 0.0)
