import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# OpenRouter configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
SITE_URL = os.getenv("SITE_URL", "http://localhost:8000")
SITE_NAME = os.getenv("SITE_NAME", "Memory AI Agent")

# Available models
MODELS = {
    "gemma": "google/gemma-3-12b-it:free",
    "phi": "microsoft/phi-4-reasoning:free"
}

# Default model
DEFAULT_MODEL = "gemma"

# Memory configuration
MEMORY_LIMIT = 10  # Number of past conversations to remember