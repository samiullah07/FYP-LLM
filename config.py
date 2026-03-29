import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
OPENALEX_EMAIL = os.getenv("OPENALEX_EMAIL", "your@email.com")
OPENALEX_BASE_URL = "https://api.openalex.org"

LLM_MODEL = "llama-3.3-70b-versatile"  # best Groq model for reasoning
MAX_PAPERS = 20
MAX_TOKENS = 2000
