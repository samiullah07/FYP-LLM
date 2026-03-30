import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
OPENALEX_EMAIL = os.getenv("OPENALEX_EMAIL", "figmaeditaccess@email.com")
OPENALEX_BASE_URL = "https://api.openalex.org"

LLM_MODEL = "openai/gpt-oss-120b"  # best Groq model for reasoning
MAX_PAPERS = 20
MAX_TOKENS = 2000
