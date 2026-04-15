# src/config.py
from pathlib import Path
from dotenv import load_dotenv
from pydantic import Extra
from pydantic_settings import BaseSettings

load_dotenv()

class Settings(BaseSettings):
    # LLM
    openai_api_key: str = ""
    groq_api_key: str = ""                     # <-- add this
    llm_model: str = "llama-3.3-70b-versatile"    # APIs
    openalex_base_url: str = "https://api.openalex.org"
    semantic_scholar_base_url: str = "https://api.semanticscholar.org/graph/v1"

    # Paths
    project_root: Path = Path(__file__).resolve().parents[1]
    data_dir: Path = project_root / "data"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = Extra.allow

settings = Settings()