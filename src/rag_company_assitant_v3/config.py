import os

import streamlit as st
from dotenv import load_dotenv

load_dotenv(".env")


def get(key, default=None):
    """Read from st.secrets first, then .env / os.environ, then default."""
    try:
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return os.getenv(key, default)



# Keys
PROVIDER = get("PROVIDER", "ollama").lower()
EMBEDDING_MODEL = get("EMBEDDING_MODEL", "nomic-embed-text:latest")
TEXT_GENERATION_MODEL = get("TEXT_GENERATION_MODEL", "qwen3:0.6b")
OLLAMA_URL = get("OLLAMA_URL", "http://localhost:11434")
COLLECTION = get("COLLECTION", "rag_documents")
QDRANT_URL = get("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = get("QDRANT_API_KEY", None)
GEMINI_API_KEY = get("GEMINI_API_KEY", None)