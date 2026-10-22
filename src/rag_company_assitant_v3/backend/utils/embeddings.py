
import os

import requests
from google import genai
from google.genai import types
from config import PROVIDER, OLLAMA_URL, EMBEDDING_MODEL, GEMINI_API_KEY

PROVIDER = PROVIDER
OLLAMA_URL = OLLAMA_URL
EMBEDDING_MODEL = EMBEDDING_MODEL
GEMINI_API_KEY = GEMINI_API_KEY

_gemini_client = None


def _get_gemini_client():
    global _gemini_client

    if _gemini_client is None:
        api_key = GEMINI_API_KEY
        if not api_key:
            raise ValueError("GEMINI_API_KEY is not configured")

        _gemini_client = genai.Client(api_key=api_key)

    return _gemini_client


def embed_text(text: str) -> list[float]:
    """Generate one embedding using the configured provider."""

    if PROVIDER == "ollama":
        response = requests.post(
            f"{OLLAMA_URL}/api/embed",
            json={
                "model": EMBEDDING_MODEL,
                "input": text,
                "dimensions": 512,
            },
            timeout=120,
        )
        response.raise_for_status()
        return response.json()["embeddings"][0]

    if PROVIDER == "gemini":
        client = _get_gemini_client()

        response = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=types.Content(
                parts=[
                    types.Part.from_text(
                        text=f"task: search result | query: {text}"
                    )
                ]
            ),
            config=types.EmbedContentConfig(
                output_dimensionality=512,
            ),
        )

        return list(response.embeddings[0].values)

    raise ValueError(
        f"Unsupported embedding provider: {PROVIDER}"
    )


def create_embeddings(chunks: list) -> list:
    """Attach embeddings to document chunks."""

    for chunk in chunks:
        chunk["embedding"] = embed_text(chunk["text"])

    print(
        f"Embeddings created: {len(chunks)} "
        f"using {PROVIDER}"
    )
    return chunks
