
import os

import requests
# from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
from backend.utils.embeddings import embed_text
from rag_company_assitant_v3.config import PROVIDER, OLLAMA_URL, QDRANT_URL, QDRANT_API_KEY, COLLECTION

def hybrid_search(
    question: str,
    session_id: str,
    top_k: int = 5,
) -> list[dict]:
    embedding = embed_text(question)

    client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY or None,
    )

    results = client.query_points(
        collection_name=COLLECTION,
        query=embedding,
        query_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="session_id",
                    match=models.MatchValue(value=session_id),
                )
            ]
        ),
        limit=top_k,
        with_payload=True,
    )

    chunks = []

    for point in results.points:
        payload = point.payload or {}
        chunks.append({
            "text": payload.get("text", payload.get("chunk", "")),
            "source": payload.get(
                "source",
                payload.get("file_name", "Unknown"),
            ),
            "score": point.score,
        })

    return [chunk for chunk in chunks if chunk["text"]]
