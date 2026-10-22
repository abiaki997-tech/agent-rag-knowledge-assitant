
from pathlib import Path
import os
import hashlib
import requests
import logging
from qdrant_client import QdrantClient, models
import datetime
from datetime import timezone
from zenml import pipeline,step

# Use timezone-aware UTC datetime
start_time = datetime.datetime.now(timezone.utc)


from backend.utils.embeddings import create_embeddings
from backend.utils.read_clean_docs import read_and_clean_pdf
from backend.utils.chunk_docs import chunk_text
from rag_company_assitant_v3.config import PROVIDER,OLLAMA_URL,QDRANT_URL,QDRANT_API_KEY,COLLECTION

logger = logging.getLogger(__name__)

QDRANT_URL = QDRANT_URL
COLLECTION = COLLECTION

# Step 1: Read and clean documents
@step
def load_documents(pdf_path: str) -> list:
    logger.info("Loading and cleaning documents...")
    print(f"Loading and cleaning documents from: {pdf_path}",flush=True)

    path = Path(pdf_path)

    if not path.is_file():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    text = read_and_clean_pdf(str(path))

    # Hash the file contents, not its temporary path.
    doc_id = hashlib.sha256(path.read_bytes()).hexdigest()

    documents = [
        {
            "text": text,
            "source": path.name,
            "doc_id": doc_id,
        }
    ]

    logger.info("Documents loaded and cleaned: %d", len(documents))
    return documents



# Step 2: Split documents into chunks
@step
def chunk_documents(documents: list) -> list:
    logger.info("Splitting documents into chunks...")
    chunks = []
    for document in documents:
        chunked_text = chunk_text(document["text"])
        for i, chunk in enumerate(chunked_text):
            chunks.append({
                "text": chunk["text"],
                "source": document["source"],
                "doc_id": document["doc_id"],
                "chunk_number": i
            })
    logger.info("Documents split into chunks: %d", len(chunks))
    return chunks


# Step 3: Create embeddings using Ollama
@step
def get_embeddings(chunks: list) -> list:
    logger.info("Creating embeddings for chunks...")
    chunks_created = create_embeddings(chunks)
    logger.info("Embeddings created: %d", len(chunks_created))
    return chunks_created


# Step 4: Store vectors and metadata in Qdrant
@step
def save_to_qdrant(chunks: list, session_id: str, filename: str) -> int:
    logger.info("Saving chunks to Qdrant...Qdrant URL: %s, Collection: %s", QDRANT_URL, COLLECTION)
    client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY or None,
    )
    logger.info("Connected to Qdrant at %s", QDRANT_URL)
    if not chunks:
        logger.info("No chunks to save")
        return 0

    vector_size = len(chunks[0]["embedding"])

    if not client.collection_exists(COLLECTION):
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE
            )
        )

    points = []

    for chunk in chunks:
        points.append(
            models.PointStruct(
                id=hashlib.sha256(
                    (
                        chunk["doc_id"]
                        + str(chunk["chunk_number"])
                    ).encode()
                ).hexdigest()[:32],
                vector=chunk["embedding"],
                payload={
                    "text": chunk["text"],
                    "source": filename,
                    "doc_id": chunk["doc_id"],
                    "chunk_number": chunk["chunk_number"],
                    "session_id": session_id,
                }
            )
        )

    client.upsert(
        collection_name=COLLECTION,
        points=points,
        wait=True
    )

    logger.info("Chunks saved to Qdrant: %d", len(points))
    return len(points)


# Step 5: Connect the steps
@pipeline(enable_cache=True)
def feature_pipeline(pdf_path: str = None, session_id: str = None, filename: str = None):
    logger.info(f"Starting feature pipeline for session: {session_id}")
    documents = load_documents(pdf_path=pdf_path)
    chunks = chunk_documents(documents)
    embedded_chunks = get_embeddings(chunks)
    points_saved = save_to_qdrant(embedded_chunks, session_id=session_id, filename=filename)
    logger.info(f"Feature pipeline completed for session: {session_id}")
    if points_saved:
        logger.info(f"Successfully saved {points_saved} chunks to Qdrant")
        return True
    else:
        logger.error("Failed to save chunks to Qdrant")
        return False

if __name__ == "__main__":
    feature_pipeline()
