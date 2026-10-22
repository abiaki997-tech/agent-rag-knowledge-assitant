# RAG Document Assistant

An AI-powered document assistant that allows users to upload PDF documents and ask questions about their content using Retrieval-Augmented Generation (RAG).

The application retrieves relevant information from uploaded documents and generates answers grounded in the retrieved context.

## Features

- **PDF Upload:** Upload up to 3 PDF documents, with a maximum size of 10 MB per file.
- **Document Processing:** Extract, clean, and split document text into chunks.
- **Vector Embeddings:** Convert text chunks into embeddings for semantic search.
- **Vector Database:** Store and retrieve document chunks using Qdrant.
- **Hybrid Search:** Combine dense vector search and BM25 keyword search using Reciprocal Rank Fusion (RRF).
- **Agentic RAG:** Use an agentic workflow to answer questions based on retrieved document context.
- **Chat Interface:** Ask questions and view responses through a Streamlit interface.
- **Local and Cloud LLMs:** Support local Ollama models for development and Gemini for cloud-based inference.

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python |
| UI | Streamlit |
| Package Management | uv |
| Pipeline Orchestration | ZenML |
| Embeddings | Ollama / Gemini |
| Vector Database | Qdrant |
| Keyword Search | BM25 |
| Retrieval | Hybrid Search + RRF |
| LLM | Ollama / Gemini |
| Testing | pytest |

## Architecture

```text
              PDF Upload
                   |
                   v
          Document Processing
                   |
                   v
          Text Cleaning
                   |
                   v
             Chunking
                   |
                   v
           Metadata Creation
                   |
                   v
          Embedding Generation
                   |
                   v
            Qdrant Storage
                   |
                   v
              User Query
                   |
                   v
            Query Analysis
                   |
                   v
           Hybrid Retrieval
          /              \
   Dense Search        BM25 Search
          \              /
                   |
                   v
             RRF Fusion
                   |
                   v
         Agentic RAG Workflow
                   |
                   v
          Context-Grounded LLM
                   |
                   v
             Final Answer
```

## Project Structure

```text
rag-company-assitant-v3/
├── app.py
├── backend/
│   ├── inference_pipeline.py
│   ├── agentic_rag.py
│   └── ...
├── frontend/
│   ├── upload.py
│   ├── chat.py
│   └── ...
├── pipelines/
│   └── feature_pipeline.py
├── retrieval/
│   ├── dense_retriever.py
│   ├── bm25_retriever.py
│   ├── hybrid_search.py
│   └── ...
├── schemas/
├── evaluation/
├── tests/
├── data/
├── pyproject.toml
├── uv.lock
├── .env.example
└── README.md
```

*Note: Update the tree to match the actual files in your repository.*

## Prerequisites

Install the following before starting:

- Python 3.11
- [uv](https://docs.astral.sh/uv/)
- A Qdrant instance (local or cloud)
- Ollama for local model inference, or a Gemini API key for cloud inference

## 1. Clone the Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd rag-company-assitant-v3
```

Replace the repository URL with your actual GitHub repository URL.

## 2. Install Dependencies

Install the project dependencies using uv:

```bash
uv sync
```

## 3. Configure Environment Variables

Create a `.env` file in the project root:

```bash
cp .env.example .env
```

Configure the required values in `.env`:

```dotenv
# Model provider: ollama or gemini
LLM_PROVIDER=ollama

# Ollama
OLLAMA_URL=http://localhost:11434
EMBED_MODEL=nomic-embed-text:latest
OLLAMA_MODEL=qwen3:0.6b

# Gemini
GEMINI_API_KEY=your_gemini_api_key
GEMINI_MODEL=your_supported_gemini_model
GEMINI_EMBED_MODEL=your_supported_embedding_model

# Qdrant
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=
COLLECTION=rag_documents
```

Use the environment variable names expected by your actual application code. Do not commit `.env` or API keys to GitHub.

For Qdrant Cloud, replace the local URL with your cluster URL and provide its API key.

## 4. Start the Required Services

### Option A: Local Ollama

Install Ollama from [ollama.com](https://ollama.com/) and pull the required models:

```bash
ollama pull qwen3:0.6b
ollama pull nomic-embed-text
```

Ensure Ollama is running and accessible at `http://localhost:11434`.

Start Qdrant locally if you are using Docker:

```bash
docker run -p 6333:6333 qdrant/qdrant
```

### Option B: Cloud Services

For a cloud deployment, configure:

- Qdrant Cloud for vector storage.
- Gemini for supported embedding and text-generation models.
- Valid API credentials stored in your deployment's secrets or environment variables.

Do not expose API keys in source code or frontend files.

## 5. Run the Application

Start the Streamlit application:

```bash
uv run streamlit run app.py
```

Open the local URL shown in the terminal, typically:

```text
http://localhost:8501
```

## 6. Use the Application

1. Upload one or more PDF documents.
2. Wait for document processing and indexing to finish.
3. Enter a question in the chat interface.
4. The application retrieves relevant document chunks using hybrid search.
5. The RAG workflow uses the retrieved context to generate an answer.

## Ingestion Pipeline

The ingestion pipeline prepares documents for retrieval.

1. **Preprocessing:** Extract and clean PDF text.
2. **Chunking:** Split text into manageable chunks.
3. **Metadata:** Attach identifiers and relevant metadata to chunks.
4. **Embeddings:** Convert chunks into vectors.
5. **Vector Storage:** Store embeddings and payloads in Qdrant.

## Inference Pipeline

The inference pipeline handles user questions.

1. Accept a question and session identifier.
2. Retrieve relevant chunks using dense and BM25 search.
3. Combine ranked results using Reciprocal Rank Fusion (RRF).
4. Pass retrieved context to the agentic RAG workflow.
5. Generate an answer using the configured LLM.
6. Return the answer to the chat interface.

## Evaluation

The retrieval pipeline can be evaluated using metrics such as:

- **Recall@K:** Measures whether relevant chunks appear in the retrieved results.
- **Precision@K:** Measures the proportion of retrieved chunks that are relevant.
- **MRR:** Measures how highly the first relevant result is ranked.

Run tests with:

```bash
uv run pytest
```

If you maintain separate evaluation scripts, add their exact execution commands here.

## Deployment

The application can be deployed using a supported Streamlit hosting platform with cloud-accessible Qdrant and an API-based LLM provider.

Before deployment:

- Configure environment variables and secrets.
- Ensure the Qdrant endpoint is reachable from the hosting environment.
- Verify that embedding and generation models are available.
- Test PDF upload, document retrieval, and question answering.
- Confirm that uploaded documents are isolated by session.
- Check provider quotas, rate limits, and hosting resource limits.

## Future Improvements

- Cross-encoder reranking.
- Query analysis and query rewriting.
- Metadata-based filtering.
- Source citations in generated answers.
- Retrieval evaluation and answer-quality testing.
- Improved document parsing and error handling.