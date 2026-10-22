# retrieval/agentic_rag.py

import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict
from backend.inference_pipeline import inference_pipeline
from rag_company_assitant_v3.config import (
    PROVIDER,
    TEXT_GENERATION_MODEL,
    GEMINI_API_KEY,
    OLLAMA_URL,
)
from langchain_ollama import ChatOllama

if PROVIDER.lower() == "ollama":
    llm = ChatOllama(
        model=TEXT_GENERATION_MODEL,
        base_url=OLLAMA_URL,
        temperature=0,
    )
else:
    llm = ChatGoogleGenerativeAI(
        model=TEXT_GENERATION_MODEL,
        google_api_key=GEMINI_API_KEY,
        temperature=0,
    )

web_search = TavilySearch(max_results=3)


class RAGState(TypedDict):
    question: str
    session_id: str
    context: str
    source: list[str]
    answer: str


def retrieve_pdf(state: RAGState):
    # TODO: Replace with your existing Qdrant hybrid search.
    # Filter results using state["session_id"].
    docs, sources = inference_pipeline(
        state["question"],
        state["session_id"],
    )

    if docs:
        print(f"Retrieved {len(docs)} documents from Qdrant for session {state['session_id']}.")
        return {"context": str(docs), "source": sources}

    return {"context": "", "source": ["GET_FROM_WEB"]}


def route_source(state: RAGState):
    if state["source"] == ["GET_FROM_WEB"]:
        return "search_web"
    return "generate"


def search_web(state: RAGState):
    results = web_search.invoke({"query": state["question"]})
    return {
        "context": str(results),
        "source": ["web"],
    }


def generate(state: RAGState):
    response = llm.invoke(f"""
        Answer the user's question using ONLY the context below.
        If the context does not contain the answer, say so.
        Include URLs when web sources provide them.

        Question: {state['question']}
        Source: {state['source']}
        Context: {state['context']}
        """)
    return {"answer": response.content}


builder = StateGraph(RAGState)

builder.add_node("retrieve_pdf", retrieve_pdf)
builder.add_node("search_web", search_web)
builder.add_node("generate", generate)

builder.add_edge(START, "retrieve_pdf")
builder.add_conditional_edges(
    "retrieve_pdf",
    route_source,
    {
        "generate": "generate",
        "search_web": "search_web",
    },
)
builder.add_edge("search_web", "generate")
builder.add_edge("generate", END)

agentic_rag = builder.compile()


def ask_question(question: str, session_id: str) -> str:
    result = agentic_rag.invoke(
        {
            "question": question,
            "session_id": session_id,
            "context": "",
            "source": "",
            "answer": "",
        }
    )
    return result["answer"], result.get("source")
