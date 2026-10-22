import random
import tempfile
from pathlib import Path
from uuid import uuid4

import streamlit as st
from dotenv import load_dotenv

from ui.chat import render_chat
from ui.upload import MAX_FILES, render_upload

load_dotenv(".env")

st.set_page_config(
    page_title="Agentic RAG Document Assistant",
    page_icon="📚",
    layout="wide",
)

if "session_id" not in st.session_state:
    st.session_state["session_id"] = uuid4().hex

if "_extracting" not in st.session_state:
    st.session_state["_extracting"] = False

st.title("📚 Agentic RAG Document Assistant")
st.caption("Upload PDFs and ask questions about their contents.")

with st.sidebar:
    st.header("Document settings")
    st.write(f"Maximum files: {MAX_FILES}")
    st.write("Maximum file size: 10 MB each")
    st.write("Supported format: PDF")

left, right = st.columns([1, 1.5], gap="large")

EXTRACT_QUOTES = [
    "☕ Sit back and relax — extracting text from your PDF...",
    "🧹 Sweeping the PDF's dusty corners, chunk by chunk...",
    "🔍 Hunting for meaning inside your document...",
    "🍳 Cooking your PDF into bite-sized chunks...",
    "🎣 Fishing out useful sentences from the PDF soup...",
    "🧠 Teaching the machine what your document is about...",
    "📦 Unpacking your PDF like a very patient moving crew...",
    "🦉 A wise owl is carefully reading your document...",
    "🚂 Chopping the PDF into tasty little chunks...",
    "🛋️ Relax — your PDF is being transformed into knowledge...",
]

with left:
    # Don't re-render the uploader (and don't re-fire its spinner)
    # while the extraction is running.
    if not st.session_state["_extracting"]:
        uploaded_pdfs = render_upload()
    else:
        uploaded_pdfs = st.session_state.get("uploaded_pdfs", {})

    if st.button("Extract and clean PDFs"):
        pdfs = st.session_state.get("uploaded_pdfs", {})

        if not pdfs:
            st.warning("Please upload a valid PDF.")
        else:
            from backend.feature_pipeline import feature_pipeline

            st.session_state["_extracting"] = True

            with st.spinner(random.choice(EXTRACT_QUOTES)):
                st.empty()  # flush spinner to browser BEFORE blocking work
                progress = st.progress(0)
                status = st.status("Starting the paperwork...", expanded=True)
                total = len(pdfs)

                try:
                    for index, (file_name, file_bytes) in enumerate(
                        pdfs.items(), start=1
                    ):
                        status.write(f"📥 Processing: {file_name}")

                        with tempfile.NamedTemporaryFile(
                            suffix=".pdf", delete=False
                        ) as tmp:
                            tmp.write(file_bytes)
                            pdf_path = tmp.name

                        try:
                            status.write(
                                "🧠 Embedding chunks — this is the slow part, hang tight..."
                            )
                            feature_pipeline(
                                pdf_path=pdf_path,
                                session_id=st.session_state["session_id"],
                                filename=file_name,
                            )
                        finally:
                            Path(pdf_path).unlink(missing_ok=True)

                        progress.progress(index / total)
                        status.write(f"✅ Done: {file_name}")

                    status.update(
                        label="All PDFs processed — take a coffee break, you earned it ☕",
                        state="complete",
                        expanded=False,
                    )
                    st.success(
                        f"Successfully processed {total} PDF(s). "
                        "The assistant is ready — ask away."
                    )

                except Exception as error:  # noqa: BLE001
                    status.update(
                        label="PDF processing failed — the PDF fought back 😅",
                        state="error",
                        expanded=True,
                    )
                    st.exception(error)

                finally:
                    st.session_state["_extracting"] = False

if not st.session_state["_extracting"]:
    with right:
        render_chat()