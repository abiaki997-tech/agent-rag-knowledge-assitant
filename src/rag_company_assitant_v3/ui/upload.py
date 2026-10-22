import io
import random

import streamlit as st
from pypdf import PdfReader

MAX_FILES = 1
MAX_FILE_SIZE_MB = 10

UPLOAD_QUOTES = [
    "☕ Sit back, relax — the PDF is doing its warm-up stretches...",
    "🐢 Slowly but surely, like a caffeinated turtle...",
    "🧘 Deep breath. Your document is meditating into the system...",
    "📦 Unpacking your PDF — please don't tap the glass...",
    "🍿 Grab a snack, this is almost as exciting as watching paint dry...",
    "🦦 The otter is reading your file. Otters are thorough...",
    "🎩 A magician is hiding your PDF in a vector hat...",
    "🚀 Fueling the rocket with pure PDF energy...",
    "🧠 Teaching the robot your document's secrets...",
    "🎧 Your PDF is listening to lo-fi beats while it uploads...",
]


def validate_pdfs(files):
    valid_files, errors, seen_names = [], [], set()

    if len(files) > MAX_FILES:
        return [], [f"Upload a maximum of {MAX_FILES} PDF file(s)."]

    for file in files:
        name = file.name

        if not name.lower().endswith(".pdf"):
            errors.append(f"{name}: Only PDF files are allowed.")
            continue

        if file.size > MAX_FILE_SIZE_MB * 1024 * 1024:
            errors.append(f"{name}: Maximum size is {MAX_FILE_SIZE_MB} MB.")
            continue

        if name in seen_names:
            errors.append(f"{name}: Duplicate filename.")
            continue

        seen_names.add(name)

        try:
            reader = PdfReader(io.BytesIO(file.getvalue()))

            if reader.is_encrypted:
                errors.append(f"{name}: Encrypted PDFs are not supported.")
                continue

            if len(reader.pages) == 0:
                errors.append(f"{name}: PDF has no pages.")
                continue

            reader.pages[0].extract_text()
            valid_files.append(file)

        except Exception:  # noqa: BLE001
            errors.append(f"{name}: Invalid or unreadable PDF.")

    return valid_files, errors


def render_upload():
    st.subheader("Upload documents")

    files = st.file_uploader(
        f"Choose up to {MAX_FILES} PDF file(s)",
        type=["pdf"],
        accept_multiple_files=MAX_FILES > 1,
        max_upload_size=MAX_FILE_SIZE_MB,
        key="pdf_upload",
    )

    if not files:
        return st.session_state.get("uploaded_pdfs", {})

    file_list = files if isinstance(files, list) else [files]
    cached = st.session_state.get("uploaded_pdfs", {})
    names = {f.name for f in file_list}

    # Already validated on a previous rerun — do NOT re-run the spinner
    # or re-parse the PDF. This is what makes the upload spinner fire
    # only when a new file actually arrives.
    if names == set(cached.keys()):
        return cached

    # Force the spinner delta to flush to the browser BEFORE the
    # blocking validation starts. Without this, the spinner appears
    # after the work is done (looks like a frozen UI).
    with st.spinner(random.choice(UPLOAD_QUOTES)):
        st.empty()  # flush
        valid_files, errors = validate_pdfs(file_list)

        for error in errors:
            st.error(error)

        if valid_files:
            st.session_state["uploaded_pdfs"] = {
                f.name: f.getvalue() for f in valid_files
            }
            st.success(f"{len(valid_files)} PDF(s) validated.")
            for f in valid_files:
                st.caption(f"{f.name} · {f.size / 1024:.1f} KB")

    return st.session_state.get("uploaded_pdfs", {})