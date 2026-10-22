import streamlit as st

from backend.inference_pipeline import inference_pipeline
from backend.agentic_rag import ask_question


def render_chat():
    st.subheader("Ask your documents")

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # UI: Input at the top
    with st.form("chat_form", clear_on_submit=True):
        cols = st.columns([8, 1])

        with cols[0]:
            question = st.text_input(
                "Ask a question about your PDFs",
                label_visibility="collapsed",
                placeholder="Ask a question about your PDFs",
            )

        with cols[1]:
            submitted = st.form_submit_button("Send")

    # UI: Scrollable messages below
    chat_container = st.container(height=500, border=False)

    with chat_container:
        for message in st.session_state["messages"]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    if not submitted or not question:
        return

    st.session_state["messages"].append(
        {"role": "user", "content": question}
    )

    with chat_container:
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Searching documents and generating answer..."):
                session_id = st.session_state.get("session_id")

                if not session_id:
                    answer = "Please upload and process a PDF first."
                else:
                    try:
                        print(f"Running inference pipeline for session: {session_id} with question: {question}")
                        answer, sources = ask_question(
                            question=question,
                            session_id=session_id,
                        )

                        if sources:
                            answer += "\n\n**Sources:** " + ", ".join(sources)

                    except Exception as exc:
                        answer = f"Inference failed: {exc}"

                st.markdown(answer)

    st.session_state["messages"].append(
        {"role": "assistant", "content": answer}
    )