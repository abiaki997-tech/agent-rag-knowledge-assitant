import streamlit as st

from backend.inference_pipeline import inference_pipeline


def render_chat():
    st.subheader("Ask your documents")

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input("Ask a question about your PDFs")

    if not question:
        return

    st.session_state["messages"].append(
        {"role": "user", "content": question}
    )

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
                    answer, sources = inference_pipeline(
                        question=question,
                        session_id=session_id,
                        top_k=3,
                    )

                    if sources:
                        answer += "\n\n**Sources:** " + ", ".join(sources)

                except Exception as exc:
                    answer = f"Inference failed: {exc}"

            st.markdown(answer)

    st.session_state["messages"].append(
        {"role": "assistant", "content": answer}
    )