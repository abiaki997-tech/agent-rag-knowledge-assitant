
from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_text(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 350,
) -> list[dict]:
    """Split cleaned text into overlapping chunks."""

    if not text.strip():
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
        length_function=len,
    )

    chunks = splitter.split_text(text)

    return [
        {
            "text": chunk,
            "chunk_number": index,
        }
        for index, chunk in enumerate(chunks)
    ]
