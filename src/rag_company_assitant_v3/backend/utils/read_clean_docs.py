
import re

from pypdf import PdfReader


def read_and_clean_pdf(pdf_path: str) -> str:
    """Extract and clean text from a PDF file."""

    with open(pdf_path, "rb") as pdf_file:
        reader = PdfReader(pdf_file)

        pages = []
        for page in reader.pages:
            text = page.extract_text() or ""
            pages.append(text)

    text = "\n\n".join(pages)

    # Remove null and unwanted control characters
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

    # Normalize whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()
