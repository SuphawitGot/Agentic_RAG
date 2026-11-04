"""Split documents while preserving their source metadata."""

from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    # Sizes are characters, not tokens.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(documents)
