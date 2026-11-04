"""Prepare an uploaded PDF for retrieval using the existing RAG functions."""

from .loader import load_pdf_documents
from .chunking import split_documents
from .embedding import embed_chunks
from .vector_store import save_chunks, get_collection


def index_pdf(path, document_id, filename):
    documents = load_pdf_documents(path, document_id, filename)
    chunks = split_documents(documents)
    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = index
    embeddings = embed_chunks(chunks)
    try:
        save_chunks(chunks, embeddings)
    except Exception:
        # A later batch may fail after earlier batches were saved.
        get_collection().delete(where={"document_id": document_id})
        raise
    return {
        "text_pages": len(documents),
        "chunks": len(chunks),
        "ocr_pages": sum("ocr" in doc.metadata["extraction_method"] for doc in documents),
    }
