"""Persist chunk text, precomputed embeddings, and metadata in local Chroma."""

import hashlib
import json
from pathlib import Path

import chromadb

DB_PATH = Path(__file__).resolve().parents[3] / "chroma_db"
COLLECTION_NAME = "sales_f2llm_v2_06b"


def get_collection():
    client = chromadb.PersistentClient(path=str(DB_PATH))
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=None,  # We supply our own vectors from embedding.py.
        configuration={"hnsw": {"space": "cosine"}},
    )


def save_chunks(chunks, embeddings):
    if len(chunks) != len(embeddings):
        raise ValueError("Each chunk must have exactly one embedding.")
    collection = get_collection()
    if not chunks:
        return collection

    # Same text and source produce the same ID on repeated runs.
    # Changed/deleted source content needs a separate cleanup/reindex step.
    ids = [
        hashlib.sha256(json.dumps(
            [chunk.page_content, chunk.metadata], sort_keys=True,
        ).encode("utf-8")).hexdigest()
        for chunk in chunks
    ]
    # Small batches also support datasets larger than Chroma's batch limit.
    for start in range(0, len(chunks), 100):
        batch = chunks[start:start + 100]
        collection.upsert(
            ids=ids[start:start + 100],
            documents=[chunk.page_content for chunk in batch],
            embeddings=[list(map(float, vector)) for vector in embeddings[start:start + 100]],
            metadatas=[chunk.metadata for chunk in batch],
        )
    return collection


def list_documents():
    """List indexed documents once per document ID, without loading vectors."""
    records = get_collection().get(include=["metadatas"])
    documents = {}
    for metadata in records["metadatas"] or []:
        metadata = metadata or {}
        document_id = metadata.get("document_id")
        filename = metadata.get("filename")
        # Legacy dataset records may not have uploaded-document metadata.
        if not isinstance(document_id, str) or not document_id.strip():
            continue
        if not isinstance(filename, str) or not filename.strip():
            continue
        if document_id in documents and documents[document_id] != filename:
            raise ValueError("Conflicting filenames for one document ID.")
        documents[document_id] = filename
    return sorted(
        [{"document_id": key, "filename": name} for key, name in documents.items()],
        key=lambda item: (item["filename"].casefold(), item["document_id"]),
    )
