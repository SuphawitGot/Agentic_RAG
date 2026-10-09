import warnings

# This workflow uses CPU embeddings. Optional dependencies probe
# CUDA during import; hide only these GPU-probe warnings for this script.
# This does not repair the driver; GPU embeddings would need a separate setup.
warnings.filterwarnings(
    "ignore",
    message=r"CUDA initialization: The NVIDIA driver on your system is too old.*",
    category=UserWarning,
    module=r"torch\.cuda",
)
warnings.filterwarnings(
    "ignore",
    message="Can't initialize NVML",
    category=UserWarning,
    module=r"torch\.cuda",
)

from .rag.loader import load_documents
from .rag.chunking import split_documents
from .rag.embedding import embed_chunks
from .rag.vector_store import DB_PATH, save_chunks


def main():
    documents = load_documents()
    chunks = split_documents(documents)
    print(f"{len(documents)} documents → {len(chunks)} chunks")
    if not chunks:
        print("No non-empty text found in this split.")
        return

    embeddings = embed_chunks(chunks)
    print(f"Embedding shape: {embeddings.shape}")
    print("First chunk metadata:", chunks[0].metadata)
    print("First chunk text:", chunks[0].page_content)
    collection = save_chunks(chunks, embeddings)
    print(f"Chroma contains {collection.count()} records at {DB_PATH}")


if __name__ == "__main__":
    main()
