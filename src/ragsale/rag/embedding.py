"""Create local CPU embeddings. The first run downloads the model."""

from functools import lru_cache

MODEL_NAME = "codefuse-ai/F2LLM-v2-0.6B"
EMBEDDING_DIMENSION = 1024


@lru_cache(maxsize=1)
def load_embedding_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(
        MODEL_NAME,
        device="cpu",
        model_kwargs={"torch_dtype": "float32"},
    )


def embed_chunks(chunks):
    if not chunks:
        raise ValueError("No chunks to embed. Load non-empty documents first.")
    model = load_embedding_model()
    # embeddings[i] corresponds to chunks[i]; keep both in the same order.
    return model.encode_document(
        [chunk.page_content for chunk in chunks],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
        batch_size=2,
    )
