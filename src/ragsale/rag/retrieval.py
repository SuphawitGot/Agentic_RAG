"""Search saved chunks: python -m ragsale.rag.retrieval 'Your question'"""

import argparse
import numpy as np

from .embedding import load_embedding_model
from .vector_store import get_collection
from .keyword_search import keyword_rank


def retrieve(question, top_k=3, document_ids=None, query_vector=None):
    """Return evidence with cosine distances; scoped searches fuse dense/keyword ranks."""
    question = question.strip()
    if not question:
        raise ValueError("Question cannot be empty.")
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
        raise ValueError("top_k must be a positive integer.")
    if document_ids is not None and not document_ids:
        return []  # An empty scope must never widen to all documents.

    # Open the same persistent collection used during document preparation.
    collection = get_collection()
    filters = {} if document_ids is None else {'where': {'document_id': {'$in': list(document_ids)}}}
    scoped = collection.get(include=['documents', 'metadatas'], **filters) if filters else None
    count = collection.count() if scoped is None else len(scoped['ids'])
    if count == 0:
        return []

    # Use the SAME model and normalization as the stored document vectors.
    if query_vector is None:
        model = load_embedding_model()
        query_vector = model.encode_query(question, normalize_embeddings=True)

    # Outer list means one question. Chroma supports multiple questions at once.
    results = collection.query(
        query_embeddings=[query_vector.tolist()],
        n_results=min(max(top_k, 12) if scoped is not None else top_k, count),
        include=["documents", "metadatas", "distances"],
        **filters,
    )

    # [0] selects results for our first (and only) question.
    dense = [
        {"id": record_id, "text": text, "metadata": metadata, "distance": distance}
        for record_id, text, metadata, distance in zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )
    ]
    if scoped is None:
        return dense
    records = [{'id': i, 'text': t, 'metadata': m} for i, t, m in
               zip(scoped['ids'], scoped['documents'], scoped['metadatas'])]
    lexical = keyword_rank(question, records)[:max(top_k, 12)]
    if not lexical:
        return dense[:top_k]
    # Reciprocal rank fusion combines rank positions, not incompatible raw scores.
    scores = {}
    for ranking in ([m['id'] for m in dense], lexical):
        for rank, record_id in enumerate(ranking, 1):
            scores[record_id] = scores.get(record_id, 0) + 1 / (60 + rank)
    ordered = sorted(scores, key=lambda i: (-scores[i], i))[:top_k]
    # Truncated rank fusion favors hits present in both pools. Reserve up to
    # half the budget (rounded up) for the strongest keyword hits (e.g. Thai query
    # against an English software caption). Otherwise fusion can erase the rescue.
    reserved = lexical[:max(1, (top_k + 1) // 2)]
    for record_id in reserved:
        if record_id not in ordered:
            replace = next(i for i in range(len(ordered) - 1, -1, -1) if ordered[i] not in reserved)
            ordered[replace] = record_id
    matches = {m['id']: m for m in dense}
    missing = [i for i in ordered if i not in matches]
    if missing:
        extra = collection.get(ids=missing, include=['documents', 'metadatas', 'embeddings'], **filters)
        q = np.asarray(query_vector)
        for i, t, m, vector in zip(extra['ids'], extra['documents'], extra['metadatas'], extra['embeddings']):
            v = np.asarray(vector)
            distance = float(1 - np.dot(q, v) / (np.linalg.norm(q) * np.linalg.norm(v)))
            matches[i] = {'id': i, 'text': t, 'metadata': m, 'distance': distance}
    # distance remains cosine distance; final ordering is the hybrid ranking.
    return [matches[i] for i in ordered if i in matches]


def retrieve_projects(question, projects, top_k=3):
    """Give each project its own result budget; encode the question only once."""
    if not projects:
        return []
    vector = load_embedding_model().encode_query(question, normalize_embeddings=True)
    candidates = []
    for project in projects:
        matches = retrieve(question, top_k,
            [d['document_id'] for d in project['documents']], query_vector=vector)
        for match in matches:
            match['metadata'] = {**match['metadata'], 'project_id': project['project_id'],
                                 'project_name': project['project_name']}
        candidates.append({**project, 'matches': matches})
    return candidates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", help="Question in quotation marks")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()
    try:
        matches = retrieve(args.question, args.top_k)
    except ValueError as error:
        parser.error(str(error))
    if not matches:
        print("No stored chunks. Run python main.py to prepare documents first.")
        return
    for rank, match in enumerate(matches, start=1):
        print(f"\nMatch {rank} | cosine distance: {match['distance']:.4f}")
        print("Metadata:", match["metadata"])
        print(match["text"])
    print("\nThese are retrieved chunks, not an LLM-generated answer.")


if __name__ == "__main__":
    main()
