"""Compare pure-vector retrieval without writing to the live collection.

python -m evaluation.compare_embeddings
Test DB and reports are separate, ignored paths. No Qwen or hybrid ranking.
"""
import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import time

import chromadb
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'chroma_db'
SOURCE_COLLECTION = 'sales_minilm_l6_v2'
BASELINE = 'sentence-transformers/all-MiniLM-L6-v2'
CANDIDATE = 'codefuse-ai/F2LLM-v2-0.6B'


def separate_paths(source, target):
    source, target = Path(source).resolve(), Path(target).resolve()
    if source == target or source in target.parents or target in source.parents:
        raise ValueError('Test/output directory must be separate from the live database.')
    return target


def read_records(collection):
    data = collection.get(include=['documents', 'metadatas', 'embeddings'])
    return {i: {'text': t, 'metadata': m, 'vector': list(map(float, v))}
            for i, t, m, v in zip(data['ids'], data['documents'], data['metadatas'], data['embeddings'])}


def fingerprint(records):
    return hashlib.sha256(json.dumps(records, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def check_vectors(vectors, count, dimension):
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.shape != (count, dimension) or not np.isfinite(vectors).all():
        raise ValueError('Wrong vector count/dimension or non-finite values.')
    if not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-4):
        raise ValueError('Vectors must be nonzero and normalized.')
    return vectors


def metrics(hits, case, k):
    expected = set(case['expected_pages'])
    found = {h['page'] for h in hits[:k] if h['filename'] == case['filename']}
    matched = expected & found
    return {'hit': bool(matched), 'page_recall': len(matched) / len(expected)}


def summarize(hits, records):
    return [{'id': i, 'filename': records[i]['metadata'].get('filename'),
             'page': records[i]['metadata'].get('page'), 'cosine_distance': float(distance)}
            for i, distance in hits]


def evaluate_baseline(records, cases, queries):
    ids = sorted(records)
    matrix = np.array([records[i]['vector'] for i in ids])
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    rows = []
    for case, q in zip(cases, queries):
        for scope in ['document', 'global']:
            allowed = [j for j, i in enumerate(ids) if scope == 'global' or records[i]['metadata'].get('filename') == case['filename']]
            start = time.perf_counter()
            scores = matrix[allowed] @ q
            order = sorted(range(len(allowed)), key=lambda j: (-scores[j], ids[allowed[j]]))[:5]
            hits = summarize([(ids[allowed[j]], 1 - scores[j]) for j in order], records)
            rows.append(result_row(case, scope, hits, time.perf_counter() - start))
    return rows


def result_row(case, scope, hits, seconds):
    return {'case': case['id'], 'question': case['question'], 'scope': scope,
            'expected_filename': case['filename'], 'expected_pages': case['expected_pages'],
            'top5': hits, 'at3': metrics(hits, case, 3), 'at5': metrics(hits, case, 5),
            'search_seconds': seconds}


def evaluate_candidate(collection, records, cases, queries):
    rows = []
    for case, q in zip(cases, queries):
        for scope in ['document', 'global']:
            doc_ids = sorted({r['metadata']['document_id'] for r in records.values()
                              if r['metadata'].get('filename') == case['filename']})
            filters = {'where': {'document_id': {'$in': doc_ids}}} if scope == 'document' else {}
            count = sum(scope == 'global' or r['metadata'].get('filename') == case['filename'] for r in records.values())
            start = time.perf_counter()
            got = collection.query(query_embeddings=[q.tolist()], n_results=min(5, count), include=['distances'], **filters)
            hits = summarize(zip(got['ids'][0], got['distances'][0]), records)
            rows.append(result_row(case, scope, hits, time.perf_counter() - start))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--test-db', type=Path, default=ROOT / 'chroma_db_f2llm_test')
    parser.add_argument('--output', type=Path, default=ROOT / 'output/embedding-comparison')
    parser.add_argument('--cases', type=Path, default=ROOT / 'evaluation/cases/embedding_smoke.json')
    parser.add_argument('--batch-size', type=int, default=2)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error('batch-size must be positive')
    test_path = separate_paths(SOURCE, args.test_db)
    output_path = separate_paths(SOURCE, args.output)
    if not (SOURCE / 'chroma.sqlite3').exists():
        raise ValueError('Live database is missing; refusing to create it.')
    source = chromadb.PersistentClient(path=str(SOURCE)).get_collection(SOURCE_COLLECTION, embedding_function=None)
    before = read_records(source)
    if not before:
        raise ValueError('Live database is empty.')
    ids = sorted(before)
    check_vectors([before[i]['vector'] for i in ids], len(ids), 384)
    cases = json.loads(args.cases.read_text())
    if not cases or len({c['id'] for c in cases}) != len(cases):
        raise ValueError('Cases must be nonempty with unique IDs.')
    for case in cases:
        pages = {r['metadata'].get('page') for r in before.values() if r['metadata'].get('filename') == case['filename']}
        if not case['expected_pages'] or not set(case['expected_pages']).issubset(pages):
            raise ValueError(f"Expected pages not indexed: {case['id']}")
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    report_dir = output_path / stamp
    report_dir.mkdir(parents=True, exist_ok=False)
    report = {'baseline_model': BASELINE, 'candidate_model': CANDIDATE, 'device': 'cpu',
              'candidate_dtype': 'float32', 'count': len(ids), 'source_fingerprint': fingerprint(before),
              'test_database': str(test_path), 'collection': 'f2llm_v2_06b_' + stamp.lower(),
              'note': 'Four paired smoke questions; not an overall accuracy benchmark. Baseline exact cosine over stored vectors; candidate Chroma cosine search. No hybrid search, scope classifier, or Qwen.'}
    print(f"Read {len(ids)} live chunks; original vectors remain untouched.", flush=True)
    try:
        from sentence_transformers import SentenceTransformer
        questions = [c['question'] for c in cases]
        baseline = SentenceTransformer(BASELINE, device='cpu')
        start = time.perf_counter()
        queries = baseline.encode(questions, batch_size=1, normalize_embeddings=True, convert_to_numpy=True)
        report['baseline_query_seconds'] = time.perf_counter() - start
        report['baseline'] = evaluate_baseline(before, cases, queries)
        del baseline
        gc.collect()
        print('Baseline complete. Loading F2LLM on CPU.', flush=True)
        model = SentenceTransformer(CANDIDATE, device='cpu', model_kwargs={'torch_dtype': 'float32'})
        # Use the model-provided query prompt; documents must not get this prefix.
        if not model.prompts.get('query'):
            raise ValueError('Candidate lacks its documented query prompt.')
        report['query_prompt'] = model.prompts['query']
        report['max_sequence_length'] = model.max_seq_length
        start = time.perf_counter()
        vectors = model.encode_document([before[i]['text'] for i in ids], batch_size=args.batch_size,
                                        normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=True)
        report['document_embedding_seconds'] = time.perf_counter() - start
        vectors = check_vectors(vectors, len(ids), 1024)
        # create_collection never overwrites an existing experiment collection.
        test = chromadb.PersistentClient(path=str(test_path)).create_collection(
            report['collection'], embedding_function=None, configuration={'hnsw': {'space': 'cosine'}},
            metadata={'embedding_model': CANDIDATE, 'source_fingerprint': report['source_fingerprint']})
        for start in range(0, len(ids), 100):
            batch = ids[start:start+100]
            test.add(ids=batch, documents=[before[i]['text'] for i in batch],
                     metadatas=[before[i]['metadata'] for i in batch], embeddings=vectors[start:start+100].tolist())
        actual = read_records(test)
        if set(actual) != set(before):
            raise ValueError('Test readback IDs differ.')
        for j, i in enumerate(ids):
            if actual[i]['text'] != before[i]['text'] or actual[i]['metadata'] != before[i]['metadata'] or not np.allclose(actual[i]['vector'], vectors[j], atol=1e-6):
                raise ValueError(f'Test readback mismatch: {i}')
        report['test_readback_passed'] = True
        start = time.perf_counter()
        queries = check_vectors(model.encode_query(questions, batch_size=1, normalize_embeddings=True,
                                                   convert_to_numpy=True), len(cases), 1024)
        report['candidate_query_seconds'] = time.perf_counter() - start
        report['candidate'] = evaluate_candidate(test, before, cases, queries)
    finally:
        report['source_unchanged'] = fingerprint(read_records(source)) == report['source_fingerprint']
        (report_dir / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
        if not report['source_unchanged']:
            raise RuntimeError('Live records changed during this run; check concurrent uploads. No live writes are performed by this script.')
    lines = ['# Embedding comparison', '', 'Smoke test only. Page recall measures expected evidence pages, not answer accuracy.', '',
             '| Scope | Question | MiniLM pages / Recall@5 | F2LLM pages / Recall@5 |', '|---|---|---|---|']
    for old, new in zip(report['baseline'], report['candidate']):
        def cell(row):
            return ', '.join(str(h['page']) + ('*' if h['filename'] != row['expected_filename'] else '') for h in row['top5']) + f" / {row['at5']['page_recall']:.0%}"
        lines.append(f"| {old['scope']} | {old['case']} | {cell(old)} | {cell(new)} |")
    lines += ['', '* marks a page from a different PDF.', '', f"Source unchanged: {report['source_unchanged']}. Test readback: {report['test_readback_passed']}.",
              f"CPU document embedding: {report['document_embedding_seconds']:.1f}s. Query totals for {len(cases)} questions: MiniLM {report['baseline_query_seconds']:.2f}s; F2LLM {report['candidate_query_seconds']:.2f}s."]
    (report_dir / 'report.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines), flush=True)
    print(f'Report: {report_dir}', flush=True)


if __name__ == '__main__':
    main()
