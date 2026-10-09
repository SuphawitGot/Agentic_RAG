"""Replay saved F2LLM retrieval results through the existing Qwen answer function.

Only reads the isolated test collection; never opens or writes the live database.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

import chromadb

from evaluation.compare_embeddings import CANDIDATE, SOURCE, fingerprint, read_records, separate_paths
from ragsale.rag.answering import GenerationError, SYSTEM_PROMPT, generate_answer


def prepare_matches(row, records, top_k):
    hits = row['top5'][:top_k]
    if not hits:
        raise ValueError('No saved retrieval hits for this case.')
    matches = []
    seen = set()
    for hit in hits:
        i = hit['id']
        if i in seen or i not in records:
            raise ValueError('Missing or duplicate retrieved chunk ID.')
        seen.add(i)
        record = records[i]
        metadata = record['metadata']
        if metadata.get('filename') != hit['filename'] or metadata.get('page') != hit['page']:
            raise ValueError('Saved retrieval metadata differs from test collection.')
        if row['scope'] == 'document' and metadata.get('filename') != row['expected_filename']:
            raise ValueError('A document-scoped result contains a different PDF.')
        matches.append({'id': i, 'text': record['text'], 'metadata': metadata,
                        'distance': hit['cosine_distance']})
    return matches


def run_case(row, matches, answer_fn=generate_answer):
    start = time.perf_counter()
    result = {'case': row['case'], 'question': row['question'], 'scope': row['scope'],
              'expected_filename': row['expected_filename'], 'expected_pages': row['expected_pages'],
              'retrieved_sources': [dict(m, citation=j) for j, m in enumerate(matches, 1)]}
    found = {m['metadata'].get('page') for m in matches if m['metadata'].get('filename') == row['expected_filename']}
    result['expected_page_recall'] = len(found & set(row['expected_pages'])) / len(set(row['expected_pages']))
    try:
        # Only question and source evidence go to Qwen, never the answer key.
        answer = answer_fn(row['question'], matches)
        result.update(answer)
        result['status'] = 'abstained' if answer['insufficient_evidence'] else 'answered'
        result['citation_validation'] = 'not_applicable' if answer['insufficient_evidence'] else 'passed'
    except GenerationError as exc:
        result.update(status='error', error=str(exc), citation_validation='not_passed')
    result['generation_seconds'] = time.perf_counter() - start
    result['factual_review'] = 'not_scored_requires_source_review'
    return result


def markdown(report):
    lines = ['# Qwen answers from F2LLM retrieval', '',
             f"Model: {report['generation_model']}. Top K: {report['top_k']}. Scope: {report['scope']}.",
             'Replays saved retrieval ranks. No hybrid search, new embeddings, or expected-page hints to Qwen.',
             'Citation validation does not establish factual correctness. These four questions are a smoke test.', '']
    for r in report['results']:
        lines += [f"## {r['case']}", '', r['question'], '',
                  f"Expected evidence: {r['expected_filename']}, pages {r['expected_pages']}. Retrieved page recall: {r['expected_page_recall']:.0%}.",
                  f"Status: {r['status']}. Time: {r['generation_seconds']:.2f}s.", '',
                  r.get('answer', r.get('error', '')), '', '### Evidence supplied to Qwen', '']
        cited = {s['citation'] for s in r.get('sources', [])}
        for s in r['retrieved_sources']:
            lines += [f"#### [{s['citation']}] {s['metadata'].get('filename')} — page {s['metadata'].get('page')}" + (' (cited)' if s['citation'] in cited else ''), '',
                      *['> ' + line for line in s['text'].splitlines()], '']
    lines += [f"Test collection unchanged: {report.get('test_collection_unchanged', 'pending')}",
              'Live database: not opened by this evaluator.', '']
    return '\n'.join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--retrieval-report', type=Path, required=True)
    p.add_argument('--top-k', type=int, choices=range(1, 6), default=5)
    p.add_argument('--scope', choices=['document', 'global'], default='document')
    args = p.parse_args()
    saved = json.loads(args.retrieval_report.read_text())
    if saved['candidate_model'] != CANDIDATE or not saved.get('test_readback_passed') or not saved.get('source_unchanged'):
        raise ValueError('Use a completed, verified F2LLM comparison report.')
    test_path = separate_paths(SOURCE, saved['test_database'])
    if not (test_path / 'chroma.sqlite3').exists():
        raise ValueError('Test database missing; refusing to create one.')
    collection = chromadb.PersistentClient(path=str(test_path)).get_collection(saved['collection'], embedding_function=None)
    if collection.metadata.get('embedding_model') != CANDIDATE or collection.metadata.get('source_fingerprint') != saved['source_fingerprint']:
        raise ValueError('Collection provenance does not match retrieval report.')
    records = read_records(collection)
    rows = [r for r in saved['candidate'] if r['scope'] == args.scope]
    if not rows:
        raise ValueError('No cases for requested scope.')
    prepared = [(r, prepare_matches(r, records, args.top_k)) for r in rows]
    out = separate_paths(SOURCE, args.retrieval_report.resolve().parent / ('answers-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')))
    out.mkdir(parents=True, exist_ok=False)
    report = {'retrieval_report': str(args.retrieval_report.resolve()), 'embedding_model': CANDIDATE,
              'generation_model': os.getenv('QWEN_MODEL', 'qwen3:8b'), 'top_k': args.top_k, 'scope': args.scope,
              'system_prompt_sha256': hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
              'test_collection': saved['collection'], 'test_fingerprint_before': fingerprint(records),
              'live_database_accessed': False, 'results': []}
    def save():
        (out / 'answers.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
        (out / 'answers.md').write_text(markdown(report))
    try:
        for row, matches in prepared:
            print(f"Generating {row['case']} from pages {[m['metadata']['page'] for m in matches]}", flush=True)
            result = run_case(row, matches)
            report['results'].append(result)
            save()
            print(result['status'] + ': ' + result.get('answer', result.get('error', '')), flush=True)
    finally:
        report['test_collection_unchanged'] = fingerprint(read_records(collection)) == report['test_fingerprint_before']
        save()
        if not report['test_collection_unchanged']:
            raise RuntimeError('Test collection changed during generation; check concurrent modifications.')
    print(f'Answers: {out}', flush=True)
    if any(r['status'] == 'error' for r in report['results']):
        raise SystemExit('Some generations failed; see saved per-case errors.')


if __name__ == '__main__':
    main()
