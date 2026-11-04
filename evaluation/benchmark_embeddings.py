"""Compare embedding models on made-up English/Thai cases (no private data, no Chroma).

uv run python -m evaluation.benchmark_embeddings
uv run --env-file .env python -m evaluation.benchmark_embeddings --models openai:text-embedding-3-small

Each model runs in its own process, so peak memory is measured per model.
Hosted models need their API key in the environment (see PROVIDERS).
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from urllib import request
import warnings

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / 'evaluation/cases/embedding_bilingual.json'
# Local models and the prefixes they expect. None means the model's own query/document prompts.
MODELS = {
    'codefuse-ai/F2LLM-v2-0.6B': None,
    'intfloat/multilingual-e5-small': ('query: ', 'passage: '),
    'intfloat/multilingual-e5-base': ('query: ', 'passage: '),
}
# Hosted models, written "provider:model". All use the OpenAI-compatible /embeddings API.
PROVIDERS = {
    'openai': ('https://api.openai.com/v1', 'OPENAI_API_KEY'),
    'gemini': ('https://generativelanguage.googleapis.com/v1beta/openai', 'GEMINI_API_KEY'),
    'jina': ('https://api.jina.ai/v1', 'JINA_API_KEY'),
}
GROUPS = ['en-en', 'th-th', 'th-en', 'en-th', 'mixed']


def load_cases(source):
    cases = source if isinstance(source, dict) else json.loads(Path(source).read_text(encoding='utf-8'))
    passage_ids = [p['id'] for p in cases['passages']]
    query_ids = [q['id'] for q in cases['queries']]
    if len(set(passage_ids)) != len(passage_ids) or len(set(query_ids)) != len(query_ids):
        raise ValueError('Passage and query IDs must be unique.')
    for query in cases['queries']:
        if not query['expected'] or set(query['expected']) - set(passage_ids):
            raise ValueError(f"Query {query['id']} expects an unknown passage.")
    return cases


def normalize(vectors):
    vectors = np.asarray(vectors, dtype=np.float32)
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def rank(query_vectors, passage_vectors, ids):
    similarity = normalize(query_vectors) @ normalize(passage_vectors).T
    return [[ids[i] for i in np.argsort(-row, kind='stable')] for row in similarity]


def score(rankings, queries):
    rows = []
    for ranking, query in zip(rankings, queries):
        position = next((n for n, i in enumerate(ranking, 1) if i in query['expected']), None)
        rows.append((query, position))

    def summary(selected):
        positions = [p for _, p in selected]
        return {'n': len(positions),
                'hit@1': sum(p == 1 for p in positions) / len(positions),
                'hit@3': sum(p is not None and p <= 3 for p in positions) / len(positions),
                'mrr': sum(1 / p for p in positions if p) / len(positions)}

    result = {group: summary([r for r in rows if r[0]['group'] == group])
              for group in GROUPS if any(r[0]['group'] == group for r in rows)}
    result['all'] = summary(rows)
    result['misses'] = [query['id'] for query, position in rows if position != 1]
    return result


class LocalModel:
    def __init__(self, name, batch_size):
        warnings.filterwarnings('ignore', category=UserWarning, module=r'torch\.cuda')
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(name, device='cpu', model_kwargs={'torch_dtype': 'float32'})
        self.prefixes, self.batch_size = MODELS.get(name), batch_size

    def encode(self, texts, kind):
        options = dict(batch_size=self.batch_size, normalize_embeddings=True, convert_to_numpy=True)
        if self.prefixes:
            prefix = self.prefixes[0] if kind == 'query' else self.prefixes[1]
            return self.model.encode([prefix + t for t in texts], **options)
        encode = self.model.encode_query if kind == 'query' else self.model.encode_document
        return encode(texts, **options)


class HostedModel:
    def __init__(self, spec, batch_size):
        provider, self.name = spec.split(':', 1)
        self.url, key_name = PROVIDERS[provider]
        self.key = os.environ.get(key_name)
        if not self.key:
            raise SystemExit(f'Set {key_name} to benchmark {spec}.')
        self.batch_size = max(batch_size, 32)

    def encode(self, texts, kind):
        vectors = []
        for start in range(0, len(texts), self.batch_size):
            body = json.dumps({'model': self.name, 'input': texts[start:start + self.batch_size]}).encode()
            call = request.Request(self.url + '/embeddings', data=body, headers={
                'Authorization': f'Bearer {self.key}', 'Content-Type': 'application/json'})
            with request.urlopen(call, timeout=60) as response:
                data = json.load(response)['data']
            vectors += [item['embedding'] for item in sorted(data, key=lambda item: item['index'])]
        return normalize(vectors)


def speed_texts(passages, count=32, size=1000):
    """Chunks about as long as the app's (1,000 characters), mixing both languages."""
    stream, chunks = ' '.join(p['text'] for p in passages), []
    while len(chunks) < count:
        start = (len(chunks) * 137) % len(stream)
        chunks.append((stream * 3)[start:start + size])
    return chunks


def peak_memory_mb():
    try:
        import resource
    except ImportError:  # Windows
        return None
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(peak / (1024 * 1024 if sys.platform == 'darwin' else 1024))


def run_one(spec, batch_size, cases):
    started = time.perf_counter()
    model = HostedModel(spec, batch_size) if spec.split(':', 1)[0] in PROVIDERS else LocalModel(spec, batch_size)
    load_seconds = time.perf_counter() - started
    passages, queries = cases['passages'], cases['queries']
    passage_vectors = model.encode([p['text'] for p in passages], 'document')
    started = time.perf_counter()
    query_vectors = np.vstack([model.encode([q['text']], 'query') for q in queries])
    query_ms = (time.perf_counter() - started) * 1000 / len(queries)
    chunks = speed_texts(passages)
    started = time.perf_counter()
    model.encode(chunks, 'document')
    chunks_per_second = len(chunks) / (time.perf_counter() - started)
    return {'model': spec, 'dimension': int(np.asarray(passage_vectors).shape[1]),
            'load_seconds': round(load_seconds, 1), 'query_ms': round(query_ms),
            'chunks_per_second': round(chunks_per_second, 2), 'peak_memory_mb': peak_memory_mb(),
            'scores': score(rank(query_vectors, passage_vectors, [p['id'] for p in passages]), queries)}


def table(results):
    lines = ['| Model | All hit@1 | All MRR | ' + ' | '.join(f'{g} hit@1' for g in GROUPS)
             + ' | Chunks/s | Query ms | Peak MB | Dim |',
             '|' + '---|' * (7 + len(GROUPS))]
    for r in results:
        s = r['scores']
        lines.append(f"| {r['model']} | {s['all']['hit@1']:.2f} | {s['all']['mrr']:.2f} | "
                     + ' | '.join(f"{s[g]['hit@1']:.2f}" for g in GROUPS)
                     + f" | {r['chunks_per_second']} | {r['query_ms']} | {r['peak_memory_mb']} | {r['dimension']} |")
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--models', nargs='+', default=list(MODELS))
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--cases', type=Path, default=CASES)
    parser.add_argument('--output', type=Path, default=ROOT / 'output/embedding-benchmark')
    parser.add_argument('--single', help=argparse.SUPPRESS)
    args = parser.parse_args()
    cases = load_cases(args.cases)
    if args.single:
        print(json.dumps(run_one(args.single, args.batch_size, cases), ensure_ascii=False))
        return

    results = []
    for spec in args.models:
        print(f'Running {spec} ...', flush=True)
        done = subprocess.run([sys.executable, '-m', 'evaluation.benchmark_embeddings', '--single', spec,
                               '--batch-size', str(args.batch_size), '--cases', str(args.cases)],
                              cwd=ROOT, capture_output=True, text=True, encoding='utf-8')
        if done.returncode:
            print(f'  failed: {(done.stderr or done.stdout).strip().splitlines()[-1]}')
            continue
        results.append(json.loads(done.stdout.strip().splitlines()[-1]))
        print(f"  misses (not ranked first): {', '.join(results[-1]['scores']['misses']) or 'none'}")
    if not results:
        return
    print('\n' + table(results))
    args.output.mkdir(parents=True, exist_ok=True)
    report = args.output / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '.json')
    report.write_text(json.dumps({'batch_size': args.batch_size, 'results': results},
                                 ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'\nSaved {report.relative_to(ROOT)}')


if __name__ == '__main__':
    main()
