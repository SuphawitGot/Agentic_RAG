from pathlib import Path
import unittest

import numpy as np

from evaluation.benchmark_embeddings import CASES, load_cases, rank, score, MODELS


class EmbeddingBenchmarkTests(unittest.TestCase):
    def test_case_file_is_consistent(self):
        cases = load_cases(CASES)
        self.assertEqual({p['lang'] for p in cases['passages']}, {'en', 'th'})
        self.assertEqual({q['group'] for q in cases['queries']}, {'en-en', 'th-th', 'th-en', 'en-th', 'mixed'})

    def test_broken_case_file_is_rejected(self):
        bad = {'passages': [{'id': 'p1', 'lang': 'en', 'text': 'x'}],
               'queries': [{'id': 'q1', 'group': 'en-en', 'text': 'y', 'expected': ['p9']}]}
        with self.assertRaises(ValueError):
            load_cases(bad)

    def test_rank_orders_by_cosine_similarity(self):
        passages = np.array([[1., 0.], [0., 1.], [0.6, 0.8]])
        self.assertEqual(rank(np.array([[0., 1.]]), passages, ['a', 'b', 'c']), [['b', 'c', 'a']])

    def test_score_reports_hits_and_mrr_per_group(self):
        queries = [{'id': 'q1', 'group': 'en-en', 'expected': ['a']},
                   {'id': 'q2', 'group': 'th-th', 'expected': ['b']},
                   {'id': 'q3', 'group': 'th-th', 'expected': ['z']}]
        result = score([['a', 'b'], ['a', 'b'], ['a', 'b']], queries)
        self.assertEqual(result['en-en'], {'n': 1, 'hit@1': 1.0, 'hit@3': 1.0, 'mrr': 1.0})
        self.assertEqual(result['th-th'], {'n': 2, 'hit@1': 0.0, 'hit@3': 0.5, 'mrr': 0.25})
        self.assertEqual(result['all']['n'], 3)
        self.assertEqual(result['misses'], ['q2', 'q3'])

    def test_default_models_include_the_live_one(self):
        from agentic_rag.rag.embedding import MODEL_NAME
        self.assertIn(MODEL_NAME, MODELS)


if __name__ == '__main__':
    unittest.main()
