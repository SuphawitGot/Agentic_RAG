import unittest
import numpy as np
from evaluation.compare_embeddings import separate_paths, check_vectors, fingerprint, metrics, evaluate_baseline


class EmbeddingComparisonTests(unittest.TestCase):
    def test_live_database_and_nested_paths_are_rejected(self):
        for target in ['/tmp/live', '/tmp/live/child', '/tmp']:
            with self.assertRaises(ValueError):
                separate_paths('/tmp/live', target)
        self.assertEqual(str(separate_paths('/tmp/live', '/tmp/test-db')), '/tmp/test-db')

    def test_invalid_vectors_are_rejected(self):
        for vectors in [[[0., 0.]], [[float('nan'), 0.]], [[2., 0.]], [[1., 0., 0.]]]:
            with self.assertRaises(ValueError):
                check_vectors(vectors, 1, 2)
        self.assertEqual(check_vectors([[1., 0.]], 1, 2).shape, (1, 2))

    def test_fingerprint_includes_vectors_metadata_and_text(self):
        original = {'a': {'text': 'hello', 'metadata': {'page': 9}, 'vector': [1., 0.]}}
        for key, value in [('text', 'changed'), ('metadata', {'page': 2}), ('vector', [0., 1.])]:
            changed = {'a': {**original['a'], key: value}}
            self.assertNotEqual(fingerprint(original), fingerprint(changed))

    def test_metrics_do_not_count_wrong_pdf_or_duplicate_pages(self):
        case = {'filename': 'expected.pdf', 'expected_pages': [5, 6]}
        hits = [{'filename': 'other.pdf', 'page': 6}, {'filename': 'expected.pdf', 'page': 5},
                {'filename': 'expected.pdf', 'page': 5}]
        self.assertEqual(metrics(hits, case, 1), {'hit': False, 'page_recall': 0})
        self.assertEqual(metrics(hits, case, 3), {'hit': True, 'page_recall': .5})

    def test_scoped_and_global_baselines_use_same_saved_vectors(self):
        records = {'a': {'text': 'target', 'metadata': {'filename': 'target.pdf', 'page': 9}, 'vector': [0., 1.]},
                   'b': {'text': 'other', 'metadata': {'filename': 'other.pdf', 'page': 1}, 'vector': [1., 0.]}}
        cases = [{'id': 'case', 'question': 'q', 'filename': 'target.pdf', 'expected_pages': [9]}]
        before = fingerprint(records)
        rows = evaluate_baseline(records, cases, np.array([[1., 0.]]))
        self.assertEqual(rows[0]['top5'][0]['id'], 'a')
        self.assertEqual(rows[1]['top5'][0]['id'], 'b')
        self.assertEqual(fingerprint(records), before)
