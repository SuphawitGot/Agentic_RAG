import tempfile
import unittest
from unittest.mock import patch

import chromadb
import numpy as np

from ragsale.rag.keyword_search import keyword_rank
from ragsale.rag.retrieval import retrieve


class KeywordSearchTests(unittest.TestCase):
    def test_literal_terms_do_not_translate_or_expand(self):
        records = [{'id': 'software', 'text': 'Software studio'},
                   {'id': 'technology', 'text': 'เทคโนโลยี'},
                   {'id': 'camera', 'text': 'Camera computer PLC servo motor'},
                   {'id': 'equipment', 'text': 'Equipment list'}]
        self.assertEqual(keyword_rank('software', records), ['software'])
        self.assertEqual(keyword_rank('เทคโนโลยี', records), ['technology'])
        self.assertEqual(keyword_rank('technology', records), [])
        self.assertEqual(keyword_rank('ซอฟต์แวร์', records), [])
        self.assertEqual(keyword_rank('equipment', records), ['equipment'])
        self.assertEqual(keyword_rank('ใช้อุปกรณ์อะไรบ้าง', records), [])

    def test_english_boundaries_and_no_match(self):
        records = [{'id': 'wrong', 'text': 'softwarehouse'}, {'id': 'right', 'text': 'Software studio'}]
        self.assertEqual(keyword_rank('software', records), ['right'])
        self.assertEqual(keyword_rank('unmentioned', records), [])

    def test_lexical_rescue_outside_dense_pool_preserves_scope(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = chromadb.PersistentClient(path=tmp).create_collection('hybrid_test', embedding_function=None)
            ids = [f'noise{i}' for i in range(13)] + ['software', 'other']
            c.upsert(ids=ids, documents=['unrelated cover'] * 13 + ['Software deep learning studio', 'software software'],
                     metadatas=[{'document_id': 'selected'}] * 14 + [{'document_id': 'other'}],
                     embeddings=[[1., 0.]] * 13 + [[0., 1.], [1., 0.]])
            with patch('ragsale.rag.retrieval.get_collection', return_value=c):
                matches = retrieve('software', 3, ['selected'], query_vector=np.array([1., 0.]))
            self.assertIn('software', [m['id'] for m in matches])
            self.assertTrue(all(m['metadata']['document_id'] == 'selected' for m in matches))
            self.assertAlmostEqual(next(m for m in matches if m['id'] == 'software')['distance'], 1.)

    def test_overlapping_weak_hits_do_not_displace_best_keyword_hit(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = chromadb.PersistentClient(path=tmp).create_collection('overlap_test', embedding_function=None)
            c.upsert(ids=[f'weak{i}' for i in range(13)] + ['software', 'technology'],
                     documents=['technology ' + 'general context ' * 100] * 13 + ['Software studio', 'Technology comparison'],
                     metadatas=[{'document_id': 'selected'}] * 15,
                     embeddings=[[1., 0.]] * 13 + [[0., 1.], [0., 1.]])
            with patch('ragsale.rag.retrieval.get_collection', return_value=c):
                matches = retrieve('software technology', 3, ['selected'], query_vector=np.array([1., 0.]))
            self.assertIn('software', [m['id'] for m in matches])
            self.assertIn('technology', [m['id'] for m in matches])
            self.assertEqual(len({m['id'] for m in matches}), 3)
