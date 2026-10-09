import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
import numpy as np
from ragsale.rag.embedding import embed_chunks, MODEL_NAME
from ragsale.rag.retrieval import retrieve, retrieve_projects


class EmbeddingTests(unittest.TestCase):
    def test_documents_use_document_encoder(self):
        model = Mock()
        model.encode_document.return_value = np.zeros((1, 1024))
        with patch('ragsale.rag.embedding.load_embedding_model', return_value=model):
            embed_chunks([SimpleNamespace(page_content='Camera')])
        model.encode_document.assert_called_once_with(['Camera'], normalize_embeddings=True,
            convert_to_numpy=True, show_progress_bar=True, batch_size=2)
        model.encode.assert_not_called()
        self.assertEqual(MODEL_NAME, 'codefuse-ai/F2LLM-v2-0.6B')

    def test_search_uses_query_encoder(self):
        model, collection = Mock(), Mock()
        model.encode_query.return_value = np.ones(1024)
        collection.count.return_value = 1
        collection.query.return_value = {'ids': [['one']], 'documents': [['Camera']],
            'metadatas': [[{'page': 9}]], 'distances': [[.2]]}
        with patch('ragsale.rag.retrieval.load_embedding_model', return_value=model), patch('ragsale.rag.retrieval.get_collection', return_value=collection):
            retrieve('Equipment?')
        model.encode_query.assert_called_once_with('Equipment?', normalize_embeddings=True)
        model.encode.assert_not_called()

    def test_project_search_encodes_question_once(self):
        model = Mock()
        model.encode_query.return_value = np.ones(1024)
        projects = [{'project_id': 'p', 'project_name': 'P', 'documents': [{'document_id': 'd'}]}]
        with patch('ragsale.rag.retrieval.load_embedding_model', return_value=model), patch('ragsale.rag.retrieval.retrieve', return_value=[]) as search:
            retrieve_projects('Equipment?', projects)
        model.encode_query.assert_called_once_with('Equipment?', normalize_embeddings=True)
        self.assertIs(search.call_args.kwargs['query_vector'], model.encode_query.return_value)
