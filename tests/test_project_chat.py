import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import chromadb
import httpx
import numpy as np

from ragsale.api.main import app
from ragsale.rag.projects import list_projects, assign_project
from ragsale.rag.project_chat import project_chat, assess_projects, ProjectDecision
from ragsale.rag.answering import GenerationError
from ragsale.rag.retrieval import retrieve, retrieve_projects


PROJECTS = [
    {'project_id': 'cars', 'project_name': 'Car detection', 'documents': [{'document_id': 'car-pdf', 'filename': 'cars.pdf'}]},
    {'project_id': 'people', 'project_name': 'Person detection', 'documents': [{'document_id': 'person-pdf', 'filename': 'people.pdf'}]},
]
CANDIDATES = [{**p, 'matches': [{'id': p['project_id'], 'text': text,
    'metadata': {'document_id': p['documents'][0]['document_id'], 'page': 1}, 'distance': .2}]}
    for p, text in zip(PROJECTS, ['Camera A detects cars with 95% accuracy.', 'Camera A detects people with 88% accuracy.'])]


class ProjectRegistryTests(unittest.TestCase):
    def test_default_mapping_grouping_and_persistence(self):
        docs = [p['documents'][0] for p in PROJECTS]
        with tempfile.TemporaryDirectory() as tmp, patch('ragsale.rag.projects.PROJECT_DB', Path(tmp) / 'projects.sqlite3'), patch('ragsale.rag.projects.list_documents', return_value=docs):
            self.assertEqual(len(list_projects()), 2)
            one = assign_project('car-pdf', 'Shared project')
            two = assign_project('person-pdf', ' shared PROJECT ')
            self.assertEqual(one['project_id'], two['project_id'])
            self.assertEqual(len(list_projects()), 1)
            self.assertEqual(len(list_projects()[0]['documents']), 2)
            with self.assertRaises(ValueError):
                assign_project('missing', 'Shared project')


class RetrievalScopeTests(unittest.TestCase):
    def test_real_chroma_filter_and_balanced_project_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            collection = chromadb.PersistentClient(path=tmp).create_collection('project_test', embedding_function=None)
            collection.upsert(ids=['c1', 'c2', 'c3', 'p1'], documents=['car1', 'car2', 'car3', 'person1'],
                metadatas=[{'document_id': x} for x in ['car-pdf', 'car-pdf', 'car-pdf', 'person-pdf']],
                embeddings=[[1., 0.], [1., .01], [1., .02], [0., 1.]])
            with patch('ragsale.rag.retrieval.get_collection', return_value=collection), patch('ragsale.rag.retrieval.load_embedding_model') as model:
                model.return_value.encode_query.return_value = np.array([1., 0.])
                self.assertEqual([x['id'] for x in retrieve('Camera accuracy?', 3, ['person-pdf'])], ['p1'])
                self.assertEqual(retrieve('Camera accuracy?', 3, []), [])
                model.return_value.encode_query.reset_mock()
                groups = retrieve_projects('Camera accuracy?', PROJECTS, 2)
                self.assertEqual([len(p['matches']) for p in groups], [2, 1])
                self.assertEqual(groups[1]['matches'][0]['metadata']['project_id'], 'people')
                model.return_value.encode_query.assert_called_once()


class ProjectChatTests(unittest.TestCase):
    def setUp(self):
        self.registry = patch('ragsale.rag.project_chat.list_projects', return_value=PROJECTS).start()
        self.search = patch('ragsale.rag.project_chat.retrieve_projects', return_value=CANDIDATES).start()
        self.assess = patch('ragsale.rag.project_chat.assess_projects').start()
        self.generate = patch('ragsale.rag.project_chat.generate_answer', return_value={
            'answer': '88% [1]', 'sources': [], 'insufficient_evidence': False}).start()
        self.addCleanup(patch.stopall)

    def decision(self, relevant=None, explicit=None, compare=False, decision='answer'):
        self.assess.return_value = ProjectDecision(relevant_project_ids=relevant or ['cars', 'people'],
            explicit_project_ids=explicit or [], comparison_requested=compare,
            comparison_scope_clear=compare, decision=decision)

    def test_shared_camera_forces_clarification_even_if_model_says_answer(self):
        self.decision()
        result = project_chat('What is Camera A accuracy?')
        self.assertEqual(result['type'], 'clarification')
        self.assertEqual([p['project_id'] for p in result['project_options']], ['cars', 'people'])
        self.assertEqual(result['question'], 'What is Camera A accuracy?')
        self.generate.assert_not_called()

    def test_user_selection_bypasses_scope_model_and_filters_original_question(self):
        self.search.return_value = [CANDIDATES[1]]
        project_chat('What is Camera A accuracy?', selected_project_ids=['people'])
        self.assess.assert_not_called()
        self.search.assert_called_once_with('What is Camera A accuracy?', [PROJECTS[1]], 3)
        self.generate.assert_called_once_with('What is Camera A accuracy?', CANDIDATES[1]['matches'])

    def test_named_project_and_explicit_comparison(self):
        self.decision(explicit=['people'])
        project_chat('What is Camera A accuracy for detecting people?')
        self.assertEqual(self.search.call_args.args[1], [PROJECTS[1]])
        self.decision(compare=True)
        self.assertEqual(project_chat('Compare Camera A accuracy across projects')['type'], 'answer')
        self.assertEqual(self.search.call_args.args[1], PROJECTS)
        self.decision(compare=True, decision='clarify')
        self.assertEqual(project_chat('Compare both projects')['type'], 'answer')

    def test_no_relevant_evidence_and_scope_model_failure(self):
        self.assess.return_value = ProjectDecision(relevant_project_ids=[], explicit_project_ids=[], comparison_requested=False, decision='insufficient')
        self.assertTrue(project_chat('Unknown subject?')['insufficient_evidence'])
        self.generate.assert_not_called()
        self.assess.side_effect = GenerationError('invalid choice')
        result = project_chat('Accuracy?')
        self.assertEqual(result['type'], 'clarification')
        self.assertEqual(result['reason'], 'scope_unavailable')

    def test_stale_selection_and_conflicting_scope_rejected(self):
        with self.assertRaises(ValueError):
            project_chat('Accuracy?', selected_project_ids=['deleted'])
        with self.assertRaises(ValueError):
            project_chat('Accuracy?', selected_project_ids=['cars'], search_all=True)
        self.generate.assert_not_called()

    def test_greeting_empty_catalog_and_budget(self):
        self.assertIn('Hello', project_chat('hello')['answer'])
        self.search.assert_not_called()
        self.registry.return_value = []
        self.assertTrue(project_chat('Accuracy?')['insufficient_evidence'])
        self.registry.return_value = [{**PROJECTS[0], 'project_id': str(i)} for i in range(13)]
        self.assertEqual(project_chat('Accuracy?')['reason'], 'catalog_limit')
        self.search.assert_not_called()


class ScopeModelContractTests(unittest.TestCase):
    def test_valid_and_invented_project_ids(self):
        for ids, valid in [(['cars', 'people'], True), (['invented'], False)]:
            content = {'relevant_project_ids': ids, 'explicit_project_ids': [],
                       'comparison_requested': False, 'decision': 'clarify'}
            response = io.BytesIO(json.dumps({'message': {'content': json.dumps(content)}}).encode())
            with patch('urllib.request.urlopen', return_value=response) as call:
                if valid:
                    self.assertEqual(assess_projects('Camera A accuracy?', CANDIDATES).decision, 'clarify')
                    payload = json.loads(call.call_args.args[0].data)
                    self.assertIn('95%', payload['messages'][1]['content'])
                    self.assertIn('88%', payload['messages'][1]['content'])
                else:
                    with self.assertRaises(GenerationError):
                        assess_projects('Camera A accuracy?', CANDIDATES)


class ProjectChatAPITests(unittest.IsolatedAsyncioTestCase):
    async def test_clarification_and_followup(self):
        with patch('ragsale.rag.project_chat.list_projects', return_value=PROJECTS), patch(
            'ragsale.rag.project_chat.retrieve_projects', return_value=CANDIDATES
        ), patch('ragsale.rag.project_chat.assess_projects', return_value=ProjectDecision(
            relevant_project_ids=['cars', 'people'], explicit_project_ids=[], comparison_requested=False, decision='clarify'
        )), patch('ragsale.rag.project_chat.answer_in_projects', return_value={
            'type': 'answer', 'answer': '88%', 'sources': [], 'insufficient_evidence': False
        }) as answer:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
                question = 'Camera A accuracy?'
                first = await client.post('/api/chat', json={'question': question})
                self.assertEqual(first.status_code, 200)
                self.assertEqual(first.json()['type'], 'clarification')
                second = await client.post('/api/chat', json={'question': question, 'selected_project_ids': ['people']})
                self.assertEqual(second.json()['answer'], '88%')
                answer.assert_called_once_with(question, 3, [PROJECTS[1]])
                invalid = await client.post('/api/chat', json={'question': question, 'selected_project_ids': ['missing']})
                self.assertEqual(invalid.status_code, 422)
