import json
import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessage, ToolMessage

from ragsale.agent import graph
from ragsale.agent.evidence_tools import Claim

PROJECTS = [
    {'project_id': 'p1', 'project_name': 'Demo Line A',
     'documents': [{'document_id': 'd1', 'filename': 'line-a.pdf'}]},
    {'project_id': 'p2', 'project_name': 'Demo Line B',
     'documents': [{'document_id': 'd2', 'filename': 'line-b.pdf'}]},
]
TEXT = {'p1': 'The station uses a BASLER acA1300 camera with ring lighting at 24 V.',
        'p2': 'Pipe diameter is 12.7 mm and the wall thickness is 0.8 mm.'}


def fake_retrieve(query, projects, top_k):
    return [{**p, 'matches': [{
        'id': 'chunk-' + p['project_id'], 'text': TEXT[p['project_id']], 'distance': 0.2,
        'metadata': {'document_id': p['documents'][0]['document_id'], 'filename': p['documents'][0]['filename'],
                     'page': 3, 'project_id': p['project_id'], 'project_name': p['project_name']}}]}
        for p in projects]


def call(name, **args):
    return {'name': name, 'args': args, 'id': f'call-{name}-{len(json.dumps(args))}'}


def ai(*calls, text=''):
    return AIMessage(content=text, tool_calls=list(calls))


class ScriptedModel:
    """Stands in for the chat model: replies from a script and records the offered tools."""
    def __init__(self, *replies):
        self.replies, self.offered, self.inputs = list(replies), [], []

    def bind_tools(self, tools):
        self.offered.append({t['function']['name'] for t in tools})
        return self

    def invoke(self, messages):
        self.inputs.append(messages)
        return self.replies.pop(0)


def run(model, **kwargs):
    with patch.object(graph, '_catalog', return_value=PROJECTS), \
            patch('ragsale.agent.evidence_tools._retrieve_projects', side_effect=fake_retrieve):
        return graph.graph_chat(kwargs.pop('question', 'Which camera is used?'), model=model, **kwargs)


def last_tool_reply(model):
    """The most recent tool result the model saw."""
    content = [m for m in model.inputs[-1] if isinstance(m, ToolMessage)][-1].content
    return json.loads(content) if content.startswith('{') else content


class AgentGraphTests(unittest.TestCase):
    def test_search_then_checked_answer(self):
        model = ScriptedModel(
            ai(call('search_project', project_id='p1', query='camera')),
            ai(call('submit_answer', claims=[
                {'text': 'It uses a BASLER acA1300 camera.', 'quote': 'BASLER acA1300 camera', 'citation': 1}])))
        result = run(model)
        self.assertEqual(result['type'], 'answer')
        self.assertIn('BASLER acA1300 camera. [1]', result['answer'])
        self.assertEqual([s['citation'] for s in result['sources']], [1])
        self.assertEqual(result['sources'][0]['metadata']['filename'], 'line-a.pdf')
        # Without evidence the model may only search; afterwards it may also finish.
        self.assertEqual(model.offered[0], set(graph.SEARCH_TOOLS))
        self.assertIn('submit_answer', model.offered[1])
        self.assertIn('ask_user', model.offered[1])

    def test_failed_claim_gets_one_retry_then_partial_answer(self):
        claims = [{'text': 'It uses a BASLER camera.', 'quote': 'BASLER acA1300 camera', 'citation': 1},
                  {'text': 'It runs at 48 V.', 'quote': 'ring lighting at 24 V', 'citation': 1}]
        model = ScriptedModel(ai(call('search_project', project_id='p1', query='camera')),
                              ai(call('submit_answer', claims=claims)),
                              ai(call('submit_answer', claims=claims)))
        result = run(model)
        self.assertIn('failed the check', last_tool_reply(model))
        self.assertIn('BASLER camera. [1]', result['answer'])
        self.assertNotIn('48 V', result['answer'])
        self.assertIn('Removed 1 statement', result['answer'])

    def test_repeated_search_is_blocked(self):
        model = ScriptedModel(ai(call('search_all_projects', query='camera')),
                              ai(call('search_all_projects', query='  CAMERA ')),
                              ai(call('report_insufficient')))
        result = run(model)
        self.assertEqual(last_tool_reply(model)['error'], 'duplicate')
        self.assertTrue(result['insufficient_evidence'])
        self.assertEqual(result['reason'], 'insufficient_evidence')

    def test_search_budget_leaves_only_finishing_tools(self):
        searches = [ai(call('search_all_projects', query=f'camera {i}')) for i in range(graph.MAX_TOOL_CALLS)]
        model = ScriptedModel(*searches, ai(call('report_insufficient')))
        run(model)
        self.assertEqual(model.offered[-1], set(graph.FINISH_TOOLS))

    def test_selected_project_is_a_hard_scope(self):
        model = ScriptedModel(ai(call('search_project', project_id='p2', query='diameter')),
                              ai(call('search_project', project_id='p1', query='camera')),
                              ai(call('ask_user', question='Which one?', project_ids=['p1', 'p2'])),
                              ai(call('report_insufficient')))
        run(model, selected_project_ids=['p1'])
        self.assertNotIn('ask_user', model.offered[-1])
        self.assertIn('Clarification is not valid', last_tool_reply(model))
        first_search = json.loads(model.inputs[1][-1].content)
        self.assertEqual(first_search['error'], 'unknown_project')

    def test_ask_user_returns_project_choices(self):
        model = ScriptedModel(ai(call('search_all_projects', query='accuracy')),
                              ai(call('ask_user', question='Which project?', project_ids=['p1', 'p2'])))
        result = run(model)
        self.assertEqual(result['type'], 'clarification')
        self.assertEqual([p['project_id'] for p in result['project_options']], ['p1', 'p2'])

    def test_plain_text_is_reminded_until_the_step_limit(self):
        model = ScriptedModel(*[ai(text='The camera is probably BASLER.') for _ in range(graph.MAX_MODEL_CALLS)])
        result = run(model)
        self.assertEqual(result['reason'], 'agent_limit')
        self.assertEqual(len(model.inputs), graph.MAX_MODEL_CALLS)
        self.assertEqual(model.inputs[-1][-1].content, graph.REMINDER)

    def test_empty_catalog_stops_before_the_model(self):
        model = ScriptedModel()
        with patch.object(graph, '_catalog', return_value=[]):
            result = graph.graph_chat('Anything?', model=model)
        self.assertEqual(result['reason'], 'empty_catalog')
        self.assertEqual(model.inputs, [])


class VerifyClaimsTests(unittest.TestCase):
    records = {1: {'text': 'Pipe diameter is 12.7 mm and the wall thickness is 0.8 mm.'}}

    def check(self, text, quote, citation=1):
        return graph.verify_claims([Claim(text=text, quote=quote, citation=citation)], self.records)

    def test_quote_must_be_in_the_cited_excerpt(self):
        self.assertEqual(self.check('Diameter is 12.7 mm.', 'diameter is 12.7 mm')[0], [('Diameter is 12.7 mm.', 1)])
        self.assertEqual(self.check('Diameter is 12.7 mm.', 'diameter is 12.7 mm', citation=2)[0], [])
        self.assertEqual(self.check('Length is 3 m.', 'length is 3 m')[0], [])

    def test_numbers_in_the_claim_must_be_in_the_quote(self):
        self.assertEqual(self.check('Diameter is 13 mm.', 'diameter is 12.7 mm')[0], [])


if __name__ == '__main__':
    unittest.main()
