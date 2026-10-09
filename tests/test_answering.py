import io
import json
import unittest
from unittest.mock import patch
from ragsale.rag.answering import answer_question, generate_answer, GenerationError

MATCHES=[{'id':'one','text':'ABC bought 20 cameras.','metadata':{'filename':'sales.pdf','page':1},'distance':0.1}]


class AnswerTests(unittest.TestCase):
    def test_standalone_greetings_skip_retrieval_and_qwen(self):
        with patch('ragsale.rag.answering.retrieve') as search, patch('urllib.request.urlopen') as model:
            for question in ['hello', ' HELLO! ', 'Hi there.', 'good morning']:
                result = answer_question(question)
                self.assertIn('Hello!', result['answer'])
                self.assertEqual(result['sources'], [])
            search.assert_not_called()
            model.assert_not_called()

    def test_greeting_with_question_still_uses_retrieval(self):
        with patch('ragsale.rag.answering.retrieve', return_value=MATCHES) as search, patch('ragsale.rag.answering.generate_answer', return_value={'answer': 'test'}) as model:
            question = 'Hello, how many cameras did ABC buy?'
            answer_question(question)
            search.assert_called_once_with(question, 3)
            model.assert_called_once_with(question, MATCHES)

    def test_document_answer_still_requires_citations(self):
        with patch('urllib.request.urlopen', return_value=self.response('20 cameras', [])):
            with self.assertRaises(GenerationError):
                generate_answer('How many cameras?', MATCHES)

    def response(self, answer, citations, insufficient=False):
        return io.BytesIO(json.dumps({'message':{'content':json.dumps({'answer':answer,'citations':citations,'insufficient_evidence':insufficient})}}).encode())

    def test_valid_citation(self):
        with patch('urllib.request.urlopen', return_value=self.response('ABC bought 20 cameras [1].',[1])) as call:
            result=generate_answer('How many cameras?',MATCHES)
            self.assertEqual(result['sources'][0]['citation'],1)
            payload=json.loads(call.call_args.args[0].data)
            self.assertFalse(payload['stream'])
            self.assertIn('20 cameras',payload['messages'][1]['content'])

    def test_invalid_citation_and_abstention(self):
        with patch('urllib.request.urlopen', return_value=self.response('20 cameras [9].',[9])):
            with self.assertRaises(GenerationError):generate_answer('How many?',MATCHES)
        with patch('urllib.request.urlopen', return_value=self.response('Unknown',[],True)):
            self.assertTrue(generate_answer('Why?',MATCHES)['insufficient_evidence'])
        with patch('urllib.request.urlopen') as call:
            self.assertEqual(generate_answer('What?',[])['sources'],[])
            call.assert_not_called()

    def test_structured_citations_without_inline_markers(self):
        with patch('urllib.request.urlopen', return_value=self.response('20',[1])):
            result=generate_answer('How many?',MATCHES)
            self.assertEqual(result['answer'],'20 [1]')

    def test_thai_context_is_readable_in_prompt(self):
        matches = [dict(MATCHES[0], text="รวมไฟล์ object เป็น executable") ]
        with patch('urllib.request.urlopen', return_value=self.response('Combines object files.', [1])) as call:
            generate_answer('What does linker do?', matches)
            payload = json.loads(call.call_args.args[0].data)
            self.assertIn('รวมไฟล์', payload['messages'][1]['content'])
            self.assertNotIn('\\u0e', payload['messages'][1]['content'])
