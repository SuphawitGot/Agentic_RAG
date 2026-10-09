import unittest
from unittest.mock import patch

from ragsale.rag.project_names import named_projects
from ragsale.rag.project_chat import project_chat


def project(identity, name):
    return {'project_id': identity, 'project_name': name,
            'documents': [{'document_id': identity, 'filename': name + '.pdf'}]}


COPPER = project('copper', 'copper pipe')
WASHER = project('washer', 'IAI_Washer Inspection')


class ProjectNameTests(unittest.TestCase):
    def test_mixed_thai_name_can_omit_generic_report_word(self):
        medicine = project('medicine', 'Test Report ยา')
        self.assertEqual(named_projects('test ยา ใช้ software หรือ technology อะไรบ้าง', [medicine, WASHER]), [medicine])
        self.assertEqual(named_projects('test ยาง ใช้ software อะไร', [medicine, WASHER]), [])

    def test_shortened_washer_name_routes_directly(self):
        question = 'Summary washer inspection'
        self.assertEqual(named_projects(question, [COPPER, WASHER]), [WASHER])
        with patch('ragsale.rag.project_chat.list_projects', return_value=[COPPER, WASHER]), patch(
            'ragsale.rag.project_chat.answer_in_projects', return_value={'type': 'answer'}
        ) as answer, patch('ragsale.rag.project_chat.assess_projects') as assess:
            self.assertEqual(project_chat(question)['type'], 'answer')
            answer.assert_called_once_with(question, 3, [WASHER])
            assess.assert_not_called()

    def test_shared_shortened_name_still_asks_and_full_name_wins(self):
        other = project('other', 'ABC Washer Inspection')
        self.assertEqual(named_projects('Summary washer inspection', [WASHER, other]), [WASHER, other])
        self.assertEqual(named_projects('Summary IAI Washer Inspection', [WASHER, other]), [WASHER])
        with patch('ragsale.rag.project_chat.list_projects', return_value=[COPPER, WASHER, other]):
            result = project_chat('Summary washer inspection')
            self.assertEqual(result['type'], 'clarification')
            self.assertEqual([p['project_id'] for p in result['project_options']], ['washer', 'other'])

    def test_generic_shortened_phrase_does_not_select_a_project(self):
        self.assertEqual(named_projects('Summarize inspection report', [project('x', 'IAI Inspection Report')]), [])
        self.assertEqual(named_projects('Summary washer', [WASHER]), [])

    def test_exact_question_and_normalized_names(self):
        for question in ['Can you summarize Scope of Work of copper pipe test ?',
                         'Summarize COPPER-PIPE.pdf', 'Summarize copper_pipe test.',
                         'Summarize copper   pipe']:
            self.assertEqual(named_projects(question, [COPPER, WASHER]), [COPPER])

    def test_partial_shared_words_do_not_establish_scope(self):
        for question in ['What is the pipe accuracy?', 'What is Camera A accuracy?',
                         'Summarize copper pipeline', 'What are inspection results?']:
            self.assertEqual(named_projects(question, [COPPER, WASHER]), [])
        self.assertEqual(named_projects('Summarize the report', [project('generic', 'report')]), [])

    def test_comparison_and_exclusion_do_not_narrow_to_one_project(self):
        for question in ['Compare copper pipe with another project',
                         'All projects except copper pipe', 'Not copper pipe',
                         'Compare copper pipe versus IAI Washer Inspection']:
            self.assertEqual(named_projects(question, [COPPER, WASHER]), [])

    def test_filename_alias_and_duplicate_names(self):
        renamed = {**COPPER, 'project_name': 'Customer project 1'}
        self.assertEqual(named_projects('Summarize copper pipe', [renamed]), [renamed])
        duplicate = project('copper2', 'copper pipe')
        self.assertEqual(named_projects('Summarize copper pipe', [COPPER, duplicate]), [COPPER, duplicate])

    def test_named_route_bypasses_scope_model_even_with_200_projects(self):
        projects = [COPPER] + [project(str(i), f'Unrelated project {i}') for i in range(199)]
        question = 'Can you summarize Scope of Work of copper pipe test ?'
        with patch('ragsale.rag.project_chat.list_projects', return_value=projects), patch(
            'ragsale.rag.project_chat.answer_in_projects', return_value={'type': 'answer', 'answer': 'Summary'}
        ) as answer, patch('ragsale.rag.project_chat.assess_projects') as assess, patch(
            'ragsale.rag.project_chat.retrieve_projects'
        ) as candidates:
            self.assertEqual(project_chat(question)['type'], 'answer')
            answer.assert_called_once_with(question, 3, [COPPER])
            assess.assert_not_called()
            candidates.assert_not_called()

    def test_duplicate_name_asks_only_matching_projects(self):
        duplicate = project('copper2', 'copper pipe')
        with patch('ragsale.rag.project_chat.list_projects', return_value=[COPPER, WASHER, duplicate]), patch(
            'ragsale.rag.project_chat.answer_in_projects'
        ) as answer:
            result = project_chat('Summarize copper pipe')
            self.assertEqual(result['type'], 'clarification')
            self.assertEqual([p['project_id'] for p in result['project_options']], ['copper', 'copper2'])
            answer.assert_not_called()
