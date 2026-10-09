import unittest
from evaluation.evaluate_chunking import evaluate_chunks


class ChunkingEvaluationTests(unittest.TestCase):
    def score(self, source, texts, references=(), contexts=(), **kwargs):
        page={'text':source,'metadata':{'filename':'a.pdf','page':1,'document_id':'a'}}
        chunks=[{'text':text,'metadata':{**page['metadata'],'chunk_index':i}} for i,text in enumerate(texts)]
        return evaluate_chunks([page],chunks,references,contexts,**kwargs)

    def test_overlap_and_stripped_whitespace_preserve_coverage(self):
        result=self.score('  Alpha\nBeta\nGamma  ',['Alpha\nBeta','Beta\nGamma'])
        self.assertTrue(result['passed'])

    def test_one_repeated_occurrence_cannot_cover_two(self):
        result=self.score('Repeat\nRepeat',['Repeat'])
        self.assertFalse(result['integrity_passed'])
        self.assertEqual(result['page_checks'][0]['missing_nonwhitespace_characters'],6)
        self.assertTrue(self.score('Repeat\nRepeat',['Repeat','Repeat'])['integrity_passed'])

    def test_deleted_and_invented_text_fail(self):
        for texts in (['Alpha'],['Alpha','Invented']):
            with self.subTest(texts=texts):
                self.assertFalse(self.score('Alpha Beta',texts)['integrity_passed'])

    def test_wrong_source_metadata_fails(self):
        page={'text':'Alpha','metadata':{'filename':'a.pdf','page':1,'document_id':'a'}}
        chunk={'text':'Alpha','metadata':{'filename':'a.pdf','page':2,'document_id':'a','chunk_index':0}}
        self.assertFalse(evaluate_chunks([page],[chunk],[])['integrity_passed'])
        chunk['metadata'].update(page=1,document_id='wrong')
        self.assertFalse(evaluate_chunks([page],[chunk],[])['integrity_passed'])

    def test_split_field_fails_even_when_characters_survive(self):
        ref={'filename':'a.pdf','page':1,'fields':[{'label':'Distance','value':'125','unit':'mm'}]}
        result=self.score('Distance: 125 mm',['Distance:','125 mm'],[ref])
        self.assertTrue(result['integrity_passed'])
        self.assertTrue(result['field_checks'][0]['source_passed'])
        self.assertFalse(result['field_checks'][0]['passed'])

    def test_conflicting_field_does_not_pass_due_to_one_good_chunk(self):
        ref={'filename':'a.pdf','page':1,'fields':[{'label':'Distance','value':'125','unit':'mm'}]}
        result=self.score('Distance: 125 mm\nDistance: 999 mm',['Distance: 125 mm','Distance: 999 mm'],[ref])
        self.assertFalse(result['field_checks'][0]['passed'])

    def test_heading_separated_from_value_fails_context_check(self):
        context={'id':'context','filename':'a.pdf','page':1,'required_phrases':['Hardware','Distance: 125 mm']}
        result=self.score('Hardware\nDistance: 125 mm',['Hardware','Distance: 125 mm'],contexts=[context])
        self.assertTrue(result['integrity_passed'])
        self.assertTrue(result['context_checks'][0]['source_passed'])
        self.assertFalse(result['context_checks'][0]['passed'])

    def test_oversized_chunk_and_empty_chunk_fail(self):
        self.assertFalse(self.score('ABCDE',['ABCDE'],max_size=4)['integrity_passed'])
        self.assertFalse(self.score('A',['A',''])['integrity_passed'])

    def test_invalid_chunk_indices_fail(self):
        page={'text':'Alpha','metadata':{'filename':'a.pdf','page':1}}
        chunk={'text':'Alpha','metadata':{**page['metadata'],'chunk_index':4}}
        self.assertFalse(evaluate_chunks([page],[chunk],[])['integrity_passed'])


if __name__=='__main__':
    unittest.main()
