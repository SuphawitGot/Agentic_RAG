import unittest

from evaluation.evaluate_extraction import evaluate


class ExtractionEvaluationTests(unittest.TestCase):
    def score(self, text, filename='report.pdf', page=8):
        reference = {'filename': 'report.pdf', 'page': 8, 'fields': [
            {'label': 'Image detection accuracy', 'value': '96.835', 'unit': '%'}]}
        return evaluate(reference, [{'metadata': {'filename': filename, 'page': page}, 'text': text}])

    def test_spacing_case_and_decorative_icons(self):
        self.assertEqual(self.score('image detection accuracy: 96.835 % *@')['correct_fields'], 1)

    def test_wrong_values_units_and_broken_association_fail(self):
        for text in ['Image detection accuracy: 96.335 %',
                     'Image detection accuracy: 96.835',
                     'Image detection accuracy: 96.835 kg',
                     'Image detection accuracy:\n96.835 %',
                     'Best achievable image detection accuracy: 96.835 %']:
            with self.subTest(text=text):
                self.assertEqual(self.score(text)['correct_fields'], 0)

    def test_correct_number_on_wrong_source_does_not_pass(self):
        text = 'Image detection accuracy: 96.835 %'
        self.assertEqual(self.score(text, page=4)['correct_fields'], 0)
        self.assertEqual(self.score(text, filename='another.pdf')['correct_fields'], 0)

    def test_conflicting_duplicate_fails(self):
        self.assertEqual(self.score('Image detection accuracy: 96.835 %\nImage detection accuracy: 99 %')['correct_fields'], 0)


if __name__ == '__main__':
    unittest.main()
