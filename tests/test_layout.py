import unittest
from types import SimpleNamespace
from ragsale.rag.layout import clean_cell, extract_layout, positioned_lines, group_caption_columns


def word(text, x, right, top=10, color=(0,0,0), size=12):
    return dict(text=text, x0=x, x1=right, top=top, bottom=top+size,
                size=size, non_stroking_color=color)


class LayoutTests(unittest.TestCase):
    def test_multiline_column_captions_stay_together(self):
        blocks = [(120,40,'Software deep'), (123,337,'Industrial computer'),
                  (123,638,'Industrial Machine Vision'), (149,40,'learning studio'),
                  (152,337,'24/7 operation'), (152,638,'Camera & Lighting')]
        result = group_caption_columns(blocks, 24)
        self.assertEqual([b[2] for b in result], ['Software deep learning studio',
            'Industrial computer 24/7 operation', 'Industrial Machine Vision Camera & Lighting'])

    def test_unrelated_or_distant_lines_are_not_joined(self):
        for blocks in [[(10,10,'Heading'), (50,10,'First paragraph')],
                       [(10,10,'Left'), (10,200,'Right'), (400,10,'Footer'), (400,200,'Other footer')]]:
            self.assertEqual(group_caption_columns(blocks,12), blocks)

    def test_wrapped_cells_preserve_numbers_and_join_label_fragments(self):
        self.assertEqual(clean_cell('Lighting Working\nDistance'), 'Lighting Working Distance')
        self.assertEqual(clean_cell('Inspection Match r\nate(%)'), 'Inspection Match rate(%)')
        self.assertEqual(clean_cell('Number of camera\ns'), 'Number of cameras')
        self.assertEqual(clean_cell('0.05\nsec/ea'), '0.05 sec/ea')
        self.assertEqual(clean_cell('a\nvalue'), 'a value')

    def test_position_and_color_separate_nearby_unrelated_labels(self):
        words = [word('CAM3',0,30,color=(1,0,0)),word(':',33,35,color=(1,0,0)),
                 word('12MP',38,60,color=(1,0,0)),word('Speed up conveyor',62,160)]
        texts=[line[2] for line in positioned_lines(words)]
        self.assertEqual(texts,['CAM3 : 12MP','Speed up conveyor'])

    def test_equals_links_specification_across_column_gap(self):
        words=[word('WD',0,20),word('=',100,105),word('800',110,130),word('mm',134,150)]
        self.assertEqual(positioned_lines(words)[0][2],'WD: 800 mm')

    def test_thai_prose_preserved_and_table_value_not_invented(self):
        rows=[['Label','Value'],['Distance\nlimit','125 mm']]
        table=SimpleNamespace(extract=lambda:rows)
        page=SimpleNamespace(find_tables=lambda:[table],chars=[])
        original='ภาษาไทย\nDistance\nlimit 125 mm'
        text,meta=extract_layout(page,original)
        self.assertEqual(text,'ภาษาไทย\nDistance limit: 125 mm')
        wrong,_=extract_layout(page,'ภาษาไทย\nDistance limit 999 mm')
        self.assertIn('999 mm',wrong)
        self.assertNotIn('125',wrong)
        self.assertEqual(meta['repaired_rows'],1)


if __name__=='__main__':unittest.main()
