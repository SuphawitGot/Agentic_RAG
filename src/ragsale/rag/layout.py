"""Conservative native-PDF layout recovery; never infer missing numeric values."""
import re
from statistics import median


def clean_cell(text):
    lines = [line.strip() for line in (text or '').splitlines() if line.strip()]
    result = ''
    for line in lines:
        # Recover narrow-cell wrapping inside a Latin word (r / ate, camera / s).
        tail = result.split()[-1] if result else ''
        head = line.split()[0]
        join_word = (re.fullmatch('[a-z]+', tail or ' ') and
                     re.match('[a-z]', head) and
                     ((len(tail) == 1 and tail not in {'a', 'i'}) or head == 's'))
        result += ('' if not result or join_word else ' ') + line
    return result


def table_rows(page):
    accepted = []
    for table in page.find_tables():
        rows = table.extract()
        # Exclude slide borders, diagram boxes, and degenerate one-column grids.
        if len(rows) < 2 or sum(sum(bool(c) for c in row) >= 2 for row in rows) < 2:
            continue
        accepted.append((table, rows))
    return accepted


def positioned_lines(words):
    lines = []
    for word in sorted(words, key=lambda w: (w['top'], w['x0'])):
        candidates = [line for line in lines if abs(line[0]['top'] - word['top']) <= 3]
        if candidates:
            candidates[-1].append(word)
        else:
            lines.append([word])
    output = []
    for line in lines:
        segments = []
        for word in sorted(line, key=lambda w: w['x0']):
            previous = segments[-1][-1] if segments else None
            same_style = previous and (previous.get('non_stroking_color') == word.get('non_stroking_color')
                                       and abs(previous.get('size', 12) - word.get('size', 12)) < 1)
            gap = word['x0'] - previous['x1'] if previous else 0
            if previous and same_style and (gap < max(20, word.get('size', 12)*2) or word['text'] == '='):
                segments[-1].append(word)
            else:
                segments.append([word])
        for segment in segments:
            text = ' '.join(w['text'] for w in segment)
            # Equal signs explicitly bind a label to its value on the same line.
            match = re.fullmatch(r'\s*[-•]?\s*([^=]+?)\s*=\s*(.+)', text)
            if match:
                text = match[1].strip() + ': ' + match[2].strip()
            output.append((min(w['top'] for w in segment), min(w['x0'] for w in segment), text))
    return output


def group_caption_columns(blocks, font_size):
    """Keep short, aligned multi-line captions together on simple column pages.

    Only accept a complete set of 2–4 non-overlapping vertical caption groups.
    Other layouts retain the existing reading order.
    """
    groups = []
    for block in sorted(blocks):
        group = next((g for g in groups if abs(g[0][1] - block[1]) <= 3), None)
        if group is None:
            groups.append([block])
        else:
            group.append(block)
    if not 2 <= len(groups) <= 4 or any(not 2 <= len(g) <= 3 for g in groups):
        return blocks
    if max(g[0][0] for g in groups) - min(g[0][0] for g in groups) > 4:
        return blocks
    if any(not 0 < b[0] - a[0] <= font_size * 1.6 for g in groups for a, b in zip(g, g[1:])):
        return blocks
    return [(g[0][0], g[0][1], ' '.join(b[2] for b in g)) for g in groups]


def extract_layout(page, native_text):
    tables = table_rows(page)
    rotated = any(abs(c.get('matrix', (1, 0))[1]) > abs(c.get('matrix', (1, 0))[0]) for c in page.chars)
    if re.search('[\u0e00-\u0e7f]', native_text) or rotated:
        # pdfminer's spatial Thai character sorting can reorder combining marks.
        # Rotated letters also risk splitting into separate words. Retain pypdf
        # prose; repair only exact Latin table sequences in that text.
        text = native_text
        repaired = 0
        for _, rows in tables:
            for row in rows:
                if len(row) != 2 or not all(row) or any(re.search('[\u0e00-\u0e7f]', c) for c in row):
                    continue
                label, value = map(clean_cell, row)
                chars = ''.join(''.join(row).split())
                pattern = r'\s*'.join(re.escape(c) for c in chars)
                text, count = re.subn(pattern, lambda _: label + ': ' + value, text, count=1)
                repaired += count
        return text, {'layout_method': 'native_text_with_table_repair', 'table_count': len(tables), 'repaired_rows': repaired}

    def inside(word, bbox):
        x = (word['x0'] + word['x1']) / 2
        y = (word['top'] + word['bottom']) / 2
        return bbox[0] <= x <= bbox[2] and bbox[1] <= y <= bbox[3]
    words = page.extract_words(extra_attrs=['non_stroking_color'], return_chars=True)
    for word in words:
        word['size'] = median(char['size'] for char in word.pop('chars'))
    outside = [w for w in words if not any(inside(w, table.bbox) for table, _ in tables)]
    blocks = positioned_lines(outside)
    if not tables and outside:
        grouped = group_caption_columns(blocks, median(w['size'] for w in outside))
        # Native reading order must corroborate each joined caption, so adjacent
        # table-like label/value rows are not accidentally read as columns.
        normalized_native = ' '.join(native_text.split())
        if all(' '.join(block[2].split()) in normalized_native for block in grouped):
            blocks = grouped
    for table, rows in tables:
        rendered = []
        for row in rows:
            cells = [clean_cell(cell) for cell in row]
            if len(cells) == 2 and all(cells):
                rendered.append(cells[0] + ': ' + cells[1])
            else:
                rendered.append(' | '.join(cell for cell in cells if cell))
        blocks.append((table.bbox[1], table.bbox[0], '\n'.join(rendered)))
    text = '\n'.join(block[2] for block in sorted(blocks))
    return text or native_text, {'layout_method': 'positioned_text_and_tables', 'table_count': len(tables), 'repaired_rows': 0}
