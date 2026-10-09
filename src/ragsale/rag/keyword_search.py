"""Small lexical companion to dense retrieval within the selected document scope."""

import math
import re
import unicodedata

STOP_WORDS = set('a an the is are was were what which how can you me please tell summarize summary of in on for to and or use uses used test report document project ใช้ หรือ อะไร อะไรบ้าง บ้าง สรุป'.split())


def normalize(text):
    return unicodedata.normalize('NFKC', text).casefold()


def keyword_rank(question, records):
    """Return IDs with positive lexical evidence, ranked with length-normalized TF/IDF."""
    q = normalize(question)
    tokens = re.findall(r'[a-z0-9]+(?:[-.][a-z0-9]+)*|[\u0e00-\u0e7f]+', q)
    terms = set(t for t in tokens if t not in STOP_WORDS and len(t) > 1)
    if not terms or not records:
        return []
    texts = [normalize(r['text']) for r in records]
    lengths = [max(1, len(text)) for text in texts]
    average = sum(lengths) / len(lengths)
    scores = [0.0] * len(records)
    for term in sorted(terms):
        # Match query terms literally; do not translate or expand vocabulary.
        frequencies = [text.count(term) if re.search(r'[\u0e00-\u0e7f]', term)
                       else len(re.findall(r'(?<![a-z0-9])' + re.escape(term) + r'(?![a-z0-9])', text))
                       for text in texts]
        found = sum(freq > 0 for freq in frequencies)
        idf = math.log(1 + (len(records) - found + 0.5) / (found + 0.5))
        for i, freq in enumerate(frequencies):
            scores[i] += idf * freq * 2.2 / (freq + 1.2 * (0.25 + 0.75 * lengths[i] / average))
    return [records[i]['id'] for i in sorted(range(len(records)), key=lambda i: (-scores[i], records[i]['id'])) if scores[i] > 0]
