"""Conservative document-scope decisions; no retrieval or generation here."""

import re
import unicodedata
from pathlib import PurePath


GENERIC_WORDS = {
    'pdf', 'the', 'and', 'of', 'test', 'report', 'document', 'inspection',
    'result', 'results', 'auto', 'label', 'revise', 'version',
}


def words(text):
    return re.findall(r'[^\W_]+', unicodedata.normalize('NFKC', text).casefold())


def decide_scope(question, documents, selected_document_ids=None, search_all=False):
    """Resolve explicit scope or return real choices. Ready does not mean evidence exists."""
    by_id = {doc['document_id']: doc for doc in documents}
    selected = list(dict.fromkeys(selected_document_ids or []))
    if selected and search_all:
        raise ValueError('Choose document IDs or search_all, not both.')
    if any(document_id not in by_id for document_id in selected):
        raise ValueError('A selected document is no longer available. Refresh the document list.')

    def result(status, reason, ids=None, prompt=None, options=None):
        return {
            'status': status, 'reason': reason, 'question': question,
            'document_ids': ids or [], 'clarification_question': prompt,
            'options': options or [],
        }

    if selected:
        return result('ready', 'explicit_selection', selected)
    qwords = words(question)
    normalized = ' '.join(qwords)
    if normalized in {'hi', 'hello', 'hey', 'hi there', 'hello there', 'good morning',
                      'good afternoon', 'good evening', 'sup'}:
        return result('greeting', 'standalone_greeting')
    if not documents:
        return result('no_documents', 'no_selectable_documents')
    if search_all or re.search(r'\b(?:all|across all) (?:the )?(?:documents|reports|files|pdfs)\b', normalized):
        return result('ready', 'explicit_all_documents', list(by_id))

    candidates = []
    query_tokens = set(qwords)
    for doc in documents:
        title_words = words(PurePath(doc['filename']).stem)
        title = ' '.join(title_words)
        distinctive = {w for w in title_words if len(w) >= 3 and not w.isdigit()
                       and w not in GENERIC_WORDS and not re.fullmatch(r'v\d+\w*', w)}
        full_title = bool(title and f' {title} ' in f' {normalized} ')
        if full_title or distinctive.intersection(query_tokens):
            candidates.append(doc)

    comparison = bool(re.search(r'\b(compare|comparison|versus|vs|difference|differences)\b', normalized))
    # Same-named uploads are distinct versions/records, not an automatic comparison.
    unique_names = len({d['filename'].casefold() for d in candidates}) == len(candidates)
    if len(candidates) == 1 and not comparison:
        return result('ready', 'named_document', [candidates[0]['document_id']])
    if len(candidates) >= 2 and comparison and unique_names:
        return result('ready', 'named_comparison', [d['document_id'] for d in candidates])
    if len(documents) == 1 and not comparison:
        return result('ready', 'only_available_document', list(by_id))
    return result(
        'clarification', 'comparison_scope_unclear' if comparison else 'document_scope_unclear',
        prompt='Which documents should I compare?' if comparison else 'Which document do you mean?',
        options=documents if comparison else (candidates or documents),
    )
