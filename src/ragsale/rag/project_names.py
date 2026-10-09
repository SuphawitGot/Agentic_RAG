"""Resolve full names or distinctive multiword portions of project/file names."""

from pathlib import PurePath

from .clarification import GENERIC_WORDS, words


def named_projects(question, projects):
    query = words(question)
    # A mention in a comparison or exclusion is not a single-project selection.
    # Leave these questions to the existing evidence-based scope assessment.
    scope_modifiers = {'compare', 'comparison', 'versus', 'vs', 'difference',
                       'differences', 'both', 'all', 'other', 'another', 'except',
                       'excluding', 'exclude', 'not', 'rather', 'instead'}
    if scope_modifiers.intersection(query):
        return []
    padded_query = ' ' + ' '.join(query) + ' '
    matches = []
    shortened_matches = []
    generic = GENERIC_WORDS | {'project', 'scope', 'work', 'summary'}
    for project in projects:
        shortened = False
        aliases = [project['project_name'],
                   *(PurePath(doc['filename']).stem for doc in project['documents'])]
        for alias in aliases:
            tokens = words(alias)
            # A title like "Report" is too generic to establish the user's scope.
            if not tokens or not any(token not in generic for token in tokens):
                continue
            if ' ' + ' '.join(tokens) + ' ' in padded_query:
                matches.append(project)
                break
            # "test ยา" may omit the generic word "Report" from "Test Report ยา".
            # Require an explicit document marker as well as the remaining title.
            optional = {'test', 'report', 'document', 'pdf'}
            core = [token for token in tokens if token not in optional]
            if (set(tokens).intersection(optional).intersection(query) and core
                    and any(token not in generic for token in core)
                    and ' ' + ' '.join(core) + ' ' in padded_query):
                shortened = True
            # Require adjacent complete words, not a loose keyword overlap.
            # For example, "washer inspection" can identify "IAI Washer Inspection".
            for start in range(len(tokens) - 1):
                pair = tokens[start:start + 2]
                distinctive = any(len(token) >= 3 and not token.isdigit()
                                  and token not in generic for token in pair)
                if distinctive and ' ' + ' '.join(pair) + ' ' in padded_query:
                    shortened = True
        if shortened:
            shortened_matches.append(project)
    # A full name is stronger than a shared portion; duplicates remain ambiguous.
    return matches or shortened_matches
