"""Evidence-only search tools for the LangGraph agent.

Ported from RagSale's hand-written agent (`rag/agent_tools.py`). The search
logic is the same; the difference is how the model sees the tools. RagSale
put the argument schemas into one JSON `Decision`; here each schema becomes a
native tool the model can call.

Tool results are candidate evidence, never verified findings.
"""
import logging
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

logger = logging.getLogger(__name__)
Query = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)]
Identifier = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=256)]


class Arguments(BaseModel):
    model_config = ConfigDict(extra='forbid')


# --- Search tools: the model calls these to collect evidence. ---

class SearchAllProjects(Arguments):
    """Find candidate evidence across a page of projects. Follow next_offset for more
    projects. Results are not confirmed matches or an exhaustive list."""
    query: Query = Field(description='Concrete words the answer itself would contain.')
    offset: int = Field(default=0, ge=0, description='Start of the project page (from next_offset).')
    limit: int = Field(default=10, ge=1, le=20, description='Projects to search in this page.')


class SearchProject(Arguments):
    """Retrieve evidence within one known project ID. Never guess a project ID."""
    project_id: Identifier = Field(description='A project_id from the catalog or earlier results.')
    query: Query = Field(description='Concrete words the answer would contain. Do not include the project name.')


class CompareProjects(Arguments):
    """Retrieve evidence for the same question from 2-6 known projects. Keep each
    project's measurements separate; results may be missing or irrelevant."""
    project_ids: list[Identifier] = Field(min_length=2, max_length=6)
    question: Query


# --- Finishing tools: calling one of these ends the agent's turn. ---
# RagSale's Decision.action had four values: tool, answer, clarify, insufficient.
# Here "tool" is any search tool above, and the other three are these tools.

class Claim(BaseModel):
    text: str = Field(description="ONE fact in the user's language, without [n] markers.")
    quote: str = Field(description='The exact words of ONE excerpt that state the fact, copied as written.')
    citation: int = Field(description="That excerpt's source number.")


class SubmitAnswer(Arguments):
    """Answer the question as a list of claims. Each claim is one fact with an exact
    quote from one numbered excerpt. Python removes any claim whose quote is not in
    its cited excerpt."""
    claims: list[Claim] = Field(min_length=1, max_length=40)


class AskUser(Arguments):
    """Ask which project the user means, only when the answer depends on the project
    and the question does not say which one."""
    question: str = Field(min_length=1, max_length=500, description='A short, focused question for the user.')
    project_ids: list[Identifier] = Field(min_length=2, max_length=6, description='Projects to offer as choices.')


class ReportInsufficient(Arguments):
    """Report that the retrieved evidence cannot answer the question."""
    reason: str = Field(default='', max_length=500)


SEARCH_TOOLS = {'search_all_projects': SearchAllProjects,
                'search_project': SearchProject,
                'compare_projects': CompareProjects}
FINISH_TOOLS = {'submit_answer': SubmitAnswer,
                'ask_user': AskUser,
                'report_insufficient': ReportInsufficient}


def as_tool(name, schema):
    """A tool definition the model can call, built from a Pydantic schema."""
    return {'type': 'function', 'function': {
        'name': name, 'description': ' '.join(schema.__doc__.split()),
        'parameters': schema.model_json_schema()}}


class ToolError(ValueError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def _select(projects, ids):
    if len(set(ids)) != len(ids):
        raise ToolError('duplicate_project_ids', 'Choose distinct project IDs.')
    by_id = {p['project_id']: p for p in projects}
    if any(i not in by_id for i in ids):
        raise ToolError('unknown_project', 'A project ID is unknown. Use IDs from the catalog.')
    return [by_id[i] for i in ids]


def _retrieve_projects(query, projects, top_k):
    from ..rag.retrieval import retrieve_projects  # Loads Chroma and the embedding model.
    return retrieve_projects(query, projects, top_k)


def _search(query, projects, top_k):
    retrieved = _retrieve_projects(query, projects, top_k) if projects else []
    groups = []
    for project, found in zip(projects, retrieved):
        evidence, seen = [], set()
        for match in found['matches']:
            if match['id'] in seen:
                continue
            seen.add(match['id'])
            evidence.append(match)
        groups.append({'project_id': project['project_id'], 'project_name': project['project_name'],
                       'evidence': evidence})
    return groups


def run_search(name, arguments, projects, top_k=3):
    """Run one allowlisted search tool. Returns (project groups, next_offset).

    `projects` is the catalog the agent may search; the graph narrows it to the
    user's selection before calling this, so the scope is enforced in Python.
    """
    if name not in SEARCH_TOOLS:
        raise ToolError('unknown_tool', 'Tool is not available.')
    args = SEARCH_TOOLS[name].model_validate(arguments)
    if name == 'search_all_projects':
        if args.offset > len(projects):
            raise ToolError('invalid_offset', 'Offset exceeds the project catalog.')
        page = projects[args.offset:args.offset + args.limit]
        following = args.offset + len(page)
        return _search(args.query, page, top_k), (following if following < len(projects) else None)
    if name == 'search_project':
        # One project gets 5 excerpts: RagSale found 3 too few for diagram pages.
        return _search(args.query, _select(projects, [args.project_id]), 5), None
    return _search(args.question, _select(projects, args.project_ids), top_k), None
