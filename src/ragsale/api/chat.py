"""Question -> vector retrieval -> local Qwen -> cited answer."""

import logging
from fastapi import APIRouter, HTTPException
from typing import Literal
from pydantic import BaseModel, Field, model_validator
from starlette.concurrency import run_in_threadpool
from .search import SearchRequest, SearchMatch
from ..rag.answering import GenerationError
from ..rag.project_chat import project_chat
from .projects import ProjectSummary

router = APIRouter(tags=['chat'])
logger = logging.getLogger(__name__)


class CitedSource(SearchMatch):
    citation: int


class ChatResponse(BaseModel):
    type: Literal['answer', 'clarification'] = 'answer'
    answer: str
    sources: list[CitedSource]
    insufficient_evidence: bool
    question: str | None = None
    project_options: list[ProjectSummary] = Field(default_factory=list)
    reason: str | None = None


class ChatRequest(SearchRequest):
    selected_project_ids: list[str] = Field(default_factory=list, max_length=12)
    search_all: bool = False

    @model_validator(mode='after')
    def check_scope(self):
        if self.selected_project_ids and self.search_all:
            raise ValueError('Choose project IDs or search_all, not both.')
        return self


@router.post('/chat', response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        return await run_in_threadpool(project_chat, request.question, request.top_k,
                                       request.selected_project_ids, request.search_all)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except GenerationError as exc:
        raise HTTPException(503, str(exc)) from exc
    except Exception as exc:
        logger.exception('Chat failed')
        raise HTTPException(503, 'Could not answer this question. Please try again.') from exc
