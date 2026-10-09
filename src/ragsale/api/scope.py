"""Preview whether a question needs document-scope clarification."""

import logging
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, model_validator
from starlette.concurrency import run_in_threadpool

from .documents import DocumentSummary
from .search import SearchRequest
from ..rag.clarification import decide_scope
from ..rag.vector_store import list_documents

router = APIRouter(tags=['chat'])
logger = logging.getLogger(__name__)


class ScopeRequest(SearchRequest):
    selected_document_ids: list[str] = Field(default_factory=list, max_length=100)
    search_all: bool = False

    @model_validator(mode='after')
    def validate_scope(self):
        if self.selected_document_ids and self.search_all:
            raise ValueError('Choose document IDs or search_all, not both.')
        return self


class ScopeResponse(BaseModel):
    status: Literal['ready', 'clarification', 'greeting', 'no_documents']
    reason: str
    question: str
    document_ids: list[str]
    clarification_question: str | None
    options: list[DocumentSummary]


@router.post('/chat/scope', response_model=ScopeResponse)
async def check_scope(request: ScopeRequest):
    try:
        documents = await run_in_threadpool(list_documents)
    except Exception as error:
        logger.exception('Could not load documents for scope checking')
        raise HTTPException(503, 'Could not load indexed documents. Please try again.') from error
    try:
        return decide_scope(request.question, documents, request.selected_document_ids, request.search_all)
    except ValueError as error:
        raise HTTPException(422, str(error)) from error
