"""Expose vector retrieval to the website; no LLM generation yet."""

import logging
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator
from starlette.concurrency import run_in_threadpool

from ..rag.retrieval import retrieve

router = APIRouter(tags=["search"])
logger = logging.getLogger(__name__)


class SearchRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=3, ge=1, le=10, strict=True)

    @field_validator("question")
    @classmethod
    def check_question(cls, value):
        value = value.strip()
        if not value:
            raise ValueError("Question cannot be empty.")
        return value


class SearchMatch(BaseModel):
    id: str
    text: str
    metadata: dict[str, Any]
    distance: float


class SearchResponse(BaseModel):
    question: str
    matches: list[SearchMatch]


@router.post("/search", response_model=SearchResponse)
async def search_documents(request: SearchRequest):
    try:
        matches = await run_in_threadpool(retrieve, request.question, request.top_k)
    except Exception as error:
        logger.exception("Document search failed")
        raise HTTPException(503, "Search is unavailable. Please try again.") from error
    return SearchResponse(question=request.question, matches=matches)
