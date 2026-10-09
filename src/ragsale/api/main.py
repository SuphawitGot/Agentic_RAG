"""Run with: uv run uvicorn ragsale.api.main:app --reload."""

from fastapi import FastAPI
from .health import router
from .search import router as search_router
from .documents import router as documents_router

app = FastAPI(title="RagSale API", version="0.1.0")
app.include_router(router, prefix="/api")

app.include_router(documents_router, prefix="/api")

app.include_router(search_router, prefix="/api")

from .chat import router as chat_router
app.include_router(chat_router, prefix="/api")

from .scope import router as scope_router
app.include_router(scope_router, prefix="/api")

from .projects import router as projects_router
app.include_router(projects_router, prefix="/api")
