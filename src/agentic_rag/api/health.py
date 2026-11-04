"""Lightweight health route; does not download data or load embedding models."""

from fastapi import APIRouter

router = APIRouter()


@router.get('/health')
def health():
    return {"status": "ok"}
