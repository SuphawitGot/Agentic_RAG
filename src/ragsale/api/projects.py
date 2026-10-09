import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from ..rag.projects import assign_project, list_projects
from .documents import DocumentSummary

router = APIRouter(tags=['projects'])
logger = logging.getLogger(__name__)


class ProjectSummary(BaseModel):
    project_id: str
    project_name: str
    documents: list[DocumentSummary]


class ProjectAssignment(BaseModel):
    project_name: str = Field(min_length=1, max_length=120)


@router.get('/projects', response_model=list[ProjectSummary])
async def get_projects():
    try:
        return await run_in_threadpool(list_projects)
    except Exception as error:
        logger.exception('Could not load projects')
        raise HTTPException(503, 'Could not load projects. Please try again.') from error


@router.put('/documents/{document_id}/project')
async def set_project(document_id: str, request: ProjectAssignment):
    try:
        return await run_in_threadpool(assign_project, document_id, request.project_name)
    except ValueError as error:
        raise HTTPException(422, str(error)) from error
    except Exception as error:
        logger.exception('Could not save project assignment')
        raise HTTPException(503, 'Could not save project assignment. Please try again.') from error
