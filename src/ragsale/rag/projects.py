"""Local project assignments, separate from immutable indexed chunk metadata."""

import sqlite3
import unicodedata
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from .vector_store import list_documents

PROJECT_DB = Path(__file__).resolve().parents[3] / 'data' / 'projects.sqlite3'


def list_projects():
    assignments = {}
    if PROJECT_DB.exists():
        with sqlite3.connect(PROJECT_DB) as db:
            db.execute('CREATE TABLE IF NOT EXISTS assignments (document_id TEXT PRIMARY KEY, project_id TEXT, project_name TEXT)')
            assignments = {row[0]: row[1:] for row in db.execute('SELECT * FROM assignments')}
    projects = {}
    for doc in list_documents():
        project_id, name = assignments.get(doc['document_id'],
            ('document:' + doc['document_id'], Path(doc['filename']).stem))
        project = projects.setdefault(project_id, {
            'project_id': project_id, 'project_name': name, 'documents': [],
        })
        project['documents'].append(doc)
    return sorted(projects.values(), key=lambda p: (p['project_name'].casefold(), p['project_id']))


def assign_project(document_id, project_name):
    if document_id not in {d['document_id'] for d in list_documents()}:
        raise ValueError('Document is no longer available. Refresh the document list.')
    name = ' '.join(unicodedata.normalize('NFKC', project_name).split())
    if not name or len(name) > 120:
        raise ValueError('Project name must contain 1–120 characters.')
    project_id = 'project:' + uuid5(NAMESPACE_URL, 'ragsale:' + name.casefold()).hex
    PROJECT_DB.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(PROJECT_DB) as db:
        db.execute('CREATE TABLE IF NOT EXISTS assignments (document_id TEXT PRIMARY KEY, project_id TEXT, project_name TEXT)')
        existing = db.execute('SELECT project_name FROM assignments WHERE project_id=? LIMIT 1', (project_id,)).fetchone()
        name = existing[0] if existing else name
        db.execute('INSERT OR REPLACE INTO assignments VALUES (?, ?, ?)', (document_id, project_id, name))
    return {'document_id': document_id, 'project_id': project_id, 'project_name': name}
