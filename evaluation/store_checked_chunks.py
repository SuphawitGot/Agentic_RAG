"""Validate improved exports, embed, replace their existing records, and verify.

Run only when intentionally updating the three benchmark reports in the live DB.
Other documents are preserved; a full logical backup is saved before writes.
"""
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone
import numpy as np

from ragsale.rag.vector_store import get_collection, save_chunks


def record_id(text, metadata):
    return hashlib.sha256(json.dumps([text, metadata], sort_keys=True).encode()).hexdigest()


def snapshot(collection):
    data=collection.get(include=['documents','metadatas','embeddings'])
    return {i: {'text':t, 'metadata':m, 'vector':list(map(float,v))}
            for i,t,m,v in zip(data['ids'],data['documents'],data['metadatas'],data['embeddings'])}


def verify(expected, actual):
    if set(expected)!=set(actual):
        raise ValueError('Stored IDs differ from expected IDs.')
    for i,r in expected.items():
        got=actual[i]
        if r['text']!=got['text'] or r['metadata']!=got['metadata']:
            raise ValueError(f'Text or metadata mismatch: {i}')
        if not np.allclose(r['vector'],got['vector'],atol=1e-5,rtol=1e-4):
            raise ValueError(f'Vector mismatch: {i}')


def main():
    from evaluation.evaluate_chunking import evaluate_chunks, read_json
    from langchain_core.documents import Document
    from ragsale.rag.embedding import embed_chunks
    refs=Path('evaluation/references/three-documents')
    manifest=read_json(refs/'manifest.json')
    contexts=read_json(Path('evaluation/references/chunk_contexts.json'))['cases']
    c=get_collection(); before=snapshot(c); docs=[];target_ids=set()
    for spec in manifest:
        filename=spec['filename']
        matching={r['metadata']['document_id'] for r in before.values() if r['metadata'].get('filename')==filename}
        if len(matching)!=1:
            raise ValueError(f'{filename}: expected exactly one existing document ID, found {len(matching)}')
        document_id=matching.pop();target_ids.add(document_id)
        exported=read_json(Path(f'output/chunks/{spec["id"]}-improved-chunks.json'))
        chunks=exported['chunks']
        if exported['count']!=len(chunks):raise ValueError('Export count mismatch')
        report=evaluate_chunks(read_json(Path(f'output/extraction/layout-improved/{spec["id"]}.json')),chunks,
            [read_json(refs/r) for r in spec['references']], [x for x in contexts if x['filename']==filename])
        if not report['passed']:raise ValueError(f'Chunk checks failed: {filename}')
        for chunk in chunks:
            metadata={**chunk['metadata'],'document_id':document_id}
            docs.append(Document(page_content=chunk['text'],metadata=metadata))
    vectors=np.asarray(embed_chunks(docs),dtype=np.float32)
    if vectors.shape!=(len(docs),384) or not np.isfinite(vectors).all() or not np.allclose(np.linalg.norm(vectors,axis=1),1,atol=1e-5,rtol=0):
        raise ValueError('Embedding integrity check failed; nothing written.')
    fresh={record_id(d.page_content,d.metadata):{'text':d.page_content,'metadata':d.metadata,'vector':v.tolist()} for d,v in zip(docs,vectors)}
    if len(fresh)!=len(docs):raise ValueError('Duplicate generated IDs')
    old={i:r for i,r in before.items() if r['metadata'].get('document_id') in target_ids}
    untouched={i:r for i,r in before.items() if i not in old}
    expected={**untouched,**fresh}
    folder=Path('output/extraction/storage')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    folder.mkdir(parents=True)
    (folder/'backup.json').write_text(json.dumps(before,ensure_ascii=False))
    (folder/'expected.json').write_text(json.dumps(expected,ensure_ascii=False))
    # Abort if another writer changed the collection while embeddings were made.
    verify(before,snapshot(c))
    obsolete=set(old)-set(fresh)
    try:
        save_chunks(docs,vectors)
        verify(fresh,{i:r for i,r in snapshot(c).items() if i in fresh})
        if obsolete:c.delete(ids=sorted(obsolete))
        verify(expected,snapshot(c))
    except Exception:
        # Restore the affected records; leave unrelated records alone.
        records=list(old.items())
        for start in range(0,len(records),100):
            batch=records[start:start+100]
            c.upsert(ids=[i for i,r in batch],documents=[r['text'] for i,r in batch],metadatas=[r['metadata'] for i,r in batch],embeddings=[r['vector'] for i,r in batch])
        new_ids=set(fresh)-set(before)
        if new_ids:c.delete(ids=sorted(new_ids))
        raise
    result={'status':'stored_and_readback_verified','improved_chunks':len(fresh),'replaced_document_ids':sorted(target_ids),'obsolete_records_removed':len(obsolete),'other_records_preserved':len(untouched),'total_records':len(expected),'backup':str(folder/'backup.json'),'expected':str(folder/'expected.json'),'restart_verified':False}
    (folder/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
