"""Training completion receipts and auditable, CPU-only artifact revalidation."""
import hashlib
import json
import math
import os
from pathlib import Path
import time
from .store import dump, now, uid, IdempotencyConflict
from .training import atomic_json


def file_record(folder, name):
    path=(folder/name).resolve()
    if not path.is_relative_to(folder.resolve()) or not path.is_file():
        raise ValueError('Missing or unsafe artifact: '+name)
    if path.stat().st_size>2*1024**3:
        raise ValueError('Artifact exceeds 2 GiB receipt limit: '+name)
    with path.open('rb') as stream:
        return {'bytes':path.stat().st_size,'sha256':hashlib.file_digest(stream,'sha256').hexdigest()}


def record_execution(store, study, aid, folder):
    names=['config.json','manifest.json','metrics.json',*study['manifest']['adapter'].get('artifact_files',[])]
    if (folder/'progress.jsonl').exists():names.append('progress.jsonl')
    files={}; errors=[]
    for name in dict.fromkeys(names):
        try:files[name]=file_record(folder,name)
        except ValueError as exc:errors.append(str(exc))
    receipt={'attempt_id':aid,'study_id':study['id'],'finished_at':now(),'files':files,'errors':errors}
    atomic_json(folder/'execution-receipt.json',receipt)
    with store.db() as db:
        updated=db.execute("UPDATE trial_attempts SET execution_state='completed',artifact_state='pending',execution_receipt=? WHERE id=? AND status='running' AND pid=?",(dump(receipt),aid,os.getpid())).rowcount
        if not updated:raise IdempotencyConflict('Execution ownership lost before receipt commit')
        db.execute('INSERT INTO study_events(study_id,time,type,payload) VALUES (?,?,?,?)',(study['id'],now(),'training_completed',dump(receipt)))
    return receipt


def validate_outputs(folder, study, receipt, limit=None):
    if receipt['errors']:raise ValueError('; '.join(receipt['errors']))
    for name,expected in receipt['files'].items():
        if file_record(folder,name)!=expected:raise ValueError('Artifact changed after training: '+name)
    if receipt['files'].get('metrics.json',{}).get('bytes',100001)>100000:raise ValueError('Missing or oversized metrics.json')
    metrics=json.loads((folder/'metrics.json').read_text(encoding='utf8'))
    if not isinstance(metrics,dict) or not metrics or any(type(v) not in (int,float) or not math.isfinite(v) for v in metrics.values()):
        raise ValueError('Metrics must be finite numbers')
    spec=study['spec']; adapter=study['manifest']['adapter']
    required={*([spec['objective']] if spec.get('objective') else []),*spec['constraints'],*spec.get('evaluation_metrics',[]),*(r['metric'] for r in spec.get('metric_rules',[]))}
    if not required<=metrics.keys():raise ValueError('Required evaluation metric missing')
    limit=limit if limit is not None else adapter.get('artifact_max_bytes',10*1024**2)
    artifacts={}
    for name in adapter.get('artifact_files',[]):
        info=receipt['files'][name]
        if info['bytes']>limit:raise ValueError(f"Experiment artifact exceeds frozen size limit: {name} ({info['bytes']} > {limit} bytes)")
        artifacts[name]=info['sha256']
    if 'progress.jsonl' in receipt['files']:artifacts['progress.jsonl']=receipt['files']['progress.jsonl']['sha256']
    result={'metrics_sha256':receipt['files']['metrics.json']['sha256'],'artifacts':artifacts,'attempt_id':receipt['attempt_id'],
            'receipt_sha256':hashlib.sha256(dump(receipt).encode()).hexdigest(),'artifact_max_bytes':limit}
    return metrics,result


def revalidate(store, aid, note, limit=None):
    if not 10 <= len(note.strip()) <= 2000:
        raise ValueError("Review note must contain 10 to 2000 characters")
    if limit is not None and (type(limit) is not int or not 1 <= limit <= 2*1024**3):
        raise ValueError("Invalid review note or artifact limit")
    from .experiments import get_study, summarize
    rows=store.query('SELECT a.*,t.study_id FROM trial_attempts a JOIN trials t ON t.id=a.trial_id WHERE a.id=?',(aid,))
    if not rows:raise KeyError('试验不存在')
    a=rows[0];study=get_study(store,a['study_id'])
    if a['status']!='failed' or a['execution_state']!='completed' or not a['execution_receipt']:
        raise ValueError('仅可重新验收有训练完成凭据的失败项；旧记录和训练异常不能据此改成成功')
    if study['status'] in ('running','pausing','cancelled','budget_exhausted'):
        raise IdempotencyConflict('请先暂停批次；取消或耗尽预算的批次保留原状态')
    receipt=json.loads(a['execution_receipt']);folder=store.root/'experiments'/aid
    review={'note':note,'old_status':a['status'],'old_error':a['error'],'artifact_max_bytes':limit,'receipt':receipt}
    try:
        metrics,result=validate_outputs(folder,study,receipt,limit)
        review['passed']=True
    except (ValueError,KeyError,OSError) as exc:
        review.update(passed=False,error=str(exc))
        store.execute('INSERT INTO artifact_reviews VALUES (?,?,?,?)',(uid('review'),aid,dump(review),now()))
        raise ValueError('重新验收失败：'+str(exc)) from exc
    with store.db() as db:
        db.execute('BEGIN IMMEDIATE')
        current=db.execute('SELECT a.status,a.execution_receipt,s.status AS study_status FROM trial_attempts a JOIN trials t ON t.id=a.trial_id JOIN studies s ON s.id=t.study_id WHERE a.id=?',(aid,)).fetchone()
        if current['status']!='failed' or current['execution_receipt']!=a['execution_receipt'] or current['study_status']!=study['status']:
            raise IdempotencyConflict('状态已改变，请刷新后重试')
        atomic_json(folder/'result-manifest.json',result)
        db.execute("UPDATE trial_attempts SET status='completed',artifact_state='accepted',metrics=?,error=NULL,finished=? WHERE id=?",(dump(metrics),time.time(),aid))
        db.execute('INSERT INTO artifact_reviews VALUES (?,?,?,?)',(uid('review'),aid,dump(review),now()))
        db.execute('INSERT INTO study_events(study_id,time,type,payload) VALUES (?,?,?,?)',(study['id'],now(),'artifact_revalidated',dump({'attempt_id':aid,**review})))
        repeats=[dict(r) for r in db.execute('SELECT * FROM trial_attempts WHERE trial_id=?',(a['trial_id'],))]
        if all(r['status']=='completed' for r in repeats):
            for r in repeats:r['metrics']=json.loads(r['metrics'])
            db.execute("UPDATE trials SET status='completed',metrics=?,error=NULL WHERE id=?",(dump(summarize(study['spec'],repeats)),a['trial_id']))
        trials=list(db.execute('SELECT status FROM trials WHERE study_id=?',(study['id'],)))
        if len(trials)==study['spec']['max_trials'] and all(t['status']=='completed' for t in trials):
            db.execute("UPDATE studies SET status='completed' WHERE id=?",(study['id'],))
    return {'attempt_id':aid,'passed':True,'note':note,'training_reexecuted':False}
