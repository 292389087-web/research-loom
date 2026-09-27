"""Deterministic, bounded experiment matrices. No model or optimiser calls."""
from itertools import product
import hashlib
import json
import math
from .store import dump, now, IdempotencyConflict


def submit_batch(store, spec, confirmation, key=None):
    from .experiments import trainer, create_study, persist_study, get_study
    payload = hashlib.sha256(json.dumps(spec.model_dump(), sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    key = key if key is not None else 'confirmation:' + confirmation
    if not key.strip() or len(key)>200:
        raise ValueError('提交标识需为 1 至 200 字符')
    def existing(db):
        row=db.execute('SELECT * FROM batch_submissions WHERE key=?',(key,)).fetchone()
        if row and row['digest']!=payload:
            raise IdempotencyConflict('相同提交标识对应不同配置，请为新批次使用新标识')
        return row['study_id'] if row else None
    # Validate the request hash even on replay, without requiring the trainer
    # files to remain present after an already accepted submission.
    if hashlib.sha256(dump(spec.model_dump()).encode()).hexdigest()!=confirmation:
        raise ValueError('配置已改变，请重新预览并确认完整矩阵')
    with store.db() as db:
        identity=existing(db)
    if identity:
        return get_study(store,identity)
    preview(spec,trainer(store,spec.trainer_id))
    manifest=create_study(store,spec,prepare_only=True)
    with store.db() as db:
        db.execute('BEGIN IMMEDIATE')
        identity=existing(db)
        if not identity:
            identity=persist_study(store,spec,manifest,db)['id']
            db.execute('INSERT INTO batch_submissions VALUES (?,?,?,?)',(key,payload,identity,now()))
    return get_study(store,identity)


def expand_matrix(spec):
    parameters = spec['parameters']
    if spec.get('batch_mode') == 'explicit':
        matrix = spec.get('configurations', [])
        if not matrix:
            raise ValueError('手工组合至少需要一行完整配置')
        for row in matrix:
            if set(row) != set(parameters) or any(v not in parameters[k] for k, v in row.items()):
                raise ValueError('每行必须覆盖全部参数，且取值属于声明候选集')
    else:
        count = math.prod(len(v) for v in parameters.values())
        if count > 100:
            raise ValueError('最多 100 组配置；请缩小候选集或使用手工组合')
        keys = sorted(parameters)
        matrix = [dict(zip(keys, values)) for values in product(*(parameters[k] for k in keys))]
    if not 1 <= len(matrix) <= 100:
        raise ValueError('批次需要 1 至 100 组配置')
    signatures = {dump(dict(sorted(row.items()))) for row in matrix}
    if len(signatures) != len(matrix):
        raise ValueError('不能提交重复配置；重复验证请使用多个随机种子或新批次')
    return matrix


def preview(spec, adapter):
    data = spec.model_dump()
    if not data.get('batch_mode'):
        raise ValueError('请明确选择网格或手工组合批次')
    if set(data['parameters']) != set(adapter['parameters']):
        raise ValueError('批次必须完整声明适配器参数')
    for k, values in data['parameters'].items():
        if not set(values) <= set(adapter['parameters'][k]):
            raise ValueError(f'{k} 超出已登记候选边界，请先更新可信适配器声明')
    matrix = expand_matrix(data)
    varying = [k for k in data['parameters'] if len({r[k] for r in matrix}) > 1]
    total = len(matrix) * len(data['seeds'])
    # Bind confirmation to the entire request, not just its parameter lists.
    digest = hashlib.sha256(dump(data).encode()).hexdigest()
    return {'matrix': matrix, 'matrix_sha256': hashlib.sha256(dump(matrix).encode()).hexdigest(),
            'confirmation_sha256': digest, 'configurations': len(matrix), 'executions': total,
            'varying_parameters': varying, 'fixed_parameters': {k:matrix[0][k] for k in data['parameters'] if k not in varying},
            'timeout_envelope_seconds': total*data['trial_timeout'],
            'notice': '超时上界不是耗时预测；总期限包含排队和暂停。每个配置/seed 从头训练，不继承上一组权重。执行不调用模型。'}


def pause_batch(store, sid):
    with store.db() as db:
        db.execute('BEGIN IMMEDIATE')
        row = db.execute('SELECT status,spec FROM studies WHERE id=?', (sid,)).fetchone()
        if not row:
            raise KeyError('批次不存在')
        import json
        if not json.loads(row['spec']).get('batch_mode') or row['status'] not in ('running','pausing','paused'):
            raise IdempotencyConflict('只有运行中的批次可以请求暂停')
        if row['status'] == 'running':
            db.execute("UPDATE studies SET status='pausing' WHERE id=?", (sid,))
            db.execute('INSERT INTO study_events(study_id,time,type,payload) VALUES (?,?,?,?)',
                       (sid,now(),'pause_requested',dump({'message':'当前 seed 完成后暂停，不启动下一个 seed'})))


def resume_batch(store, sid):
    import json
    import time
    with store.db() as db:
        db.execute('BEGIN IMMEDIATE')
        row = db.execute('SELECT * FROM studies WHERE id=?',(sid,)).fetchone()
        if not row:
            raise KeyError('批次不存在')
        if not json.loads(row['spec']).get('batch_mode') or row['status'] != 'paused' or row['stop']:
            raise IdempotencyConflict('仅可恢复未取消的已暂停批次')
        if row['deadline'] and time.time() >= row['deadline']:
            raise ValueError('原始总期限已到；请导出结果并为剩余配置建立新批次')
        if db.execute("SELECT COUNT(*) FROM studies WHERE status IN ('running','pausing')").fetchone()[0] >= 8:
            raise OverflowError('实验队列已满')
        db.execute("UPDATE studies SET status='running' WHERE id=?",(sid,))
        db.execute('INSERT INTO study_events(study_id,time,type,payload) VALUES (?,?,?,?)',
                   (sid,now(),'resumed',dump({'message':'保留原期限，继续尚未执行的配置/seed；不重跑失败项'})))
