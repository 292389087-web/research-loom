"""Independent, immutable research data snapshots and deterministic analysis."""
import csv
import hashlib
import io
import json
import math
import statistics
from decimal import Decimal
from typing import Literal
from xml.sax.saxutils import escape
from pydantic import Field, model_validator
from .schemas import Strict
from .store import uid, now, dump
from .experiments import get_study
from .experiment_design import check_rules, groups
from .audit import numeric


class CSVImport(Strict):
    name: str = Field(min_length=1, max_length=100)
    csv_text: str = Field(min_length=1, max_length=55000)
    source_note: str = Field(min_length=5, max_length=1000)
    allow_remote: bool = False


class StudySelection(Strict):
    name: str = Field(min_length=1, max_length=100)
    study_ids: list[str] = Field(min_length=1, max_length=30)


class ParameterFilter(Strict):
    parameter: str = Field(pattern=r'^[A-Za-z][A-Za-z0-9_]{0,60}$')
    values: list[float] = Field(min_length=1,max_length=30)

    @model_validator(mode='after')
    def finite(self):
        if any(not math.isfinite(v) for v in self.values):raise ValueError('筛选值必须有限')
        return self


class DataPlan(Strict):
    operation: Literal['compare', 'summary', 'failures', 'rank', 'trajectory'] = 'compare'
    metrics: list[str] = Field(default_factory=list, max_length=10)
    baseline: str | None = None
    sort_metric: str | None = None
    direction: Literal['minimize', 'maximize'] | None = None
    clarification: str = Field(default='', max_length=500)
    filters: list[ParameterFilter] = Field(default_factory=list,max_length=12)
    day_start: int | None = Field(default=None,ge=1,le=100000)
    day_end: int | None = Field(default=None,ge=1,le=100000)

    @model_validator(mode='after')
    def valid_range(self):
        if self.day_start and self.day_end and self.day_start>self.day_end:raise ValueError('起始天不能大于结束天')
        if len({f.parameter for f in self.filters})!=len(self.filters):raise ValueError('同一参数只允许一个筛选条件')
        if self.operation!='trajectory' and (self.day_start is not None or self.day_end is not None):raise ValueError('天数范围仅用于逐日分析')
        return self


class DataQuestion(Strict):
    dataset_id: str
    question: str = Field(min_length=2, max_length=2000)
    model_policy: Literal['auto', 'local', 'api'] = 'auto'
    allow_remote: bool = False
    metrics: list[str] = Field(default_factory=list, max_length=10)
    baseline: str | None = None
    sort_metric: str | None = None
    direction: Literal['minimize', 'maximize'] | None = None
    filters: list[ParameterFilter] = Field(default_factory=list,max_length=12)
    day_start: int | None = Field(default=None,ge=1,le=100000)
    day_end: int | None = Field(default=None,ge=1,le=100000)


def digest(data):
    return hashlib.sha256(dump(data).encode()).hexdigest()


def save_dataset(store, name, payload):
    identity = uid('dataset')
    payload = {**payload, 'id':identity, 'name':name, 'snapshot':digest(payload)}
    store.execute('INSERT INTO research_datasets VALUES (?,?,?,?)', (identity, name, dump(payload), now()))
    return payload


def dataset(store, identity):
    rows = store.query('SELECT payload FROM research_datasets WHERE id=?', (identity,))
    if not rows:
        raise KeyError('数据集不存在')
    return json.loads(rows[0]['payload'])


def import_csv(store, req, demo=False):
    reader = csv.DictReader(io.StringIO(req.csv_text.lstrip('\ufeff')))
    required = ['group', 'seed', 'metric', 'value', 'unit', 'protocol', 'phase']
    if reader.fieldnames != required:
        raise ValueError('CSV 表头及顺序须为：' + ','.join(required))
    rows, seen, units = [], set(), {}
    for n, r in enumerate(reader, 2):
        if len(rows) >= 1000 or None in r or any(not isinstance(r.get(k), str) or not r[k].strip() or len(r[k])>160 for k in required):
            raise ValueError(f'第 {n} 行缺字段/字段过长；最多1000行')
        r = {k:v.strip() for k,v in r.items()}
        if r['phase'] not in ('training', 'evaluation'):
            raise ValueError('phase 只能为 training 或 evaluation')
        try:
            seed = int(r['seed'])
        except ValueError:
            raise ValueError('seed 必须为整数') from None
        value = numeric(r['value'])
        key = (r['group'], seed, r['metric'])
        if key in seen:
            raise ValueError('同一实验组/seed/指标重复，不能重复计权')
        if r['metric'] in units and units[r['metric']] != r['unit']:
            raise ValueError('同一指标单位不一致，请显式换算后重新导入')
        units[r['metric']] = r['unit']; seen.add(key)
        rows.append({**r, 'seed':seed, 'value':str(value), 'status':'completed', 'source':f'csv:line:{n}',
                     'comparable_key':digest([r['protocol'],r['phase']]), 'constraint_checks':[]})
    if not rows:
        raise ValueError('CSV 不能为空')
    return save_dataset(store, req.name, {'kind':'synthetic_demo' if demo else 'csv', 'allow_remote':req.allow_remote,
        'source_note':req.source_note, 'source_sha256':hashlib.sha256(req.csv_text.encode()).hexdigest(),
        'rows':rows, 'metrics':units, 'study_ids':[], 'excluded':[],
        'notice':'合成演示数据，不代表任何真实实验成绩。' if demo else '数据及协议由用户声明，平台未独立核实来源。'})


def demo_dataset(store):
    text = 'group,seed,metric,value,unit,protocol,phase\n'
    # Deliberately synthetic constants. No fake benchmark or fabricated provenance.
    for group, values in [('基准',[100,102,98]),('奖励调整',[94,96,95]),('超参数调整',[97,96,98])]:
        for seed, energy in zip((11,23,37),values):
            for metric, value, unit in [('grid_energy',energy,'kWh'),('solar_used',150-energy,'kWh'),('max_temperature',70+(100-energy)*.5,'C')]:
                text += f'{group},{seed},{metric},{value},{unit},synthetic-fixed-v1,evaluation\n'
    return import_csv(store, CSVImport(name='演示：多轮实验与指标取舍',csv_text=text,
        source_note='内置合成数据；用于演示分析、图表和数据核查，不对应真实训练。'), demo=True)


def from_studies(store, req):
    rows, excluded, units, series, series_units = [], [], {}, [], {}
    for sid in dict.fromkeys(req.study_ids):
        s = get_study(store, sid); spec = s['spec']; adapter = s['manifest']['adapter']
        parameter_groups = groups(adapter)
        for trial in s['trials']:
            protocol_params = {k:v for k,v in trial['params'].items() if parameter_groups.get(k) in ('protocol','unclassified')}
            base_key = digest({'protocol':spec['protocol'], 'source':s['manifest']['source_hashes'],
                'datasets':s['manifest'].get('datasets'), 'assets':s['manifest'].get('assets',{}), 'packages':s['manifest'].get('packages'),
                'parameters':protocol_params, 'phase':s['manifest'].get('evaluation','unspecified')})
            for attempt in trial['repeats']:
                if attempt['status'] != 'completed' or not attempt['metrics']:
                    excluded.append({'study_id':sid,'attempt_id':attempt['id'],'status':attempt['status'],'error':attempt['error'],'group':sid+':'+trial['id'],'parameters':trial['params'], 'execution_state':attempt.get('execution_state','legacy'),'artifact_state':attempt.get('artifact_state','legacy')})
                    continue
                for metric,value in attempt['metrics'].items():
                    if not isinstance(value,(int,float)) or not math.isfinite(value):
                        continue
                    meta = adapter.get('metric_metadata',{}).get(metric,{})
                    unit = meta.get('unit','未声明')
                    if metric in units and units[metric] != unit:
                        raise ValueError('所选实验存在不同单位的同名指标，请分别导入分析')
                    units[metric] = unit
                    key = base_key
                    if meta.get('role') == 'reward' or metric in ('mean_reward','reward_slope_per_day'):
                        key = digest([base_key, {k:v for k,v in trial['params'].items() if parameter_groups.get(k)=='reward'}])
                    rows.append({'group':sid+':'+trial['id'], 'label':spec['name'] + (f" · 配置 {trial['number']+1}" if spec.get('batch_mode') else ''), 'parameters':trial['params'], 'seed':attempt['seed'],
                        'metric':metric,'value':str(value),'unit':unit,'protocol':spec['protocol'],
                        'phase':s['manifest'].get('evaluation','unspecified'), 'comparable_key':key,
                        'status':'completed','source':attempt['id'],'study_id':sid,
                        'constraint_checks':check_rules(attempt['metrics'],spec)})
                declared=adapter.get('timeseries_artifact')
                if declared:
                    from .artifact_finalization import file_record
                    folder=store.root/'experiments'/attempt['id']
                    result=json.loads((folder/'result-manifest.json').read_text(encoding='utf8'))
                    info=file_record(folder,declared)
                    if info['bytes']>10*1024**2 or info['sha256']!=result['artifacts'].get(declared):
                        raise ValueError('逐日产物超限或哈希不一致，拒绝创建快照')
                    if attempt.get('execution_receipt'):
                        receipt=json.loads(attempt['execution_receipt'])
                        if receipt['files'].get(declared)!=info:
                            raise ValueError('逐日产物与数据库中的训练完成凭据不一致')
                    daily=json.loads((folder/declared).read_text(encoding='utf8'))
                    if not isinstance(daily,list) or len(daily)>10000:raise ValueError('逐日产物必须为有界 JSON 数组')
                    seen_days=set()
                    for point in daily:
                        day=point.get(adapter.get('timeseries_axis','day')) if isinstance(point,dict) else None
                        if type(day) is not int or not 1<=day<=100000 or day in seen_days:raise ValueError('逐日坐标必须是唯一正整数')
                        seen_days.add(day)
                        for metric,meta in adapter.get('timeseries_metadata',{}).items():
                            value=point
                            for part in metric.split('.'):value=value.get(part) if isinstance(value,dict) else None
                            if type(value) not in (int,float) or not math.isfinite(value):raise ValueError('逐日产物缺少已声明的有限指标：'+metric)
                            unit=meta.get('unit','未声明')
                            if metric in series_units and series_units[metric]!=unit:raise ValueError('逐日指标单位不一致')
                            series_units[metric]=unit
                            key=digest([base_key,'trajectory',adapter.get('timeseries_axis','day'),metric,meta])
                            if meta.get('role')=='reward':key=digest([key,{k:v for k,v in trial['params'].items() if parameter_groups.get(k)=='reward'}])
                            series.append({'group':sid+':'+trial['id'],'label':spec['name']+f" · 配置 {trial['number']+1}",
                                'parameters':trial['params'],'seed':attempt['seed'],'day':day,'metric':metric,'value':str(value),'unit':unit,
                                'phase':s['manifest'].get('evaluation','unspecified'),'comparable_key':key,'source':attempt['id'],
                                'artifact':declared,'artifact_sha256':info['sha256'],'constraint_checks':[]})
                            if len(series)>20000:raise ValueError('逐日数据过大，请缩小所选实验范围')
    if len(rows)>20000:
        raise ValueError('分析数据过大，请缩小所选范围')
    return save_dataset(store,req.name,{'kind':'studies','allow_remote':False,'study_ids':list(dict.fromkeys(req.study_ids)),
        'rows':rows,'metrics':units,'series':series,'series_metrics':series_units,'excluded':excluded,'source_note':'所选平台实验的不可变指标快照',
        'notice':'记录 phase 保留适配器原声明；validation_only 不代表已独立运行测试集，须核对评价协议。'})


def save_report(store, ds, result, identity=None):
    rid = identity or uid('report')
    result = {**result,'id':rid,'dataset_id':ds['id'],'dataset_snapshot':ds['snapshot'],
              'dataset_kind':ds['kind'],'notice':ds['notice'],'created':now()}
    store.execute('INSERT OR IGNORE INTO research_reports VALUES (?,?,?,?)',(rid,ds['id'],dump(result),now()))
    return report(store,rid)


def report(store, identity):
    rows=store.query('SELECT result FROM research_reports WHERE id=?',(identity,))
    if not rows: raise KeyError('分析报告不存在')
    return json.loads(rows[0]['result'])


def analyze_data(store, ds, plan, identity=None):
    if identity and store.query('SELECT id FROM research_reports WHERE id=?',(identity,)):
        return report(store,identity)
    if plan.clarification:
        return save_report(store,ds,{'mode':'research_data','status':'needs_clarification','conclusion':plan.clarification,'plan':plan.model_dump(),'rows':[]},identity)
    if plan.filters:
        params={r['group']:r.get('parameters',{}) for r in [*ds['rows'],*ds.get('series',[]),*ds['excluded']] if 'group' in r}
        known={k for p in params.values() for k in p}
        if any(f.parameter not in known for f in plan.filters):raise ValueError('筛选引用不存在或未记录的参数')
        selected={g for g,p in params.items() if all(p.get(f.parameter) in f.values for f in plan.filters)}
        if not selected:raise ValueError('没有实验组满足参数筛选条件')
        ds={**ds,'rows':[r for r in ds['rows'] if r['group'] in selected],
            'series':[r for r in ds.get('series',[]) if r['group'] in selected],
            'excluded':[r for r in ds['excluded'] if r.get('group') in selected]}
    if plan.operation=='trajectory':return analyze_trajectory(store,ds,plan,identity)
    metrics=plan.metrics or list(ds['metrics'])[:3]
    if not set(metrics)<=set(ds['metrics']): raise ValueError('分析计划引用不存在的指标')
    all_groups={r['group'] for r in ds['rows']}
    if plan.baseline and plan.baseline not in all_groups: raise ValueError('基准实验组不存在')
    if plan.operation=='rank' and (plan.sort_metric not in metrics or not plan.direction):
        raise ValueError('排名须显式指定排序指标和方向')
    output=[]
    for group in sorted(all_groups):
        for metric in metrics:
            samples=[r for r in ds['rows'] if r['group']==group and r['metric']==metric]
            if not samples: continue
            keys={r['comparable_key'] for r in samples}
            if len(keys)>1: raise ValueError('同一实验组内评价协议不一致，不能合并 seed')
            values=[float(r['value']) for r in samples]
            mean=statistics.mean(values)
            checks=[c for r in samples for c in r['constraint_checks']]
            output.append({'group':group,'label':samples[0].get('label',group),'metric':metric,'unit':samples[0]['unit'],
                'parameters':samples[0].get('parameters',{}),
                'mean':mean,'std':statistics.stdev(values) if len(values)>1 else None,'n':len(values),
                'seeds':sorted(r['seed'] for r in samples),'comparable_key':next(iter(keys)),
                'phase':samples[0]['phase'],'constraints':'未设置' if not checks else ('满足' if all(c['passed'] for c in checks) else '违反'),
                'sources':[r['source'] for r in samples],'delta':None,'comparison':'未选择基准'})
    for row in output:
        base=next((x for x in output if x['group']==plan.baseline and x['metric']==row['metric']),None)
        if base:
            same=(row['comparable_key']==base['comparable_key'] and row['seeds']==base['seeds'] and row['unit']==base['unit'] and row['unit']!='未声明')
            row['comparison']='可作描述性对照；非因果/显著性结论' if same else '不可直接比较：协议、奖励定义、单位或 seed 集合不符/未声明'
            if same: row['delta']=row['mean']-base['mean']
    if plan.operation=='rank':
        ranked=[r for r in output if r['metric']==plan.sort_metric]
        if any(r['unit']=='未声明' for r in ranked) or len({(r['comparable_key'],tuple(r['seeds'])) for r in ranked})>1:
            raise ValueError('所选实验协议、单位或 seed 不一致/未声明，拒绝跨口径排名；请查看多指标对照')
        order={r['group']:i for i,r in enumerate(sorted(ranked,key=lambda r:r['mean'],reverse=plan.direction=='maximize'))}
        output.sort(key=lambda r:(order.get(r['group'],len(order)),r['metric']))
    conclusion=f'汇总 {len(all_groups)} 个实验组、{len(metrics)} 个指标；均值按 seed 等权计算。'
    if plan.operation=='failures': conclusion=f'存在 {len(ds["excluded"])} 条未完成或失败记录，未作为零值计入均值。'
    if any(r['comparison'].startswith('不可') for r in output): conclusion+=' 部分轮次不可直接比较，已禁止计算对应差值。'
    if any(r['constraints']=='违反' for r in output): conclusion+=' 存在违反约束的轮次。'
    differences=[r for r in output if r['group']!=plan.baseline and r['delta'] is not None]
    highlights=[f"{r['label']}：{r['metric']} 比基准{'增加' if r['delta']>0 else '减少' if r['delta']<0 else '变化'} {abs(r['delta']):.6g} {r['unit']}（{r['n']} seeds）。" for r in differences]
    conclusion+=' 多指标可能存在取舍；本报告不自动认定最优或统计显著。'
    return save_report(store,ds,{'mode':'research_data','status':'completed','conclusion':conclusion,
        'highlights':highlights,
        'plan':plan.model_dump(),'rows':output,'excluded':ds['excluded'],'formula':'mean=sum(values)/n; std=sample standard deviation (n-1); delta=current-baseline',
        'lineage':sorted({r['source'] for r in ds['rows']})},identity)


def analyze_trajectory(store, ds, plan, identity=None):
    source=ds.get('series',[])
    if not source:raise ValueError('此快照没有已声明、校验过的逐日数据；请从新适配器运行结果创建快照')
    metrics=plan.metrics or list(ds['series_metrics'])[:3]
    if not set(metrics)<=set(ds['series_metrics']):raise ValueError('逐日分析引用不存在的指标；累计指标不能代替逐日数值')
    selected=[r for r in source if r['metric'] in metrics and (plan.day_start is None or r['day']>=plan.day_start) and (plan.day_end is None or r['day']<=plan.day_end)]
    if not selected:raise ValueError('指定范围没有逐日数据')
    groups=sorted({r['group'] for r in [*source,*ds['rows']]})
    if plan.baseline and plan.baseline not in groups:raise ValueError('基准不在筛选后的逐日数据中')
    output=[]
    for metric in metrics:
        cohort=[r for r in selected if r['metric']==metric]
        if len({r['comparable_key'] for r in cohort})>1 or any(r['unit']=='未声明' for r in cohort):
            raise ValueError('逐日曲线口径或奖励定义不一致/未声明，请筛选可比组')
        expected_seed_set=None;expected_days=None
        for group in groups:
            samples=[r for r in cohort if r['group']==group]
            days=sorted({r['day'] for r in samples})
            if not days:raise ValueError('部分组缺少逐日指标')
            start=plan.day_start if plan.day_start is not None else days[0]
            end=plan.day_end if plan.day_end is not None else days[-1]
            if days!=list(range(start,end+1)):raise ValueError('逐日范围存在缺口，不补零、不插值')
            if expected_days is not None and days!=expected_days:raise ValueError('各组逐日范围不一致，请选择共同范围')
            expected_days=days
            for day in days:
                points=[r for r in samples if r['day']==day];seeds=sorted(r['seed'] for r in points)
                if set(seeds)!={r['seed'] for r in [*source,*ds['rows']] if r['group']==group}:raise ValueError('逐日 seed 不完整，拒绝静默丢弃已完成的实验')
                if len(set(seeds))!=len(seeds):raise ValueError('同一组/seed/天/指标重复')
                if expected_seed_set is not None and seeds!=expected_seed_set:raise ValueError('逐日 seed 集合不一致，请核查缺失记录')
                expected_seed_set=seeds
                values=[float(r['value']) for r in points]
                output.append({'group':group,'label':points[0].get('label',group),'parameters':points[0].get('parameters',{}),
                    'day':day,'metric':metric,'mean':statistics.mean(values),'std':statistics.stdev(values) if len(values)>1 else None,
                    'n':len(values),'seeds':seeds,'unit':points[0]['unit'],'phase':points[0]['phase'],
                    'sources':[r['source'] for r in points],'comparable_key':points[0]['comparable_key'],'constraints':'未对逐日点设置',
                    'delta':None,'comparison':'同日、同协议、同 seed 的描述性曲线；非显著性结论'})
    if plan.baseline:
        for r in output:
            base=next(x for x in output if x['group']==plan.baseline and x['metric']==r['metric'] and x['day']==r['day'])
            r['delta']=r['mean']-base['mean']
    return save_report(store,ds,{'mode':'research_data','status':'completed','plan':plan.model_dump(),'rows':output,
        'conclusion':f'逐日分析 {len(groups)} 个实验组；按同一天的 seed 等权计算均值，不把累计量当逐日量。',
        'excluded':ds['excluded'],'formula':'per-day mean; sample std (n-1); delta=same-day current-baseline',
        'lineage':sorted({r['source'] for r in selected}),'source_rows':selected},identity)


class Operand(Strict):
    group: str
    metric: str


class DataAudit(Strict):
    dataset_id: str
    operation: Literal['sum','reduction_percent']
    operands: list[Operand] = Field(min_length=1,max_length=10)
    reported: str = Field(max_length=80)
    tolerance: str = Field(default='0.01',max_length=80)
    statement: str = Field(min_length=1,max_length=2000)
    draft_id: str | None = None


def audit_data(store, req):
    ds=dataset(store,req.dataset_id); values=[]; records=[]; units=set(); keys=set(); seeds=set()
    if len({(o.group,o.metric) for o in req.operands}) != len(req.operands): raise ValueError('不能重复选择同一数据作为子项')
    for o in req.operands:
        rows=[r for r in ds['rows'] if r['group']==o.group and r['metric']==o.metric]
        if not rows: raise ValueError('核查引用了不存在的数据')
        units.update(r['unit'] for r in rows); keys.update(r['comparable_key'] for r in rows)
        seeds.add(tuple(sorted(r['seed'] for r in rows)))
        values.append(sum((numeric(r['value']) for r in rows),Decimal(0))/len(rows)); records.extend(rows)
    if len(units)!=1 or '未声明' in units or len(keys)!=1 or len(seeds)!=1:
        raise ValueError('单位、评价协议或 seed 不一致/未声明，不能进行数值核查')
    if req.operation=='reduction_percent':
        if len({o.metric for o in req.operands}) != 1:
            raise ValueError('降幅核查必须比较同一个指标')
        if len(values)!=2 or values[0]<=0: raise ValueError('降幅核查需要正基准值和一个对照值')
        expected=(values[0]-values[1])/values[0]*100; unit='%'; formula='(baseline-current)/baseline*100'
    else:
        expected=sum(values,Decimal(0)); unit=next(iter(units)); formula='sum(seed_mean(component))'
    reported=numeric(req.reported); tolerance=numeric(req.tolerance)
    if tolerance<0: raise ValueError('容差不能为负')
    binding=None
    if req.draft_id:
        from .workspace import draft_record
        draft=draft_record(store,req.draft_id)
        if req.statement not in draft['text']: raise ValueError('核查语句不在所选初稿版本中')
        binding={'draft_id':draft['id'],'draft_sha':draft['sha']}
    return save_report(store,ds,{'mode':'data_audit','operation':req.operation,'statement':req.statement,
        'expected':str(expected),'reported':str(reported),'difference':str(reported-expected),'unit':unit,
        'passed':abs(reported-expected)<=tolerance,'tolerance':str(tolerance),'formula':formula,'draft':binding,
        'inputs':req.model_dump(),'source_rows':records,'notice_scope':'仅证明已绑定数值与公式一致，不证明原始数据真实。'})


def export_csv(result):
    output=io.StringIO(); writer=csv.writer(output)
    keys=['label','day','metric','unit','mean','std','n','constraints','delta','comparison']
    writer.writerow(['report_id','dataset_snapshot']+keys)
    for r in result.get('rows',[]):
        values=[result['id'],result['dataset_snapshot']]+[r.get(k,'') for k in keys]
        writer.writerow(["'"+v if isinstance(v,str) and v.lstrip().startswith(('=','+','-','@')) else v for v in values])
    return '\ufeff'+output.getvalue()


def chart_svg(result, metric):
    if result.get('plan',{}).get('operation')=='trajectory':return trajectory_svg(result,metric)
    rows=[r for r in result.get('rows',[]) if r['metric']==metric]
    if not rows: raise ValueError('图表指标不存在')
    if len(rows)>30: raise ValueError('图表最多30组，请缩小范围')
    vals=[r['mean'] for r in rows]; low=min(0,min(vals)); high=max(0,max(vals)); span=high-low or 1
    def x(v): return 210+(v-low)/span*450
    label=escape(metric+' ('+rows[0]['unit']+')')
    svg=[f'<svg xmlns="http://www.w3.org/2000/svg" width="800" height="{110+len(rows)*42}" viewBox="0 0 800 {110+len(rows)*42}" role="img"><rect width="100%" height="100%" fill="white"/><g font-family="sans-serif" font-size="13"><text x="20" y="28">{label} — seed 均值；原始量纲</text>']
    for i,r in enumerate(rows):
        y=48+i*42; a=min(x(0),x(r['mean'])); width=abs(x(r['mean'])-x(0))
        svg.append(f'<text x="15" y="{y+18}">{escape(r["label"][:22])}</text><rect x="{a:.2f}" y="{y}" width="{max(width,1):.2f}" height="24" fill="#367c62"/><text x="675" y="{y+18}">{r["mean"]:.4g} (n={r["n"]})</text>')
    svg.append(f'<text x="20" y="{90+len(rows)*42}">{escape(result["id"])} · {"合成演示" if result["dataset_kind"]=="synthetic_demo" else "请核对协议；非显著性结论"}</text></g></svg>')
    return ''.join(svg)


def trajectory_svg(result, metric):
    rows=[r for r in result.get('rows',[]) if r['metric']==metric]
    if not rows:raise ValueError('图表指标不存在')
    groups=sorted({r['group'] for r in rows})
    if len(groups)>20 or len(rows)>4000:raise ValueError('逐日图表过大，请缩小范围')
    days=[r['day'] for r in rows];vals=[r['mean'] for r in rows]
    xmin,xmax=min(days),max(days);lo,hi=min(vals),max(vals)
    def x(v):return 75+(v-xmin)/max(xmax-xmin,1)*650
    def y(v):return 290-(v-lo)/(hi-lo or 1)*220
    colors=['#235e4a','#bd4c32','#4169a1','#9467bd','#9b7520']
    svg=[f'<svg xmlns="http://www.w3.org/2000/svg" width="820" height="{360+len(groups)*24}" role="img"><rect width="100%" height="100%" fill="white"/><g font-family="sans-serif" font-size="13"><text x="20" y="28">{escape(metric)} ({escape(rows[0]["unit"])}) · 逐日 seed 均值</text><path d="M75 55 V300 H750" stroke="black" fill="none"/>']
    for day in sorted(set(days)):
        if len(set(days))<=30 or day in (xmin,xmax):svg.append(f'<text x="{x(day):.2f}" y="320">{day}</text>')
    for i,g in enumerate(groups):
        points=sorted((r for r in rows if r['group']==g),key=lambda r:r['day']);color=colors[i%len(colors)]
        coords=' '.join(f'{x(r["day"]):.2f},{y(r["mean"]):.2f}' for r in points)
        svg.append(f'<polyline points="{coords}" stroke="{color}" stroke-width="2" fill="none"/>')
        for r in points:svg.append(f'<circle cx="{x(r["day"]):.2f}" cy="{y(r["mean"]):.2f}" r="3" fill="{color}"><title>{escape(r["label"])} day={r["day"]} mean={r["mean"]:.6g} n={r["n"]}</title></circle>')
        svg.append(f'<text x="25" y="{355+i*24}" fill="{color}">{escape(points[0]["label"][:85])}</text>')
    svg.append(f'<text x="6" y="70">{hi:.4g}</text><text x="6" y="294">{lo:.4g}</text><text x="760" y="320">天</text></g></svg>')
    return ''.join(svg)
