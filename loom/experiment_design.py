"""Controlled changes and explicit evaluation contracts."""
import math
from typing import Literal
from pydantic import Field, model_validator
from .schemas import Strict

Scope = Literal['baseline', 'reward', 'hyperparameter', 'repeat']


class MetricRule(Strict):
    metric: str = Field(min_length=1, max_length=61)
    operator: Literal['le', 'ge'] = 'le'
    value: float

    @model_validator(mode='after')
    def finite(self):
        if not math.isfinite(self.value):
            raise ValueError('约束值必须有限')
        return self


def groups(adapter):
    return {k: adapter.get('parameter_groups', {}).get(k, 'unclassified') for k in adapter['parameters']}


def allowed_changes(adapter, scope):
    return [k for k, group in groups(adapter).items() if group == scope] if scope != 'repeat' else []


def check_changes(base, candidate, allowed):
    if set(base) != set(candidate):
        raise ValueError('下一轮必须保留基准轮的完整参数集合')
    changed = [k for k in base if base[k] != candidate[k]]
    forbidden = set(changed) - set(allowed)
    if forbidden:
        raise ValueError('本轮不允许改变此参数组：' + ', '.join(sorted(forbidden)))
    return changed


def check_derived(parent, spec, adapter):
    if parent['status'] not in ('completed', 'failed', 'cancelled', 'budget_exhausted') or not parent['trials']:
        raise ValueError('基准轮必须已结束并有参数记录')
    old = parent['spec']
    if groups(adapter) != groups(parent['manifest']['adapter']):
        raise ValueError('参数分类已更新，请建立带分类的新基线')
    if spec.trainer_id != old['trainer_id']:
        raise ValueError('派生实验不能更换训练适配器')
    if spec.project_name != old.get('project_name','我的研究'):
        raise ValueError('派生实验不能更换研究项目')
    for name in ('protocol', 'seeds', 'objective', 'direction', 'constraints', 'evaluation_metrics', 'metric_rules'):
        default = [] if name in ('evaluation_metrics', 'metric_rules') else None
        if spec.model_dump()[name] != old.get(name, default):
            raise ValueError(f'派生实验必须锁定评价协议：{name}；改变协议请建立新基线')
    return check_changes(parent['trials'][-1]['params'], {k:v[0] for k,v in spec.parameters.items()}, allowed_changes(adapter, spec.adjustment_scope))


def check_rules(metrics, spec):
    rules = [{'metric':k, 'operator':'le', 'value':v} for k,v in spec.get('constraints', {}).items()] + spec.get('metric_rules', [])
    results = []
    for rule in rules:
        value = metrics.get(rule['metric'])
        ok = value is not None and math.isfinite(value) and (value <= rule['value'] if rule['operator']=='le' else value >= rule['value'])
        results.append({**rule, 'actual':value, 'passed':ok})
    return results
