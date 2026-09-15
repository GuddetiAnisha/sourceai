"""Deterministic assessments and recommendations, requiring no external AI API."""
import math
import pandas as pd
from .data import FACTORS, validate

DIMENSIONS = ['Strategy', 'Data', 'Technology', 'Process', 'Governance', 'People']
LEVELS = {1: 'Initial', 2: 'Emerging', 3: 'Defined', 4: 'Managed', 5: 'Optimizing'}
ACTIONS = {
    'Strategy': ('Agree sourcing outcomes and sponsor', 'Sourcing lead', 'Approved KPI baseline'),
    'Data': ('Create supplier data dictionary and quality checks', 'Data steward', '95% required-field completeness'),
    'Technology': ('Pilot a versioned analytics workflow', 'Engineering lead', 'Reproducible scoring run'),
    'Process': ('Standardize bid intake and review gates', 'Category manager', '90% bids use standard template'),
    'Governance': ('Define human approval and model review policy', 'Risk lead', 'All awards have reviewer sign-off'),
    'People': ('Train sourcing team using pilot scenarios', 'Enablement lead', '80% pilot team completes training'),
}

def assess_maturity(scores):
    if set(scores) != set(DIMENSIONS) or any(not isinstance(v, (int, float)) or not math.isfinite(v) or v < 1 or v > 5 for v in scores.values()):
        raise ValueError('All six maturity dimensions require a score from 1 to 5.')
    average = sum(scores.values()) / 6
    level = min(5, math.floor(average))
    return {'average': average, 'level': level, 'label': LEVELS[level], 'scores': dict(scores)}

def prioritize(use_cases):
    out = validate(use_cases, 'use_cases')
    out['priority_product'] = out[FACTORS].prod(axis=1)
    out['priority_percent'] = out.priority_product / 625 * 100
    out['bottleneck'] = out[FACTORS].idxmin(axis=1)
    return out.sort_values(['priority_product', 'use_case'], ascending=[False, True]).reset_index(drop=True)

def roadmap(maturity, priorities, suppliers):
    rows = []
    for dimension, score in sorted(maturity['scores'].items(), key=lambda pair: pair[1]):
        if score < 4:
            action, owner, measure = ACTIONS[dimension]
            rows.append({'phase': '0–30 days', 'action': action, 'owner': owner, 'success_measure': measure})
    high = suppliers[suppliers.risk_class == 'High']
    if len(high):
        rows.append({'phase': '0–30 days', 'action': 'Review high-risk suppliers: ' + ', '.join(high.name), 'owner': 'Risk lead', 'success_measure': 'Mitigation owner assigned to each flagged supplier'})
    for _, case in priorities.head(3).iterrows():
        ready = case[FACTORS].min() >= 3 and maturity['scores']['Data'] >= 3 and maturity['scores']['Governance'] >= 3
        rows.append({'phase': '31–60 days' if ready else '61–90 days', 'action': ('Pilot ' if ready else 'Resolve readiness gaps before piloting ') + case.use_case, 'owner': case.owner, 'success_measure': 'Measure cycle time and reviewer agreement against baseline'})
    rows.append({'phase': '91–180 days', 'action': 'Review pilot evidence and approve or stop expansion', 'owner': 'Executive sponsor', 'success_measure': 'Document measured benefit, failure cases, and approval decision'})
    return pd.DataFrame(rows)
