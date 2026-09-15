from io import StringIO
import numpy as np
import pandas as pd
import pytest
from sourceai.data import sample, read_csv, validate, FACTORS
from sourceai.scoring import rank_suppliers, evaluate_bids, normalized_weights, DEFAULT_WEIGHTS
from sourceai.risk import classify, train_model
from sourceai.planning import assess_maturity, prioritize, roadmap, DIMENSIONS
from sourceai.reports import build_report, csv_bytes

def test_samples_and_csv_roundtrip():
    for kind in ['suppliers', 'bids', 'use_cases']:
        frame = sample(kind)
        pd.testing.assert_frame_equal(frame, read_csv(StringIO(frame.to_csv(index=False)), kind))

@pytest.mark.parametrize('value', [None, 'oops', -1, 101, float('inf')])
def test_invalid_quality_rejected(value):
    data = sample('suppliers').astype(object)
    data.loc[0, 'quality'] = value
    with pytest.raises(ValueError, match='quality'):
        validate(data, 'suppliers')

def test_empty_missing_duplicate_blank_rejected():
    data = sample('suppliers')
    for invalid in [data.iloc[:0], data.drop(columns='risk'), pd.concat([data, data.iloc[:1]]), data.assign(name=' ')]:
        with pytest.raises(ValueError):
            validate(invalid, 'suppliers')

@pytest.mark.parametrize('weights', [{k: 0 for k in DEFAULT_WEIGHTS}, {**DEFAULT_WEIGHTS, 'risk': -1}, {**DEFAULT_WEIGHTS, 'risk': np.nan}, {'cost': 1}])
def test_invalid_weights(weights):
    with pytest.raises(ValueError):
        normalized_weights(weights)

def test_score_formula_and_weight_scaling():
    data = sample('suppliers').iloc[:1].assign(annual_cost=200000, quality=80, delivery=90, risk=20, sustainability=60, ai_readiness=70)
    a = rank_suppliers(data, DEFAULT_WEIGHTS)
    expected = .25*50 + .2*80 + .2*90 + .15*80 + .1*60 + .1*70
    assert a.iloc[0].score == pytest.approx(expected)
    b = rank_suppliers(data, {k: 2*v for k, v in DEFAULT_WEIGHTS.items()})
    assert a.iloc[0].score == pytest.approx(b.iloc[0].score)

def test_filter_stability_ties_and_cost_direction():
    data = sample('suppliers')
    all_scores = rank_suppliers(data, DEFAULT_WEIGHTS).set_index('supplier_id').score
    filtered = rank_suppliers(data.iloc[:3], DEFAULT_WEIGHTS).set_index('supplier_id').score
    assert all_scores.loc[filtered.index].equals(filtered)
    cost_only = {k: int(k == 'cost') for k in DEFAULT_WEIGHTS}
    ranked = rank_suppliers(data, cost_only, budget=50000)
    assert ranked.iloc[0].annual_cost == data.annual_cost.min()
    tied = data.iloc[:2].copy()
    tied.loc[:, 'annual_cost'] = 90000
    assert rank_suppliers(tied, cost_only).supplier_id.tolist() == ['S001', 'S002']

def test_bid_tco_constraints_and_unknown_supplier():
    suppliers = rank_suppliers(sample('suppliers'), DEFAULT_WEIGHTS)
    bids = sample('bids')
    result = evaluate_bids(bids, suppliers, 3, 95, 90).set_index('bid_id')
    assert result.loc['B001', 'tco'] == 296000
    assert not result.loc['B004', 'eligible']
    assert pd.isna(result.loc['B004', 'eligible_rank'])
    assert result.loc['B002', 'eligible']
    assert result.loc['B007', 'price_score'] == pytest.approx(100*(65000*3+25000)/(85000*3+12000))
    with pytest.raises(ValueError, match='unknown'):
        evaluate_bids(bids.assign(supplier_id='missing'), suppliers)
    assert not evaluate_bids(bids, suppliers, minimum_sla=100).eligible.any()

def test_risk_override_reproducible_and_holdout():
    frame = sample('suppliers').iloc[:3].copy()
    frame.loc[0, ['risk', 'delivery']] = [80, 99]
    frame.loc[1, ['risk', 'delivery']] = [0, 59]
    a = classify(frame)
    assert a.iloc[:2].risk_class.tolist() == ['High', 'High']
    assert a.iloc[0].risk_reason.startswith('Override')
    pd.testing.assert_frame_equal(a, classify(frame))
    _, metrics = train_model()
    assert metrics['holdout_rows'] == 600
    assert sum(map(sum, metrics['confusion_matrix'])) == 600
    assert metrics['synthetic_holdout_accuracy'] > .75

@pytest.mark.parametrize('value,level', [(1, 1), (2.99, 2), (3, 3), (4.99, 4), (5, 5)])
def test_maturity_boundaries(value, level):
    assert assess_maturity(dict.fromkeys(DIMENSIONS, value))['level'] == level

def test_maturity_invalid():
    for scores in [{}, dict.fromkeys(DIMENSIONS, 0), dict.fromkeys(DIMENSIONS, np.nan)]:
        with pytest.raises(ValueError):
            assess_maturity(scores)

def test_priority_product_and_readiness_gate():
    cases = sample('use_cases').iloc[:1].copy()
    cases[FACTORS] = 5
    priorities = prioritize(cases)
    assert priorities.iloc[0].priority_product == 625
    assert priorities.iloc[0].priority_percent == 100
    ready = assess_maturity(dict.fromkeys(DIMENSIONS, 4))
    blocked = assess_maturity({**ready['scores'], 'Governance': 1})
    suppliers = classify(sample('suppliers'))
    assert 'Pilot Supplier evidence extraction' in roadmap(ready, priorities, suppliers).action.tolist()
    assert any('Resolve readiness gaps' in action for action in roadmap(blocked, priorities, suppliers).action)

def test_safe_export_and_report_context():
    assert "'=SUM(1)" in csv_bytes(pd.DataFrame({'name': ['=SUM(1)']})).decode('utf-8-sig')
    ranked = classify(rank_suppliers(sample('suppliers'), DEFAULT_WEIGHTS))
    bids = evaluate_bids(sample('bids'), ranked)
    maturity = assess_maturity(dict.fromkeys(DIMENSIONS, 3))
    priorities = prioritize(sample('use_cases'))
    plan = roadmap(maturity, priorities, ranked)
    report = build_report(ranked, bids, maturity, priorities, plan, normalized_weights(DEFAULT_WEIGHTS), 100000, 3, 95, 90, 'Managed services')
    for text in ['## Scenario', '## Roadmap', '## Bid evaluation', 'minimum_sla', 'not a calibrated probability', 'Managed services']:
        assert text in report
