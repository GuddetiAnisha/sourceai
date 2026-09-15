"""Fixed-reference normalization keeps supplier scores stable across filters."""
import numpy as np
import pandas as pd

DEFAULT_WEIGHTS = {'cost': 25, 'quality': 20, 'delivery': 20, 'risk': 15, 'sustainability': 10, 'ai_readiness': 10}

def normalized_weights(weights):
    if set(weights) != set(DEFAULT_WEIGHTS):
        raise ValueError('Provide exactly the six scoring criteria.')
    values = np.array(list(weights.values()), dtype=float)
    if not np.isfinite(values).all() or (values < 0).any() or values.sum() <= 0:
        raise ValueError('Weights must be finite, nonnegative, and total more than zero.')
    return {k: float(v / values.sum()) for k, v in weights.items()}

def rank_suppliers(suppliers, weights, budget=100000):
    if not np.isfinite(budget) or budget <= 0:
        raise ValueError('Reference annual budget must be positive.')
    weights = normalized_weights(weights)
    out = suppliers.copy()
    out['cost_score'] = (100 * budget / out.annual_cost).clip(upper=100)
    for key in weights:
        score = out.cost_score if key == 'cost' else (100-out.risk if key == 'risk' else out[key])
        out[f'{key}_contribution'] = score * weights[key]
    out['score'] = out[[f'{k}_contribution' for k in weights]].sum(axis=1)
    out = out.sort_values(['score', 'supplier_id'], ascending=[False, True]).reset_index(drop=True)
    out['rank'] = range(1, len(out)+1)
    return out

def evaluate_bids(bids, ranked, years=3, minimum_sla=95, max_transition=90):
    if years <= 0:
        raise ValueError('Contract years must be positive.')
    unknown = set(bids.supplier_id) - set(ranked.supplier_id)
    if unknown:
        raise ValueError('Bids reference unknown suppliers: ' + ', '.join(sorted(unknown)))
    out = bids.merge(ranked[['supplier_id', 'name', 'score']], on='supplier_id', validate='many_to_one')
    out['tco'] = out.annual_fee * years + out.setup_fee
    out['eligible'] = (out.sla >= minimum_sla) & (out.transition_days <= max_transition)
    out['eligibility_reason'] = out.apply(lambda r: '; '.join(filter(None, ['SLA below minimum' if r.sla < minimum_sla else '', 'Transition exceeds limit' if r.transition_days > max_transition else ''])) or 'Meets constraints', axis=1)
    # Compare economics within one service lot; costs are a common currency.
    out['price_score'] = 100 * out.groupby('lot').tco.transform('min') / out.tco
    out['bid_score'] = 0.45*out.price_score + 0.40*out.score + 0.15*out.sla
    out['eligible_rank'] = out.loc[out.eligible].groupby('lot').bid_score.rank(method='min', ascending=False)
    return out.sort_values(['lot', 'eligible', 'bid_score'], ascending=[True, False, False])
