"""Strict, reusable CSV contracts; no silent imputation."""
from pathlib import Path
import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[1] / 'data'
SUPPLIER_RANGES = {'annual_cost': (1, 1e12), 'quality': (0, 100), 'delivery': (0, 100), 'risk': (0, 100), 'sustainability': (0, 100), 'ai_readiness': (0, 100)}
BID_RANGES = {'annual_fee': (1, 1e12), 'setup_fee': (0, 1e12), 'transition_days': (1, 3650), 'sla': (0, 100)}
FACTORS = ['business_value', 'feasibility', 'data_readiness', 'adoption_potential']

def validate(frame, kind):
    contracts = {
        'suppliers': (['supplier_id', 'name', 'category'], SUPPLIER_RANGES),
        'bids': (['bid_id', 'supplier_id', 'lot'], BID_RANGES),
        'use_cases': (['use_case', 'owner'], {k: (1, 5) for k in FACTORS}),
    }
    texts, ranges = contracts[kind]
    required = texts + list(ranges)
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError('Missing columns: ' + ', '.join(sorted(missing)))
    if frame.empty:
        raise ValueError('Dataset must contain at least one row.')
    if len(frame) > 10000:
        raise ValueError('Prototype supports at most 10,000 rows.')
    out = frame[required].copy()
    for col in texts:
        if out[col].isna().any() or out[col].astype(str).str.strip().eq('').any():
            raise ValueError(f'{col}: blank values are not allowed.')
        out[col] = out[col].astype(str).str.strip()
    if out[texts[0]].duplicated().any():
        raise ValueError(f'{texts[0]} must be unique.')
    for col, (low, high) in ranges.items():
        out[col] = pd.to_numeric(out[col], errors='coerce')
        if not (np.isfinite(out[col]) & out[col].between(low, high)).all():
            raise ValueError(f'{col} must contain finite numbers between {low} and {high}.')
    return out

def read_csv(source, kind):
    try:
        frame = pd.read_csv(source, dtype=str)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeError) as exc:
        raise ValueError(f'Cannot read CSV: {exc}') from exc
    return validate(frame, kind)

def sample(kind):
    return read_csv(DATA / f'{kind}.csv', kind)
