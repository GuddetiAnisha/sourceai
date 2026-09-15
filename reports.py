"""Portable report and spreadsheet-safe exports."""
import json
from datetime import datetime, timezone

def csv_bytes(frame):
    clean = frame.copy()
    for col in clean.select_dtypes(include=['object', 'string']).columns:
        clean[col] = clean[col].map(lambda v: "'" + v if isinstance(v, str) and v.lstrip().startswith(('=', '+', '-', '@')) else v)
    return clean.to_csv(index=False).encode('utf-8-sig')

def build_report(ranked, bids, maturity, priorities, plan, weights, budget, years, minimum_sla, max_transition, category, data_sources=None):
    return '\n\n'.join([
        '# SourceAI decision support report',
        f'Generated {datetime.now(timezone.utc).isoformat()} | Portfolio prototype; synthetic sample data unless replaced.',
        'Human review required. Risk model learns synthetic policy labels; it is not a calibrated probability of supplier failure. No external LLM is used.',
        '## Data provenance\n' + json.dumps(data_sources or {'inputs': 'Unspecified; verify provenance before reuse'}, indent=2) + '\nUse-case ratings may include edits made in the UI. Supplier rows reflect the category; bids cover all available lots.',
        '## Scenario\n' + json.dumps({'weights': weights, 'reference_budget_eur': budget, 'contract_years': years, 'minimum_sla': minimum_sla, 'max_transition_days': max_transition, 'category': category}, indent=2),
        f"## Maturity\nLevel {maturity['level']} — {maturity['label']}; mean {maturity['average']:.2f}/5.\n" + json.dumps(maturity['scores'], indent=2),
        '## Supplier ranking\n```csv\n' + csv_bytes(ranked).decode('utf-8-sig') + '```',
        '## Bid evaluation\n```csv\n' + csv_bytes(bids).decode('utf-8-sig') + '```',
        '## Prioritized use cases\n```csv\n' + csv_bytes(priorities).decode('utf-8-sig') + '```',
        '## Roadmap\n```csv\n' + csv_bytes(plan).decode('utf-8-sig') + '```',
    ])
