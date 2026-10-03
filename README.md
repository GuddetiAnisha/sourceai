# SourceAI — Intelligent Service Sourcing Decision Support Platform

A runnable, independent Computer Science / AI master's portfolio prototype. Explore supplier tradeoffs, compare bids, inspect a small ML risk model, assess organizational maturity, and turn priorities into an actionable roadmap.

## Run locally

Use Python 3.11 or newer (3.12 recommended). From this directory:

```bash
python -m venv .venv
```

Activate on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Or on macOS/Linux:

```bash
source .venv/bin/activate
```

Then:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open the local address printed by Streamlit, normally http://localhost:8501. No API keys, database, external LLM or account required. Stop with Ctrl+C. If PowerShell activation is restricted, use `.venv\Scripts\python.exe` in place of `python` without activating.

## What is implemented

1. **Data ingestion:** bundled synthetic suppliers, bids and use cases; optional CSV uploads; required columns, unique IDs, finite values, bounds, blank and foreign-key checks.
2. **Supplier ranking:** six weighted criteria, normalized weights, fixed budget reference, criterion contributions, ranking changes against default weights and category filtering.
3. **Bid evaluation:** one to five-year total cost, SLA and transition eligibility, service-lot comparisons, weighted bid scores and eligible ranks.
4. **Risk analytics:** shallow decision tree trained on synthetic policy labels, explicit critical-risk overrides, explanations, learned rules and held-out metrics.
5. **Maturity:** Strategy, Data, Technology, Process, Governance and People; anchored 1–5 self-assessment and radar chart.
6. **Use-case prioritization:** editable 1–5 ratings; Business Value × Feasibility × Data Readiness × Adoption Potential, bottleneck identification and bubble chart.
7. **Recommendations:** readiness-gated pilots, actions for maturity gaps, high-risk supplier reviews, owners, suggested success measures and four delivery phases.
8. **Exports:** supplier rankings, all bid results, maturity, priorities and roadmap as CSV; a consolidated Markdown report with scenario inputs and calculated results.

## Adaptation and provenance

Inspired by the supplied opportunity text associated with **Ericsson Master Thesis Req ID 790824**, concerning AI-enabled service sourcing. This is independent portfolio work with no affiliation, endorsement, company data or claim to implement Ericsson's internal processes. The requisition identifier comes from the project brief, not an independently authenticated job listing.

| Inspiration | Portfolio adaptation |
|---|---|
| AI sourcing maturity and opportunities | Interactive six-dimension self-assessment and editable use-case ranking |
| Practical recommendations | Deterministic, downloadable roadmap tied to assessment inputs |
| Stakeholder interviews | Omitted as an implemented feature |
| Cross-industry academic benchmarking | Omitted as an implemented feature |
| Research-oriented thesis | Runnable decision-support application with algorithms, modular Python and tests |
| Broad sourcing exploration | Supplier scoring, bid economics, ML risk triage, scenario simulation and dashboard |

The two sample service categories are demonstration datasets, not an academic cross-industry benchmark. All supplier names, costs, bids and ratings are synthetic. No real supplier performance is implied.

## Five-minute demonstration

1. Launch the app and choose **Managed services**. Inspect the weighted ranking and contribution chart.
2. Raise Quality and AI readiness weights; inspect rank changes. Change the annual budget to explore cost sensitivity. Weights need not total 100, but cannot all be zero.
3. Open Bid comparison. Increase the SLA threshold to 99, then change contract years. Observe how setup costs affect total cost and which bids remain eligible.
4. Inspect Delta Support's high-risk override and the learned tree in Risk lab.
5. Raise Data and Governance to 3 in Maturity assessment. Inspect the readiness effects on pilot recommendations.
6. Edit use-case ratings, then download the report and roadmap.
7. Download a sample CSV to use as an upload template. For custom suppliers, upload matching bids with valid supplier IDs; mismatched bids disable bid comparison while other tools remain usable.

## Algorithms and assumptions

### Supplier score

Every criterion is expressed on 0–100. Higher quality, delivery, sustainability and AI readiness are better. The supplied `risk` is an adverse exposure index: higher is worse. Its score is `100 − risk`. Cost score is `min(100, 100 × reference_budget / annual_cost)`. The final score is the weighted sum after dividing each nonnegative weight by the total weight. Scores are retained at full precision for ranking; equal scores use supplier ID for deterministic display order. Rank change means baseline position minus scenario position.

Costs below the reference budget receive equal cost scores: this intentionally models an affordability target rather than always rewarding the cheapest supplier. Fixed-reference scoring avoids score changes caused solely by filtering. Compare only equivalent scopes within a category. Default weights are demonstrative, not validated procurement policy. Ranking is compensatory: high quality can offset other weaknesses. High-risk leaders are therefore flagged for review rather than silently excluded.

### Bid evaluation

`TCO = annual_fee × years + setup_fee`. All costs use EUR, excluding taxes, escalation, discounting and uncertain change orders. Bids in a lot must cover the same scope. `price_score = 100 × cheapest TCO in lot / bid TCO`. The cheapest submitted bid is the reference even if ineligible. `bid_score = 0.45 × price_score + 0.40 × supplier_score + 0.15 × SLA`. Supplier score itself contains annual supplier cost, so this illustrative policy intentionally counts commercial considerations at both levels. Change this policy before use where that is inappropriate. Eligibility requires both `SLA >= minimum` and `transition_days <= maximum`; failed bids receive no eligible rank. Tied eligible scores share a rank. No automatic award is made. Bids use the active weights and budget across all lots, independently of the supplier category display filter.

### Risk model card

- Inputs: adverse risk index, delivery, quality, AI readiness.
- Generation: 2,400 uniform synthetic rows, NumPy seed 42. Label proxy = `0.55×risk + 0.25×(100−delivery) + 0.15×(100−quality) + 0.05×(100−readiness)`. Labels: Low <30, Medium <55, otherwise High.
- Training: decision tree with maximum depth 4, minimum 30 samples per leaf; seeded, stratified 75/25 training/holdout split. Model fits only the training split, then scores the holdout.
- Hybrid override: `risk >= 80` or `delivery < 60` always produces High. The UI reports the original tree class and final class separately.
- Explainability: full tree rules, input values and override reason available in the app. Accuracy and confusion matrix refer to the tree on synthetic holdout labels before overrides.
- Limits: this demonstrates a reproducible ML pipeline, not empirical predictive validity. Synthetic holdout accuracy measures policy imitation. No real-world failure probabilities, calibration, fairness assurance or causal conclusions. A supplied risk score is already a major input; this is not independent discovery of risk.
- Next validation steps: collect consented historical outcomes, define label horizon, prevent temporal leakage, compare with the transparent policy baseline, evaluate class recall and calibration, audit sensitivity and operational costs, monitor drift and require human review.

### Maturity and priorities

Each maturity dimension uses the shared anchors shown in the app. Overall level is the floor of the arithmetic mean: [1,2) is Level 1 through [4,5) Level 4; exactly 5 is Level 5. This deliberately conservative portfolio rubric is not an externally certified model. Recommendations target dimensions below 4. Priorities multiply four 1–5 factors (1–625), with a display percentage of `product / 625 × 100`; this is not a success probability. Equal products sort alphabetically. A pilot is placed in days 31–60 only when all four factors and both Data and Governance are at least 3; otherwise it is a readiness action in days 61–90. Timing, owners and measures are suggested starting points.

## Data contracts

UTF-8 comma-separated files with a header. Extra columns are ignored. Blank required fields, duplicate primary IDs, invalid or infinite numbers, out-of-range values and empty files are rejected. Maximum 10,000 rows; UI file limit 10 MB. No silent imputation. IDs are read as strings to preserve leading zeros.

| File | Required columns | Units / constraints |
|---|---|---|
| suppliers.csv | supplier_id, name, category, annual_cost, quality, delivery, risk, sustainability, ai_readiness | Unique supplier_id; annual_cost 1–1e12 EUR; other numeric fields 0–100. Delivery is on-time %, quality/sustainability/readiness are normalized indices. |
| bids.csv | bid_id, supplier_id, lot, annual_fee, setup_fee, transition_days, sla | Unique bid_id; supplier_id must exist; annual_fee 1–1e12 EUR, setup_fee 0–1e12 EUR; transition 1–3650 days; SLA 0–100 %. |
| use_cases.csv | use_case, owner, business_value, feasibility, data_readiness, adoption_potential | Unique use_case; all four numeric factors 1–5. |

Uploaded files are processed in app memory and are not intentionally saved. This is a local prototype without authentication or a durable audit log. Do not expose a public instance with sensitive data. CSV downloads prefix formula-like text with an apostrophe to reduce spreadsheet formula injection risk.

## Structure

```text
SourceAI/
├── app.py                    # Streamlit interface and orchestration
├── sourceai/
│   ├── data.py               # CSV loading and validation
│   ├── scoring.py            # Supplier and bid calculations
│   ├── risk.py               # Synthetic ML and policy overrides
│   ├── planning.py           # Maturity, priorities, roadmap
│   └── reports.py            # CSV and Markdown exports
├── data/                     # Three synthetic input CSVs
├── tests/                    # Domain and Streamlit interaction tests
├── .streamlit/config.toml    # Theme and local app settings
├── requirements.txt
└── pyproject.toml
```

## Tests

```bash
python -m pytest -q
```

Tests cover validation failures, scoring formulas and invariants, normalization, cost ordering, filtering stability, bid constraints and foreign keys, risk overrides, synthetic holdout sanity, maturity boundaries, prioritization, readiness gates, exports and UI scenario interactions. UI tests use Streamlit's [official AppTest framework](https://docs.streamlit.io/develop/api-reference/app-testing). No remote services are required.

`requirements-tested.txt` records the exact direct dependency versions verified on Python 3.14.4 on Windows. Use `python -m pip install -r requirements-tested.txt` to reproduce those direct versions; transitive dependencies are not locked. Other Python versions have not been tested in this delivery.

## Validation results

The core sourcing decision logic was validated locally on Windows using the bundled demonstration supplier and bid datasets.

- Automated test suite: **3/3 tests passed**.
- Default supplier ranking was reproduced successfully across 12 suppliers.
- Weight normalization was verified to sum exactly to **1.0**.
- Repeated runs produced the **same deterministic ranking**.
- Sensitivity testing changed rankings in the expected direction when decision priorities changed:
  - with a 60% cost weight, **Aster Services** ranked first;
  - with a 50% quality weight, **Cobalt Systems** ranked first.
- Bid evaluation correctly enforced the default constraints of **SLA >= 95** and **transition <= 90 days**.
- Ineligible bids received no eligible rank, while eligible bids were ranked independently within each service lot.

These checks validate the implemented scoring, normalization, sensitivity, determinism, and bid-constraint logic for the synthetic demonstration scenarios. They do **not** establish real-world procurement effectiveness or predictive validity.

### Validation summary

| Check | Result |
|---|---|
| Automated tests | 3/3 passed |
| Weight normalization | Sum = 1.0 |
| Ranking determinism | Passed |
| Cost-priority sensitivity | Ranking changed as expected |
| Quality-priority sensitivity | Ranking changed as expected |
| SLA constraint handling | Passed |
| Transition constraint handling | Passed |
| Eligible ranking by lot | Passed |

## Portfolio discussion and future work

The implementation separates domain logic from the UI so calculations can be reused and tested independently. A shallow tree is intentionally inspectable, while deterministic roadmap rules make the decision chain auditable. This offers concrete topics for a portfolio presentation: multi-criteria decision analysis, synthetic-data limitations, validation, human oversight, sensitivity and explainability.

Useful extensions include a Pareto-front comparison, persisted scenario snapshots, real outcome-based model evaluation, Monte Carlo cost uncertainty and evidence-backed document extraction. These are future work, not current features.


## Test portfolio optimization extension

SourceAI now includes a software-only test-portfolio analysis module for identifying potentially redundant experiments and constructing a smaller candidate plan while preserving explicit requirement coverage.

Implemented capabilities:
- requirement-to-test coverage matrix generation
- pairwise requirement-overlap analysis
- numeric outcome-similarity comparison
- redundant-test candidate detection
- cost- and duration-aware greedy test-plan optimization
- before/after reporting for selected tests, removed tests, requirement coverage, total cost and total duration
- synthetic example portfolio and automated tests

The optimizer is a transparent greedy heuristic, not proof of a globally minimal design of experiments. The sample data is synthetic and the extension does not use Volvo Penta test data or represent a validated industrial testing process.

### Example use

```python
import pandas as pd
from test_optimization import (
    redundant_candidates,
    optimize_test_plan,
)

tests = pd.read_csv("test_portfolio_sample.csv")
redundant = redundant_candidates(
    tests,
    outcome_columns=["signal_a", "signal_b"],
)
result = optimize_test_plan(tests)

print(redundant)
print(result)
```

### CV-safe extension description

- Extended SourceAI with a test-portfolio optimization module that detects overlapping tests using requirement coverage and numeric outcome similarity.
- Implemented a cost- and duration-aware greedy selector that reduces candidate test sets while preserving explicit requirement coverage.
