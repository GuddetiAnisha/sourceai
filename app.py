"""Run with: python -m streamlit run app.py"""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sourceai.data import sample, read_csv, FACTORS
from sourceai.scoring import DEFAULT_WEIGHTS, normalized_weights, rank_suppliers, evaluate_bids
from sourceai.risk import classify, train_model, model_rules
from sourceai.planning import DIMENSIONS, LEVELS, assess_maturity, prioritize, roadmap
from sourceai.reports import csv_bytes, build_report

st.set_page_config(page_title='SourceAI | Service sourcing', page_icon='◈', layout='wide')
st.markdown('### ◈ SOURCEAI / DECISION INTELLIGENCE')
st.title('Better sourcing decisions. Explained.')
st.caption('An independent portfolio prototype • Synthetic demonstration data • Human-led supplier decisions')

with st.sidebar:
    st.header('Scenario studio')
    st.caption('Adjust priorities to explore tradeoffs. Weights are normalized automatically.')
    weights = {k: st.slider('AI readiness' if k == 'ai_readiness' else k.title(), 0, 100, v, key=f'weight_{k}') for k, v in DEFAULT_WEIGHTS.items()}
    budget = st.number_input('Reference annual budget (€)', min_value=1, value=100000, step=10000)
    st.caption('Cost score = min(100, 100 × budget ÷ annual cost). Risk is inverted; all other criteria reward higher values.')
    st.divider()
    st.markdown('**Local by design**\n\nNo external AI API or credentials required. Uploaded data stays in this running app process.')

with st.expander('Data workspace — upload CSVs or inspect the samples', expanded=False):
    st.info('Supplier costs must describe a comparable annual service scope in EUR within each category. Bid fees use EUR and a common scope within each lot. Sample bids cover shared-service lots.')
    cols = st.columns(3)
    datasets = {}
    data_sources = {}
    for col, kind in zip(cols, ['suppliers', 'bids', 'use_cases']):
        with col:
            st.subheader(kind.replace('_', ' ').title())
            uploaded = st.file_uploader(f'Upload {kind} CSV', type=['csv'], key=f'upload_{kind}')
            try:
                datasets[kind] = read_csv(uploaded, kind) if uploaded is not None else sample(kind)
                data_sources[kind] = 'User-uploaded CSV' if uploaded is not None else 'Bundled synthetic CSV'
            except ValueError as exc:
                st.error(str(exc))
                st.stop()
            st.caption(f"{len(datasets[kind])} rows • {'Uploaded' if uploaded is not None else 'Synthetic sample'}")
            st.download_button('Download sample schema', csv_bytes(sample(kind)), f'{kind}.csv', 'text/csv', key=f'sample_{kind}')
    st.dataframe(datasets['suppliers'], hide_index=True, width='stretch')

try:
    normalized = normalized_weights(weights)
except ValueError as exc:
    st.warning(str(exc))
    st.stop()

suppliers = datasets['suppliers']
category = st.selectbox('Service category', sorted(suppliers.category.unique()))
portfolio = classify(rank_suppliers(suppliers, weights, budget))
ranked = rank_suppliers(portfolio[portfolio.category == category], weights, budget)
baseline = rank_suppliers(suppliers[suppliers.category == category], DEFAULT_WEIGHTS, budget)
ranked = ranked.merge(baseline[['supplier_id', 'rank']].rename(columns={'rank': 'baseline_rank'}), on='supplier_id')
ranked['rank_change'] = ranked.baseline_rank - ranked['rank']

metrics = st.columns(4)
metrics[0].metric('Suppliers in category', len(ranked))
metrics[1].metric('Leading score', f"{ranked.iloc[0]['score']:.1f}/100")
metrics[1].caption(ranked.iloc[0]['name'])
metrics[2].metric('High-risk suppliers', int((ranked.risk_class == 'High').sum()))
metrics[3].metric('Median annual cost', f'€{ranked.annual_cost.median():,.0f}')

dashboard, bid_tab, risk_tab, maturity_tab, priorities_tab, report_tab = st.tabs(['Supplier dashboard', 'Bid comparison', 'Risk lab', 'Maturity assessment', 'Use-case priorities', 'Roadmap & exports'])

with dashboard:
    st.subheader('Explore the shortlist')
    st.caption('Rank change is relative to default weights at the same budget. Equal scores sort by supplier ID; ranks are display positions.')
    left, right = st.columns([3, 2])
    with left:
        fig = px.bar(ranked.sort_values('score'), x='score', y='name', color='risk_class', orientation='h', color_discrete_map={'Low': '#14b8a6', 'Medium': '#f59e0b', 'High': '#ef4444'}, labels={'name': '', 'score': 'Weighted score / 100'})
        fig.update_xaxes(range=[0, 100])
        st.plotly_chart(fig, width='stretch')
    with right:
        st.plotly_chart(px.scatter(ranked, x='annual_cost', y='score', color='risk_class', size='sustainability', hover_name='name', labels={'annual_cost': 'Annual cost (€)', 'score': 'Weighted score'}, color_discrete_map={'Low': '#14b8a6', 'Medium': '#f59e0b', 'High': '#ef4444'}), width='stretch')
    st.dataframe(ranked[['rank', 'name', 'score', 'rank_change', 'annual_cost', 'risk_class', 'quality', 'delivery', 'sustainability', 'ai_readiness']], hide_index=True, width='stretch')
    contributions = ranked.melt(id_vars=['name'], value_vars=[f'{k}_contribution' for k in weights], var_name='criterion', value_name='points')
    st.plotly_chart(px.bar(contributions, x='name', y='points', color='criterion', title='What contributes to each score?'), width='stretch')
    st.download_button('Export supplier ranking', csv_bytes(ranked), 'supplier_ranking.csv', 'text/csv')

with bid_tab:
    st.subheader('Compare bids for a common service scope')
    st.caption('Bid score = 45% relative total cost + 40% supplier score + 15% SLA. Relative cost uses the cheapest submitted bid in each lot, including bids that fail constraints. Eligibility is a separate gate; no automatic award is made.')
    c1, c2, c3 = st.columns(3)
    years = c1.slider('Contract years', 1, 5, 3)
    min_sla = c2.slider('Minimum SLA (%)', 0, 100, 95)
    max_days = c3.slider('Maximum transition days', 1, 180, 90)
    try:
        bids = evaluate_bids(datasets['bids'], portfolio, years, min_sla, max_days)
    except ValueError as exc:
        st.warning(str(exc) + '. Upload matching bids to enable this comparison.')
        bids = pd.DataFrame()
    if not bids.empty:
        lot = st.selectbox('Bid lot', sorted(bids.lot.unique()))
        selected_bids = bids[bids.lot == lot]
        st.plotly_chart(px.bar(selected_bids, x='name', y='tco', color='eligible', hover_data=['bid_score', 'sla', 'transition_days'], labels={'tco': f'{years}-year total cost (€)'}), width='stretch')
        st.dataframe(selected_bids, hide_index=True, width='stretch')
        eligible = selected_bids[selected_bids.eligible]
        if eligible.empty:
            st.warning('No bids meet the current constraints. Review requirements or seek revised bids.')
        else:
            leader = eligible.iloc[0]
            st.success(f"Leading eligible bid: {leader['bid_id']} · {leader['name']} · score {leader['bid_score']:.1f}. Validate scope and risk before award.")
        st.download_button('Export all evaluated bids', csv_bytes(bids), 'bid_evaluation.csv', 'text/csv')

with risk_tab:
    st.subheader('Transparent risk triage')
    st.warning('Educational ML model: trained on synthetic labels generated from a policy formula. Holdout accuracy measures imitation of that formula, not real-world supplier failure prediction. Classes are not probabilities.')
    st.dataframe(ranked[['name', 'model_risk', 'risk_class', 'risk_reason']], hide_index=True, width='stretch')
    _, model_metrics = train_model()
    with st.expander('Model card, validation and learned rules'):
        st.write('Decision tree • depth 4 • minimum leaf 30 • seed 42 • 2,400 synthetic rows • 75/25 stratified split. Features: risk, delivery, quality, AI readiness. High override: input risk ≥80 or delivery <60.')
        st.write('Policy proxy = 0.55×risk + 0.25×(100−delivery) + 0.15×(100−quality) + 0.05×(100−readiness). Low <30; Medium <55; High ≥55.')
        st.metric('Synthetic holdout accuracy', f"{model_metrics['synthetic_holdout_accuracy']:.1%}")
        st.caption('Confusion matrix: rows actual, columns predicted; Low, Medium, High.')
        st.dataframe(pd.DataFrame(model_metrics['confusion_matrix'], index=['Low', 'Medium', 'High'], columns=['Low', 'Medium', 'High']))
        st.code(model_rules())
        st.write('Limits: no real outcome validation, no calibration, no fairness audit, no monitoring of changing data. The model partly repeats the supplied risk score. Real deployment needs historical outcomes, temporal validation, sensitivity analysis and independent procurement review.')

with maturity_tab:
    st.subheader('AI sourcing maturity')
    st.caption('Self-assessment, not an audited certification. Score each dimension using the anchors below. Overall level = floor of the six-dimension mean; weakest dimensions guide the roadmap.')
    with st.expander('Assessment anchors', expanded=True):
        st.markdown('**1 Initial:** ad hoc activities, no owner or evidence. **2 Emerging:** isolated pilots and informal practices. **3 Defined:** documented standards, owners and repeatable processes. **4 Managed:** measured outcomes, quality controls and regular review. **5 Optimizing:** sustained evidence of continuous improvement and organization-wide learning.')
    prompts = {'Strategy': 'Documented outcomes, investment and sponsorship', 'Data': 'Accessible supplier data with ownership and quality checks', 'Technology': 'Integrated, reproducible and monitored analytics', 'Process': 'Standard sourcing workflows and review checkpoints', 'Governance': 'Accountability, auditability and human approval', 'People': 'Skills, training and adoption support'}
    maturity_scores = {}
    columns = st.columns(2)
    for i, dim in enumerate(DIMENSIONS):
        maturity_scores[dim] = columns[i % 2].slider(dim, 1, 5, 2 if dim in ['Data', 'Governance', 'People'] else 3, help=prompts[dim], key=f'maturity_{dim}')
    maturity = assess_maturity(maturity_scores)
    st.metric('Overall maturity', f"Level {maturity['level']} · {maturity['label']}", f"Mean {maturity['average']:.2f} / 5", delta_color='off')
    fig = go.Figure(go.Scatterpolar(r=list(maturity_scores.values()) + [maturity_scores['Strategy']], theta=DIMENSIONS + ['Strategy'], fill='toself', line_color='#14b8a6'))
    fig.update_layout(polar={'radialaxis': {'visible': True, 'range': [0, 5]}}, height=380)
    st.plotly_chart(fig, width='stretch')

with priorities_tab:
    st.subheader('Prioritize the next useful pilot')
    st.caption('Edit ratings from 1 (low) to 5 (high). Product = Business Value × Feasibility × Data Readiness × Adoption Potential; maximum 625. A weak factor reduces the result multiplicatively.')
    edited = st.data_editor(datasets['use_cases'], hide_index=True, width='stretch', disabled=['use_case', 'owner'], column_config={k: st.column_config.NumberColumn(k.replace('_', ' ').title(), min_value=1, max_value=5, step=1) for k in FACTORS}, key='priorities_editor')
    try:
        priorities = prioritize(edited)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()
    st.plotly_chart(px.scatter(priorities, x='feasibility', y='business_value', size='priority_product', color='data_readiness', hover_name='use_case', hover_data=['adoption_potential', 'bottleneck'], range_x=[.5, 5.5], range_y=[.5, 5.5], color_continuous_scale='Teal'), width='stretch')
    st.dataframe(priorities, hide_index=True, width='stretch')

with report_tab:
    st.subheader('From evidence to action')
    st.caption('Rule-based recommendations update with maturity scores, use-case ratings and supplier risks. Supplier recommendations use the selected category; bid exports include all lots.')
    plan = roadmap(maturity, priorities, ranked)
    st.dataframe(plan, hide_index=True, width='stretch')
    leader = ranked.iloc[0]
    if leader.risk_class == 'High':
        st.warning(f"{leader['name']} leads the weighted ranking but is high risk. Complete mitigation review before shortlisting.")
    else:
        st.info(f"Review {leader['name']} as a shortlist candidate; confirm service scope, source evidence and bid eligibility.")
    report = build_report(ranked, bids, maturity, priorities, plan, normalized, budget, years, min_sla, max_days, category, data_sources)
    st.download_button('Download decision report (.md)', report, 'sourceai_report.md', 'text/markdown')
    st.download_button('Download roadmap CSV', csv_bytes(plan), 'roadmap.csv', 'text/csv')
    st.download_button('Download use-case priorities CSV', csv_bytes(priorities), 'use_case_priorities.csv', 'text/csv')
    st.download_button('Download maturity CSV', csv_bytes(pd.DataFrame([{'dimension': k, 'score': v, 'overall_level': maturity['level']} for k, v in maturity_scores.items()])), 'maturity.csv', 'text/csv')

st.divider()
st.caption('SourceAI adapts the theme of AI-enabled service sourcing from Ericsson Master Thesis Req ID 790824. Independent work; no affiliation or endorsement. Interviews and cross-industry academic benchmarking are outside this software prototype.')

