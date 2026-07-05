"""
streamlit_app.py  v12
Luminos Brand Analytics
5 tabs: Nielsen Retail | Funnel + Cohort | A/B Test | Churn | Retention
"""

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import norm, chi2_contingency
import plotly.graph_objects as go
import plotly.express as px

# ── Page config ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "Luminos Brand Analytics",
    page_icon  = None,
    layout     = "wide",
)

# ── CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #ffffff; }
    .kpi-card {
        background: #f0f4ff; border-left: 4px solid #4361ee;
        border-radius: 8px; padding: 16px 20px; margin: 4px 0;
    }
    .kpi-card-green {
        background: #f0fff4; border-left: 4px solid #2ec4b6;
        border-radius: 8px; padding: 16px 20px; margin: 4px 0;
    }
    .kpi-card-red {
        background: #fff5f5; border-left: 4px solid #e63946;
        border-radius: 8px; padding: 16px 20px; margin: 4px 0;
    }
    .kpi-card-orange {
        background: #fffbf0; border-left: 4px solid #f9a825;
        border-radius: 8px; padding: 16px 20px; margin: 4px 0;
    }
    .kpi-label {
        font-size: 12px; color: #6c757d; font-weight: 600;
        text-transform: uppercase; letter-spacing: 0.5px;
    }
    .kpi-value { font-size: 26px; font-weight: 700; color: #212529; margin: 4px 0; }
    .kpi-delta { font-size: 12px; color: #4361ee; font-weight: 500; }
    .section-header {
        font-size: 17px; font-weight: 700; color: #212529;
        border-bottom: 2px solid #4361ee;
        padding-bottom: 6px; margin: 24px 0 14px 0;
    }
    .callout {
        background: #fff8e1; border-left: 4px solid #f9a825;
        border-radius: 8px; padding: 14px 18px; margin: 10px 0;
        font-size: 14px; color: #212529;
    }
    .callout-green {
        background: #f0fff4; border-left: 4px solid #2ec4b6;
        border-radius: 8px; padding: 14px 18px; margin: 10px 0;
        font-size: 14px; color: #212529;
    }
    .callout-blue {
        background: #f0f4ff; border-left: 4px solid #4361ee;
        border-radius: 8px; padding: 14px 18px; margin: 10px 0;
        font-size: 14px; color: #212529;
    }
    .callout-red {
        background: #fff5f5; border-left: 4px solid #e63946;
        border-radius: 8px; padding: 14px 18px; margin: 10px 0;
        font-size: 14px; color: #212529;
    }
    .finding-card {
        background: #f8f9fa; border-radius: 10px;
        padding: 16px; margin: 6px 0; border: 1px solid #dee2e6;
    }
    .finding-title { font-size: 13px; font-weight: 700; color: #4361ee; margin-bottom: 6px; }
    .finding-body  { font-size: 13px; color: #495057; line-height: 1.5; }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════
@st.cache_data
def load_all():
    pre    = pd.read_csv('data/pre_exp_sessions.csv',
                         parse_dates=['session_start','session_end'])
    exp    = pd.read_csv('data/exp_sessions.csv',
                         parse_dates=['session_start','session_end'])
    events = pd.read_csv('data/events.csv',
                         parse_dates=['event_timestamp'])
    churn  = pd.read_csv('data/churn_scores.csv')
    ret    = pd.read_csv('data/retention_master.csv')
    orders = pd.read_csv('data/orders.csv',
                         parse_dates=['order_created_at'])
    ab     = pd.read_csv('data/ab_assignments.csv',
                         parse_dates=['assigned_at','first_exposure_at'])
    return pre, exp, events, churn, ret, orders, ab

pre, exp, events, churn_scores, retention, orders, ab = load_all()

EXP_START  = pd.Timestamp('2024-01-08')
OBS_CUTOFF = pd.Timestamp('2023-10-01')
OBS_START  = pd.Timestamp('2023-04-01')


# ══════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## Luminos Brand Analytics")
    st.markdown("**Personal Care Analytics · 2023–2024**")
    st.divider()
    st.markdown("#### Filters")
    st.caption("Applies to Funnel tab only")
    device_opts   = ["All"] + sorted(pre['device_category'].dropna().unique().tolist())
    source_opts   = ["All"] + sorted(pre['traffic_source'].dropna().unique().tolist())
    category_opts = ["All"] + sorted(pre['product_category'].dropna().unique().tolist())
    tier_opts     = ["All","low","mid","high"]
    sel_device   = st.selectbox("Device",           device_opts)
    sel_source   = st.selectbox("Traffic Source",   source_opts)
    sel_category = st.selectbox("Product Category", category_opts)
    sel_tier     = st.selectbox("Price Tier",       tier_opts)
    st.divider()
    st.markdown("#### Project Data")
    st.caption("Pre-experiment: Apr–Dec 2023")
    st.caption("Experiment:     Jan–Apr 2024")
    st.caption("Users: 20,000 | Sessions: 45,677")
    st.divider()
    st.markdown("""
    **Funnel stages:**
    Home → PLP → PDP → ATC → Checkout → Purchase
    """)
    st.divider()
    st.markdown("""
    **Nielsen markets:** 10 US markets  
    **Channels:** Food · Mass · Drug  
    **Period:** 104 weeks (2023–2024)
    """)


# ══════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════
COLORS = {
    'primary'  :'#4361ee', 'secondary':'#2ec4b6',
    'warning'  :'#f9a825', 'danger'   :'#e63946',
    'muted'    :'#adb5bd', 'light'    :'#f0f4ff',
}

def clean_layout(fig, height=380):
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#f8f9fa',
        font=dict(family='sans-serif', color='#212529'),
        margin=dict(t=40, b=20, l=20, r=20), height=height,
    )
    return fig

def kpi(col, label, value, delta, card='kpi-card'):
    with col:
        st.markdown(f"""<div class="{card}">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-delta">{delta}</div>
        </div>""", unsafe_allow_html=True)

def section(title):
    st.markdown(f'<div class="section-header">{title}</div>',
                unsafe_allow_html=True)

def callout(msg, kind=''):
    cls = f'callout{"-"+kind if kind else ""}'
    st.markdown(f'<div class="{cls}">{msg}</div>',
                unsafe_allow_html=True)

def two_prop_z(n1, x1, n2, x2):
    p1 = x1/n1 if n1>0 else 0
    p2 = x2/n2 if n2>0 else 0
    pp = (x1+x2)/(n1+n2) if (n1+n2)>0 else 0
    se = np.sqrt(pp*(1-pp)*(1/n1+1/n2)) if pp>0 else 0
    z  = (p2-p1)/se if se>0 else 0
    p  = 1 - norm.cdf(abs(z))
    return p1, p2, p, p<0.05

def sig_badge(sig, p):
    if sig:
        return (f"<span style='background:#d4edda;color:#155724;"
                f"font-size:11px;font-weight:600;padding:2px 8px;"
                f"border-radius:10px;'>✓ Significant (p={p:.4f})</span>")
    return (f"<span style='background:#fff3cd;color:#856404;"
            f"font-size:11px;font-weight:600;padding:2px 8px;"
            f"border-radius:10px;'>~ Not significant (p={p:.4f})</span>")

def guardrail_badge(passed):
    if passed:
        return ("<span style='background:#d4edda;color:#155724;"
                "font-size:11px;font-weight:600;padding:2px 8px;"
                "border-radius:10px;'>✓ Passed</span>")
    return ("<span style='background:#f8d7da;color:#721c24;"
            "font-size:11px;font-weight:600;padding:2px 8px;"
            "border-radius:10px;'>✗ Breached</span>")

def apply_filters(df):
    d = df.copy()
    if sel_device   != "All": d = d[d['device_category']  == sel_device]
    if sel_source   != "All": d = d[d['traffic_source']    == sel_source]
    if sel_category != "All": d = d[d['product_category']  == sel_category]
    if sel_tier     != "All": d = d[d['price_tier']        == sel_tier]
    return d

filtered = apply_filters(pre)


# ── Funnel HTML ──────────────────────────────────────────────────────
def build_funnel_html(stages, volumes):
    colors = ['#4361ee','#4895ef','#f9a825','#e63946','#2ec4b6','#1d9e75']
    max_v  = volumes[0]
    rows   = ""
    for i,(stage,vol) in enumerate(zip(stages, volumes)):
        bar_w    = (vol/max_v)*100
        pct      = (vol/max_v)*100
        drop_pct = ((volumes[i-1]-vol)/volumes[i-1]*100) if i>0 else None
        is_worst = (i == 3)
        drop_html = ""
        if drop_pct is not None:
            bg  = 'rgba(226,75,74,0.10)' if is_worst else 'transparent'
            clr = '#A32D2D' if is_worst else '#888'
            drop_html = (
                f'<span style="color:#ccc;margin:0 8px;">|</span>'
                f'<span style="font-size:13px;font-weight:600;'
                f'color:{clr};background:{bg};padding:2px 6px;'
                f'border-radius:4px;">{drop_pct:.1f}% drop</span>'
            )
        rows += f"""
        <div style="display:flex;align-items:center;gap:12px;margin-bottom:10px;">
          <div style="width:100px;font-size:12px;color:#6c757d;
                      text-align:right;flex-shrink:0;font-weight:500;">
            {stage}</div>
          <div style="flex:1;background:#eef1fb;border-radius:6px;
                      height:36px;position:relative;overflow:visible;">
            <div style="width:{bar_w:.1f}%;height:100%;
                        background:{colors[i]};border-radius:6px;
                        min-width:4px;"></div>
            <div style="position:absolute;left:calc({bar_w:.1f}% + 10px);
                        top:50%;transform:translateY(-50%);
                        display:flex;align-items:center;white-space:nowrap;">
              <span style="font-size:13px;font-weight:700;color:#212529;">
                {vol:,}</span>
              <span style="color:#ccc;margin:0 8px;">|</span>
              <span style="font-size:12px;color:#6c757d;">{pct:.1f}%</span>
              {drop_html}
            </div>
          </div>
        </div>"""
    return (f'<!DOCTYPE html><html><head><style>'
            f'body{{margin:0;padding:6px 2px;font-family:'
            f'-apple-system,BlinkMacSystemFont,sans-serif;'
            f'background:transparent;}}</style></head>'
            f'<body>{rows}</body></html>')


# ══════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Funnel + Cohort Analysis",
    "Nielsen Retail Analytics",
    "A/B Test Results",
    "Churn Analysis",
    "Retention Modeling",
])


# ══════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════
# TAB 1 — FUNNEL + COHORT  (v13 — slimmed to 5 sections)
# ══════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("## Funnel + Cohort Analysis")
    st.info("Note: All data in this project is synthetically generated and calibrated to realistic personal care DTC and CPG retail benchmarks. No real customer or sales data is used.")

    # Compact overview strip
    ov1, ov2, ov3, ov4 = st.columns(4)
    with ov1:
        st.markdown("""<div class="finding-card" style="text-align:center;">
            <div class="finding-title">Brand</div>
            <div class="finding-body" style="font-size:14px;font-weight:600;">Luminos Personal Care</div>
            <div class="finding-body">Shampoo · Body Wash · Conditioner</div>
        </div>""", unsafe_allow_html=True)
    with ov2:
        st.markdown("""<div class="finding-card" style="text-align:center;">
            <div class="finding-title">Data Period</div>
            <div class="finding-body" style="font-size:14px;font-weight:600;">Apr 2023 – Apr 2024</div>
            <div class="finding-body">9 months pre-experiment + 90-day A/B test</div>
        </div>""", unsafe_allow_html=True)
    with ov3:
        st.markdown("""<div class="finding-card" style="text-align:center;">
            <div class="finding-title">Users</div>
            <div class="finding-body" style="font-size:14px;font-weight:600;">20,000</div>
            <div class="finding-body">12,000 pre-experiment · 8,000 experiment</div>
        </div>""", unsafe_allow_html=True)
    with ov4:
        st.markdown("""<div class="finding-card" style="text-align:center;">
            <div class="finding-title">Workstream</div>
            <div class="finding-body" style="font-size:14px;font-weight:600;">DTC Analytics</div>
            <div class="finding-body">Funnel · A/B Test · Churn · Retention</div>
        </div>""", unsafe_allow_html=True)

    with st.expander("About this workstream"):
        st.markdown("""**Workstream 1 — DTC Analytics (Luminos.com)**

Luminos is a mid-size personal care brand experiencing flat DTC revenue despite growing paid social traffic.
This analysis covers five areas:

- **Funnel Analysis:** Full 6-stage funnel (Home to Purchase) enriched with behavioral intent levels and drop-off reason classification
- **A/B Testing:** 90-day RCT testing PDP value messaging — ingredient benefit callouts and free shipping threshold banner
- **Churn Modeling:** Logistic Regression and XGBoost (AUC 0.92) scoring users for 60-day churn risk
- **Retention Modeling:** RFM segmentation, 6-month CLV prediction (BG/NBD + Gamma-Gamma), and 30-day purchase propensity scoring

Use sidebar filters to explore funnel metrics by device, traffic source, product category, or price tier.
""")

    # ── Section 1: Full Funnel Overview ───────────────────────────────
    section("Section 1 — Full Funnel Overview")

    total    = len(filtered)
    home     = filtered['reached_home'].sum()
    plp      = filtered['reached_plp'].sum()
    pdp      = filtered['reached_pdp'].sum()
    atc      = filtered['reached_atc'].sum()
    checkout = filtered['reached_checkout'].sum()
    purchase = filtered['purchased'].sum()
    aov      = filtered.loc[filtered['purchased']==True,'order_revenue'].mean()
    aov      = aov if not pd.isna(aov) else 0

    c1,c2,c3,c4,c5,c6 = st.columns(6)
    for col,lbl,val,dlt in [
        (c1,"Home",       f"{home:,}",     "100%"),
        (c2,"PLP",        f"{plp:,}",      f"{plp/home:.1%} of home"),
        (c3,"PDP",        f"{pdp:,}",      f"{pdp/plp:.1%} of PLP"),
        (c4,"Add to Cart",f"{atc:,}",      f"{atc/pdp:.1%} of PDP"),
        (c5,"Checkout",   f"{checkout:,}", f"{checkout/atc:.1%} of ATC"),
        (c6,"Purchase",   f"{purchase:,}", f"{purchase/home:.1%} CVR"),
    ]:
        kpi(col, lbl, val, dlt)

    st.markdown(f"<p style='color:#6c757d;font-size:13px;margin-top:6px;'>"
                f"AOV: <strong>${aov:.2f}</strong> &nbsp;|&nbsp; "
                f"Overall CVR: <strong>{purchase/home:.2%}</strong></p>",
                unsafe_allow_html=True)

    stages_lbl = ['Home','PLP','PDP','Add to Cart','Checkout','Purchase']
    volumes    = [int(home),int(plp),int(pdp),int(atc),int(checkout),int(purchase)]
    components.html(build_funnel_html(stages_lbl, volumes), height=300, scrolling=False)

    step_rates  = [plp/home, pdp/plp, atc/pdp, checkout/atc, purchase/checkout]
    step_labels = ['Home→PLP','PLP→PDP','PDP→ATC','ATC→Checkout','Checkout→Buy']
    worst_idx   = int(np.argmin(step_rates))
    callout(f"<strong>Biggest drop-off:</strong> "
            f"<strong>{step_labels[worst_idx]}</strong> — "
            f"<strong>{(1-step_rates[worst_idx])*100:.1f}%</strong> "
            f"of users drop off at this stage.")

    st.divider()

    # ── Section 2: Drop-off Reason ────────────────────────────────────
    section("Section 2 — Why Are Users Dropping Off?")
    st.markdown("""Instead of just knowing **where** users drop off,
    we know **why**. Each reason is inferred from behavioral signals —
    scroll depth, session duration, number of PDP views, and return visits.
    Only fixable reasons are addressable with PDP content changes.""")

    with st.expander("Drop-off Reason Definitions"):
        ec1, ec2 = st.columns(2)
        with ec1:
            st.markdown("""
| Reason | Category | Signal |
|---|---|---|
| **Decision Friction** | Fixable | High scroll (0.70+), long session, 2+ views, returns |
| **Comparison Intent** | Fixable | Medium scroll, short session, cross-session return |
| **Price Barrier** | Partial | Low scroll (0.10–0.30), short session, no return |
| **Distraction** | External | Mid engagement, abrupt exit, no clear pattern |
""")
        with ec2:
            st.markdown("""
| Reason | Category | Signal |
|---|---|---|
| **Price Shock** | Upstream | Very short session, exits on price reveal |
| **Impulse Faded** | Upstream | Near-zero scroll, very short session, no return |
| **Wrong Audience** | Upstream | Extremely short session (2–10s), minimal scroll |
| **Ad/PDP Mismatch** | Upstream | Short session, ad promised something PDP did not |
""")
        st.caption("Fixable = addressable with PDP content changes · Partial = may respond to price intervention · Upstream = ad/targeting problem · External = outside product control")

    pdp_df = filtered[filtered['reached_pdp']==True].copy()
    non_conv    = pdp_df[pdp_df['dropoff_reason']!='converted'].copy()
    reason_dist = non_conv['dropoff_reason'].value_counts()
    reason_pct  = reason_dist / len(non_conv)

    fixable_pct  = (reason_pct.get('decision_friction',0) +
                    reason_pct.get('comparison_intent',0))
    partial_pct  = reason_pct.get('price_barrier',0)
    upstream_pct = (reason_pct.get('impulse_faded',0) +
                    reason_pct.get('price_shock',0) +
                    reason_pct.get('wrong_audience',0) +
                    reason_pct.get('ad_pdp_mismatch',0))
    external_pct = (reason_pct.get('distraction',0) +
                    reason_pct.get('out_of_stock',0))

    dr1, dr2 = st.columns(2)

    with dr1:
        rs = reason_pct.sort_values(ascending=True)
        bar_colors = [
            '#2ec4b6' if r in ['decision_friction','comparison_intent'] else
            '#f9a825' if r == 'price_barrier' else '#e63946'
            for r in rs.index
        ]
        fig_reason = go.Figure(go.Bar(
            x=rs.values*100, y=rs.index,
            orientation='h', marker_color=bar_colors, opacity=0.85,
            text=[f'{v:.1%}' for v in rs.values], textposition='outside',
        ))
        fig_reason.update_layout(
            title='Drop-off Reason Distribution<br>'
                  '<sup>Teal = fixable · Orange = partial · Red = upstream</sup>',
            xaxis=dict(title='% of Non-Converting PDP Sessions',
                       range=[0, rs.values.max()*140]),
        )
        clean_layout(fig_reason, height=340)
        st.plotly_chart(fig_reason, use_container_width=True)

    with dr2:
        fig_pie = go.Figure(go.Pie(
            labels=[f'Fixable\n{fixable_pct:.1%}',
                    f'Partial\n{partial_pct:.1%}',
                    f'Upstream\n{upstream_pct:.1%}',
                    f'External\n{external_pct:.1%}'],
            values=[fixable_pct, partial_pct, upstream_pct, external_pct],
            hole=0.45,
            marker=dict(colors=['#2ec4b6','#f9a825','#e63946','#adb5bd']),
            textinfo='label+percent', textfont=dict(size=12),
        ))
        fig_pie.update_layout(
            title='Drop-off Category Mix<br>'
                  '<sup>Fixable = decision_friction + comparison_intent</sup>',
        )
        clean_layout(fig_pie, height=340)
        st.plotly_chart(fig_pie, use_container_width=True)

    d1,d2,d3,d4 = st.columns(4)
    for col,lbl,val,card in [
        (d1,"Fixable",  f"{fixable_pct:.1%}","kpi-card-green"),
        (d2,"Partial",  f"{partial_pct:.1%}","kpi-card-orange"),
        (d3,"Upstream", f"{upstream_pct:.1%}","kpi-card-red"),
        (d4,"External", f"{external_pct:.1%}","kpi-card"),
    ]:
        kpi(col, lbl, val, "of non-converting PDP sessions", card)



    st.divider()

    # ── Section 3: Scroll Depth ───────────────────────────────────────
    section("Section 3 — Scroll Depth vs Conversion")
    st.markdown("""**One signal, one chart.**
    Users who read the full page convert at a dramatically higher rate.
    Decision-friction users scroll deeply — confirming genuine interest,
    not disengagement.""")

    pdp_df['scroll_bucket'] = pd.cut(
        pdp_df['avg_scroll_depth'],
        bins=[0, 0.25, 0.50, 0.75, 1.0],
        labels=['0–25%','25–50%','50–75%','75–100%']
    )
    scroll_atc = (
        pdp_df.groupby('scroll_bucket', observed=True)
        .agg(sessions=('reached_atc','count'), atc=('reached_atc','sum'))
        .reset_index()
    )
    scroll_atc['atc_rate'] = scroll_atc['atc'] / scroll_atc['sessions']

    sc_col1, sc_col2 = st.columns([1.2, 1])

    with sc_col1:
        fig_scroll = go.Figure(go.Bar(
            x=scroll_atc['scroll_bucket'].astype(str),
            y=scroll_atc['atc_rate']*100,
            marker_color=['#e63946','#f9a825','#4361ee','#2ec4b6'],
            opacity=0.85,
            text=[f'{v:.1%}' for v in scroll_atc['atc_rate']],
            textposition='outside',
        ))
        fig_scroll.update_layout(
            title='ATC Rate by Scroll Depth',
            yaxis=dict(title='ATC Rate (%)', range=[0, 90]),
            xaxis_title='Scroll Depth Range',
        )
        clean_layout(fig_scroll, height=320)
        st.plotly_chart(fig_scroll, use_container_width=True)

    with sc_col2:
        atc_low  = scroll_atc[scroll_atc['scroll_bucket']=='0–25%']['atc_rate'].values[0]
        atc_high = scroll_atc[scroll_atc['scroll_bucket']=='75–100%']['atc_rate'].values[0]
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown(f"""<div class="kpi-card-red">
            <div class="kpi-label">0–25% Scroll ATC Rate</div>
            <div class="kpi-value">{atc_low:.1%}</div>
            <div class="kpi-delta">low engagement</div>
        </div>""", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"""<div class="kpi-card-green">
            <div class="kpi-label">75–100% Scroll ATC Rate</div>
            <div class="kpi-value">{atc_high:.1%}</div>
            <div class="kpi-delta">{atc_high/atc_low:.1f}x lift vs low scroll</div>
        </div>""", unsafe_allow_html=True)



    st.divider()

    # ── Section 4: Behavioral Cohorts ─────────────────────────────────
    section("Section 4 — Who Are the High-Value Users?")
    st.markdown("""Three behavioral cohorts emerge from PDP engagement.
    **Cohort B is the money:** high scroll, multiple visits, but not converting.
    Once users hit Cohort C (cart), purchase rate is very high.
    The gap between B and C is what the A/B test is designed to close.""")

    def assign_cohort(row):
        if row['reached_atc']:         return 'C_atc_user'
        elif row['pdp_view_count']>=2: return 'B_repeat_viewer'
        else:                          return 'A_single_viewer'

    pdp_df['beh_cohort'] = pdp_df.apply(assign_cohort, axis=1)

    cohort_metrics = (
        pdp_df.groupby('beh_cohort')
        .agg(users=('beh_cohort','count'),
             purchased=('purchased','sum'),
             avg_pdp_views=('pdp_view_count','mean'),
             avg_scroll=('avg_scroll_depth','mean'),
             avg_dur_sec=('session_duration_sec','mean'))
        .reset_index()
    )
    # For each cohort, show the meaningful "next step" rate:
    # A (single viewer): % who go on to ATC
    # B (repeat viewer): % who go on to ATC (they came back but still didn't ATC)
    # C (added to cart): % who purchase
    cohort_metrics['next_step_rate'] = np.where(
        cohort_metrics['beh_cohort'] == 'C_atc_user',
        cohort_metrics['purchased'] / cohort_metrics['users'],
        cohort_metrics['purchased'] / cohort_metrics['users']  # 0 for A and B - recalculate below
    )
    # Recalculate properly using pdp_df
    for cohort in ['A_single_viewer', 'B_repeat_viewer', 'C_atc_user']:
        sub = pdp_df[pdp_df['beh_cohort'] == cohort]
        if cohort == 'C_atc_user':
            rate = sub['purchased'].sum() / len(sub) if len(sub) > 0 else 0
            label_str = 'ATC to Purchase'
        else:
            rate = sub['reached_atc'].sum() / len(sub) if len(sub) > 0 else 0
            label_str = 'PDP to ATC'
        cohort_metrics.loc[cohort_metrics['beh_cohort']==cohort, 'next_step_rate'] = rate

    cohort_metrics['cvr']         = cohort_metrics['purchased']/cohort_metrics['users']
    cohort_metrics['avg_dur_min'] = cohort_metrics['avg_dur_sec']/60
    cohort_metrics['pct_of_pdp']  = cohort_metrics['users']/len(pdp_df)
    cohort_metrics = cohort_metrics.sort_values('beh_cohort')

    label_map = {
        'A_single_viewer':'A: Single PDP View',
        'B_repeat_viewer':'B: Repeat PDP Viewer (2+)',
        'C_atc_user'     :'C: Added to Cart',
    }

    a_row = cohort_metrics[cohort_metrics['beh_cohort']=='A_single_viewer'].iloc[0]
    b_row = cohort_metrics[cohort_metrics['beh_cohort']=='B_repeat_viewer'].iloc[0]
    c_row = cohort_metrics[cohort_metrics['beh_cohort']=='C_atc_user'].iloc[0]

    bc1, bc2, bc3 = st.columns(3)
    for col, row, accent, bg in [
        (bc1, a_row, '#adb5bd', '#f8f9fa'),
        (bc2, b_row, '#f9a825', '#fffbf0'),
        (bc3, c_row, '#2ec4b6', '#f0fff4'),
    ]:
        label = label_map[row['beh_cohort']]
        with col:
            st.markdown(f"""<div style="background:{bg};border-radius:10px;
                padding:18px;border-left:4px solid {accent};
                border:1px solid #dee2e6;text-align:center;">
                <div style="font-size:13px;font-weight:700;color:{accent};
                            margin-bottom:8px;">{label}</div>
                <div style="font-size:28px;font-weight:700;color:#212529;">
                    {int(row['users']):,}</div>
                <div style="font-size:12px;color:#6c757d;margin:3px 0;">
                    {row['pct_of_pdp']:.1%} of PDP sessions</div>
                <div style="font-size:12px;color:#495057;margin:3px 0;">
                    Scroll: {row['avg_scroll']:.2f} &nbsp;·&nbsp;
                    {row['avg_pdp_views']:.1f} views &nbsp;·&nbsp;
                    {row['avg_dur_min']:.1f} min</div>
                <div style="font-size:13px;font-weight:600;color:{accent};
                            margin-top:8px;">{"ATC→Purchase: " if row["beh_cohort"]=="C_atc_user" else "PDP→ATC: "}{row["next_step_rate"]:.1%}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    ch_left, ch_right = st.columns(2)

    with ch_left:
        # Engagement quality: scroll depth + session duration per cohort
        fig_eng = go.Figure()
        cohort_order = ['A_single_viewer', 'B_repeat_viewer', 'C_atc_user']
        cohort_labels = [label_map[c] for c in cohort_order]
        cohort_colors = ['#adb5bd', '#f9a825', '#2ec4b6']

        scroll_vals = [cohort_metrics[cohort_metrics['beh_cohort']==c]['avg_scroll'].values[0]
                       for c in cohort_order]
        dur_vals    = [cohort_metrics[cohort_metrics['beh_cohort']==c]['avg_dur_min'].values[0]
                       for c in cohort_order]
        view_vals   = [cohort_metrics[cohort_metrics['beh_cohort']==c]['avg_pdp_views'].values[0]
                       for c in cohort_order]

        x = np.arange(len(cohort_labels))
        width = 0.28

        fig_eng.add_trace(go.Bar(
            name='Avg Scroll Depth',
            x=cohort_labels, y=scroll_vals,
            marker_color=['rgba(173,181,189,0.85)',
                          'rgba(249,168,37,0.85)',
                          'rgba(46,196,182,0.85)'],
            text=[f'{v:.2f}' for v in scroll_vals],
            textposition='outside',
            offsetgroup=0,
        ))
        fig_eng.add_trace(go.Bar(
            name='Avg PDP Views',
            x=cohort_labels, y=view_vals,
            marker_color=['rgba(173,181,189,0.45)',
                          'rgba(249,168,37,0.45)',
                          'rgba(46,196,182,0.45)'],
            text=[f'{v:.1f}' for v in view_vals],
            textposition='outside',
            offsetgroup=1,
        ))
        fig_eng.update_layout(
            barmode='group',
            title='Engagement Quality by Cohort<br>'
                  '<sup>Cohort B has the highest scroll depth — not disinterest</sup>',
            yaxis=dict(title='Value', range=[0, max(scroll_vals + view_vals)*1.4]),
            legend=dict(orientation='h', y=1.15, font=dict(size=9)),
        )
        clean_layout(fig_eng, height=320)
        st.plotly_chart(fig_eng, use_container_width=True)

    with ch_right:
        # Purchase rate: only Cohort C converts — stark contrast
        purchase_rates = [
            cohort_metrics[cohort_metrics['beh_cohort']==c]['next_step_rate'].values[0]
            for c in cohort_order
        ]
        fig_pur = go.Figure(go.Bar(
            x=cohort_labels,
            y=[v*100 for v in purchase_rates],
            marker_color=['#adb5bd', '#f9a825', '#2ec4b6'],
            opacity=0.85,
            text=[f'{v:.1%}' for v in purchase_rates],
            textposition='outside',
        ))
        fig_pur.update_layout(
            title='Purchase Rate by Cohort<br>'
                  '<sup>A + B: PDP→ATC · C: ATC→Purchase</sup>',
            yaxis=dict(title='Rate (%)',
                       range=[0, max(purchase_rates)*150 if max(purchase_rates)>0 else 60]),
        )
        clean_layout(fig_pur, height=320)
        st.plotly_chart(fig_pur, use_container_width=True)



    st.divider()

    # ── Section 5: Cohort Retention ───────────────────────────────────
    section("Section 5 — Acquisition Cohort Retention")
    st.markdown("""Flat retention across all cohorts = **structural loyalty problem**,
    not a campaign issue. The same pattern Nielsen shows at retail —
    Luminos loses repurchase customers in both channels.""")

    pre_events = events[events['event_timestamp'] < EXP_START].copy()
    pre_events['month'] = pre_events['event_timestamp'].dt.to_period('M')

    user_first = (
        pre_events.groupby('user_pseudo_id')['month']
        .min().reset_index().rename(columns={'month':'cohort_month'})
    )
    user_activity = (
        pre_events.groupby(['user_pseudo_id','month'])
        .size().reset_index(name='event_count')
    )
    cohort_data = user_activity.merge(user_first, on='user_pseudo_id', how='left')
    cohort_data['period_number'] = (
        cohort_data['month'] - cohort_data['cohort_month']
    ).apply(lambda x: x.n)

    cohort_counts = (
        cohort_data.groupby(['cohort_month','period_number'])
        ['user_pseudo_id'].nunique().reset_index(name='users')
    )
    cohort_matrix = cohort_counts.pivot_table(
        index='cohort_month', columns='period_number', values='users'
    )
    cohort_sizes     = cohort_matrix[0]
    retention_matrix = cohort_matrix.divide(cohort_sizes, axis=0)
    retention_matrix.index = retention_matrix.index.astype(str)

    max_periods = min(8, retention_matrix.shape[1])
    plot_matrix = retention_matrix.iloc[:,:max_periods].copy()
    avg_ret     = plot_matrix.mean()
    m1_ret = avg_ret.iloc[1] if len(avg_ret)>1 else 0
    m3_ret = avg_ret.iloc[3] if len(avg_ret)>3 else 0

    rk1, rk2, rk3 = st.columns(3)
    kpi(rk1,"Cohorts Tracked","9","Apr – Dec 2023")
    kpi(rk2,"Avg Month-1 Retention",f"{m1_ret:.1%}",
        f"{(1-m1_ret):.1%} lost after first visit","kpi-card-orange")
    kpi(rk3,"Avg Month-3 Retention",f"{m3_ret:.1%}",
        "steep decay","kpi-card-red")

    z_vals    = plot_matrix.values.astype(float)
    x_labels  = [f'Month {i}' for i in range(max_periods)]
    text_vals = [[f'{v:.0%}' if not np.isnan(v) else '' for v in row]
                 for row in z_vals]

    fig_hm = go.Figure(go.Heatmap(
        z=z_vals, x=x_labels, y=plot_matrix.index.tolist(),
        colorscale='Blues', text=text_vals,
        texttemplate='%{text}', textfont=dict(size=11),
        zmin=0, zmax=1, showscale=True,
        colorbar=dict(tickformat='.0%', title='Retention'),
    ))
    fig_hm.update_layout(
        title='Monthly Cohort Retention Heatmap — Luminos.com',
        xaxis_title='Months Since Acquisition',
        yaxis_title='Cohort (Acquisition Month)',
    )
    clean_layout(fig_hm, height=360)
    st.plotly_chart(fig_hm, use_container_width=True)




# ══════════════════════════════════════════════════════════════════════
# TAB 2 — NIELSEN RETAIL ANALYTICS  (v13 — slimmed to 4 sections)
# ══════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("## Nielsen Retail Analytics")
    st.info("Note: All data in this project is synthetically generated and calibrated to realistic personal care DTC and CPG retail benchmarks. No real customer or sales data is used.")
    callout("""<strong>Workstream 2 — Retail Measurement (Nielsen):</strong>
    Luminos holds 14.6% national dollar share in the personal care category,
    ranking fourth behind Dove, Pantene, and Suave. This analysis uses Nielsen-style
    retail scanner data across 10 US markets and three channels (Food, Mass, Drug)
    over 104 weeks (2023–2024). The core diagnostic question: is Luminos losing share
    because stores are reducing shelf presence (distribution problem) or because
    consumers are not choosing it at shelf (velocity problem)?
    The answer determines whether the fix is a sales force investment
    or a marketing and promotion investment.""", 'blue')

    @st.cache_data
    def load_nielsen():
        weekly  = pd.read_csv('nielsen/data/nielsen_weekly.csv',
                              parse_dates=['week_end_date'])
        mkt_cls = pd.read_csv('nielsen/data/market_classification.csv')
        diag    = pd.read_csv('nielsen/data/market_diagnostic.csv')
        weekly  = weekly.merge(mkt_cls[['market','market_type']],
                               on='market', how='left')
        return weekly, mkt_cls, diag

    try:
        nielsen, mkt_cls, diag = load_nielsen()
        nielsen_loaded = True
    except FileNotFoundError:
        nielsen_loaded = False
        st.warning("⚠️ Nielsen data not found. Place CSVs in `nielsen/data/` "
                   "and restart the app.")

    if nielsen_loaded:

        BRAND_COLORS_N = {
            'Luminos':       '#2563EB',
            'Dove':          '#DC2626',
            'Pantene':       '#D97706',
            'Suave':         '#059669',
            'Private Label': '#6B7280',
        }
        MKT_COLORS = {
            'declining': '#DC2626',
            'stable':    '#6B7280',
            'growing':   '#059669',
        }

        # ── Section 1: National Position ──────────────────────────────
        section("Section 1 — National Market Position")

        nat_share = (
            nielsen.groupby('brand')[['dollar_sales','unit_sales']]
            .sum()
            .assign(
                dollar_share=lambda x: x['dollar_sales']/x['dollar_sales'].sum(),
                unit_share  =lambda x: x['unit_sales']  /x['unit_sales'].sum()
            )
            .reset_index()
            .sort_values('dollar_share', ascending=False)
        )

        luminos_row  = nat_share[nat_share['brand']=='Luminos'].iloc[0]
        dove_row     = nat_share[nat_share['brand']=='Dove'].iloc[0]
        n_declining  = (mkt_cls['market_type']=='declining').sum()
        n_growing    = (mkt_cls['market_type']=='growing').sum()
        luminos_rank = nat_share['brand'].tolist().index('Luminos') + 1
        luminos_acv  = nielsen[nielsen['brand']=='Luminos']['acv_distribution'].mean()
        dove_acv     = nielsen[nielsen['brand']=='Dove']['acv_distribution'].mean()

        nk1,nk2,nk3,nk4,nk5 = st.columns(5)
        for col,lbl,val,dlt,card in [
            (nk1,"National Dollar Share",
             f"{luminos_row['dollar_share']:.1%}",
             f"#{luminos_rank} in category","kpi-card"),
            (nk2,"Share Gap vs Dove",
             f"{(dove_row['dollar_share']-luminos_row['dollar_share'])*100:.1f} pp",
             f"Dove at {dove_row['dollar_share']:.1%}","kpi-card-orange"),
            (nk3,"ACV vs Dove",
             f"{(luminos_acv-dove_acv)*100:.0f} pp gap",
             f"Luminos {luminos_acv:.0%} vs Dove {dove_acv:.0%}","kpi-card-orange"),
            (nk4,"Declining Markets",
             f"{n_declining} of 10",
             "by share slope analysis","kpi-card-red"),
            (nk5,"Growing Markets",
             f"{n_growing} of 10",
             "holding or gaining share","kpi-card-green"),
        ]:
            kpi(col, lbl, val, dlt, card)

        st.divider()

        # ── Section 2: Market Classification ──────────────────────────
        section("️ Section 2 — Which Markets Are in Trouble?")
        st.markdown("""Market type is **not hardcoded** — derived by fitting a linear
        slope to Luminos's weekly dollar share per market over 104 weeks.
        Declining < –0.5 pp/yr · Growing > +0.3 pp/yr · else Stable.""")

        mc1, mc2 = st.columns([1.3, 1])

        with mc1:
            slope_plot  = mkt_cls.sort_values('slope_annual_pp')
            bar_colors  = [MKT_COLORS[t] for t in slope_plot['market_type']]
            fig_slope   = go.Figure(go.Bar(
                x=slope_plot['slope_annual_pp'],
                y=slope_plot['market'],
                orientation='h',
                marker_color=bar_colors, opacity=0.85,
                text=[f'{v:+.2f} pp/yr' for v in slope_plot['slope_annual_pp']],
                textposition='outside',
            ))
            fig_slope.add_vline(x=0, line_color='black', line_width=0.8)
            fig_slope.add_vline(x=-0.5, line_dash='dash',
                                line_color=MKT_COLORS['declining'], opacity=0.5)
            fig_slope.add_vline(x=0.3, line_dash='dash',
                                line_color=MKT_COLORS['growing'], opacity=0.5)
            fig_slope.update_layout(
                title='Luminos Dollar Share Slope by Market (pp/yr)',
                xaxis_title='Annual Share Change (pp)',
            )
            clean_layout(fig_slope, height=380)
            st.plotly_chart(fig_slope, use_container_width=True)

        with mc2:
            tbl = mkt_cls[['market','market_type','slope_annual_pp',
                           'share_y1_avg','share_y2_avg','change_pp']].copy()
            tbl['slope_annual_pp'] = tbl['slope_annual_pp'].map('{:+.2f}'.format)
            tbl['share_y1_avg']    = tbl['share_y1_avg'].map('{:.1%}'.format)
            tbl['share_y2_avg']    = tbl['share_y2_avg'].map('{:.1%}'.format)
            tbl['change_pp']       = tbl['change_pp'].map('{:+.2f}'.format)
            tbl.columns = ['Market','Type','Slope pp/yr','Y1 Share','Y2 Share','Δ pp']
            tbl = tbl.sort_values('Slope pp/yr')
            st.dataframe(tbl, use_container_width=True, hide_index=True, height=360)

        worst_mkt = mkt_cls.sort_values('slope_annual_pp').iloc[0]
        callout(f"""<strong>Worst market: {worst_mkt['market']}</strong> —
        declining at {worst_mkt['slope_annual_pp']:+.2f} pp/yr
        ({worst_mkt['share_y1_avg']:.1%} → {worst_mkt['share_y2_avg']:.1%}).
        <strong>{n_declining} of 10 markets</strong> are in structural decline.""", 'red')

        st.divider()

        # ── Section 3: Distribution vs Velocity 2×2 ───────────────────
        section("Section 3 — Why Are They Losing Share?")
        st.markdown("""**The key diagnostic:** is it a distribution problem
        (stores dropping Luminos) or a velocity problem (consumers not choosing it)?
        The answer determines whether the fix is a sales force issue or a
        marketing issue.""")

        fig_2x2 = go.Figure()

        shapes = [
            dict(type='rect', x0=-5.5, x1=0, y0=0,   y1=10,
                 fillcolor='rgba(220,38,38,0.05)', line_width=0),
            dict(type='rect', x0=-5.5, x1=0, y0=-15, y1=0,
                 fillcolor='rgba(220,38,38,0.10)', line_width=0),
            dict(type='rect', x0=0, x1=3, y0=-15, y1=0,
                 fillcolor='rgba(217,119,6,0.05)', line_width=0),
            dict(type='rect', x0=0, x1=3, y0=0,   y1=10,
                 fillcolor='rgba(5,150,105,0.05)', line_width=0),
        ]
        annotations = [
            dict(x=-5.3, y=9.5,  text='Velocity problem',  showarrow=False,
                 font=dict(size=10, color=MKT_COLORS['declining']), xanchor='left'),
            dict(x=-5.3, y=-13,  text='Both problems',     showarrow=False,
                 font=dict(size=10, color=MKT_COLORS['declining']), xanchor='left',
                 font_weight='bold'),
            dict(x=0.15,  y=9.5, text='Healthy',           showarrow=False,
                 font=dict(size=10, color=MKT_COLORS['growing']), xanchor='left'),
            dict(x=0.15,  y=-13, text='Distribution only', showarrow=False,
                 font=dict(size=10, color='#D97706'), xanchor='left'),
        ]

        for _, row in diag.iterrows():
            color = MKT_COLORS.get(row['market_type'], '#6B7280')
            size  = 80 + abs(row['change_pp']) * 50
            fig_2x2.add_trace(go.Scatter(
                x=[row['acv_change_pp']], y=[row['vel_change_pct']],
                mode='markers+text',
                marker=dict(size=size/10, color=color,
                            opacity=0.85, line=dict(color='white', width=2)),
                text=[row['market']], textposition='top center',
                textfont=dict(size=9, color=color),
                name=row['market_type'], showlegend=False,
            ))

        fig_2x2.add_hline(y=0, line_color='black', line_width=0.8)
        fig_2x2.add_vline(x=0, line_color='black', line_width=0.8)
        fig_2x2.update_layout(
            title='Distribution vs Velocity Diagnostic<br>'
                  '<sup>Bubble size = share loss magnitude · '
                  'Red = declining · Gray = stable · Green = growing</sup>',
            xaxis=dict(title='ACV Distribution Change (pp, Y1→Y2)', range=[-5.5, 3]),
            yaxis=dict(title='Velocity Change (%, Y1→Y2)', range=[-14, 10]),
            shapes=shapes, annotations=annotations,
        )
        clean_layout(fig_2x2, height=420)
        st.plotly_chart(fig_2x2, use_container_width=True)

        dv1, dv2, dv3 = st.columns(3)
        driver_counts = diag['driver'].value_counts()
        for col, driver, card in [
            (dv1, 'Both ACV & Velocity', 'kpi-card-red'),
            (dv2, 'Velocity-led',         'kpi-card-orange'),
            (dv3, 'Distribution-led',     'kpi-card'),
        ]:
            kpi(col, driver, str(driver_counts.get(driver, 0)),
                'of declining markets', card)

        callout("""<strong>The problem is primarily velocity</strong> — consumers
        are choosing competitors at shelf even where Luminos is available.
        This is a consumer pull problem, not a shelf presence problem.
        The fix is marketing, pricing, and promotion — not the sales force.""", 'red')

        st.divider()

        # ── Section 4: Price + Promo + Cross-Channel ──────────────────
        section("Section 4 — Root Cause: Price & Promo Position")
        st.markdown("""Luminos sits in an awkward mid-tier — priced above value
        (Suave) but promoting less frequently than both Suave and Dove.
        In a price-elastic category (elasticity –3.1), under-promoting
        directly costs velocity.""")

        promo_nat = (
            nielsen.groupby('brand')
            .agg(promo_freq=('on_promo','mean'),
                 avg_price =('effective_price','mean'),
                 avg_share =('dollar_share','mean'))
            .reset_index()
        )

        pp1, pp2 = st.columns(2)

        with pp1:
            fig_bubble = go.Figure()
            for _, row in promo_nat.iterrows():
                fig_bubble.add_trace(go.Scatter(
                    x=[row['promo_freq']*100],
                    y=[row['avg_price']],
                    mode='markers+text',
                    marker=dict(
                        size=row['avg_share']*600,
                        color=BRAND_COLORS_N[row['brand']],
                        opacity=0.85,
                        line=dict(color='white', width=2)
                    ),
                    text=[row['brand']],
                    textposition='top center',
                    textfont=dict(size=10, color=BRAND_COLORS_N[row['brand']]),
                    name=row['brand'], showlegend=False,
                ))

            # Luminos crosshairs
            lum = promo_nat[promo_nat['brand']=='Luminos'].iloc[0]
            fig_bubble.add_hline(y=lum['avg_price'], line_dash='dot',
                                 line_color='#2563EB', opacity=0.4)
            fig_bubble.add_vline(x=lum['promo_freq']*100, line_dash='dot',
                                 line_color='#2563EB', opacity=0.4)
            fig_bubble.update_layout(
                title='Price vs Promo Frequency<br>'
                      '<sup>Bubble size = national dollar share · '
                      'Luminos crosshairs shown</sup>',
                xaxis=dict(title='Promo Frequency (% weeks on promo)', range=[15, 48]),
                yaxis=dict(title='Avg Effective Price per Unit ($)'),
            )
            clean_layout(fig_bubble, height=380)
            st.plotly_chart(fig_bubble, use_container_width=True)

        with pp2:
            # Key numbers
            lum_freq  = lum['promo_freq']
            suave_row = promo_nat[promo_nat['brand']=='Suave'].iloc[0]
            dove_row2 = promo_nat[promo_nat['brand']=='Dove'].iloc[0]
            lum_price = lum['avg_price']
            suave_price = suave_row['avg_price']
            dove_price  = dove_row2['avg_price']

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("""<div class="finding-card">
                <div class="finding-title">The Positioning Problem</div>
                <div class="finding-body">
                Luminos is priced <strong>above Suave</strong> but promotes
                <strong>less frequently than both Suave and Dove</strong>.
                Consumers see no compelling reason to choose Luminos —
                it's neither the value option nor the premium option.
                </div>
            </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            pp_cols = st.columns(2)
            for col, lbl, val, card in [
                (pp_cols[0], "Luminos Promo Freq",  f"{lum_freq:.0%}", "kpi-card-orange"),
                (pp_cols[1], "Suave Promo Freq",    f"{suave_row['promo_freq']:.0%}", "kpi-card"),
            ]:
                kpi(col, lbl, val, "of weeks on promo", card)

            st.markdown("<br>", unsafe_allow_html=True)

            pp_cols2 = st.columns(2)
            for col, lbl, val, card in [
                (pp_cols2[0], "Luminos Price",  f"${lum_price:.2f}", "kpi-card-orange"),
                (pp_cols2[1], "Suave Price",    f"${suave_price:.2f}", "kpi-card"),
            ]:
                kpi(col, lbl, val, "avg effective price", card)

        # Cross-channel unified callout
        st.markdown("<br>", unsafe_allow_html=True)
        callout(f"""<strong>The unified diagnosis across retail and DTC:</strong>
        Luminos has a <strong>consumer confidence and loyalty problem</strong>.
        At retail — velocity declining in {n_declining} markets, consumers choosing
        Dove or Suave at shelf. Online (Luminos.com) — {m1_ret:.0%} Month-1 retention,
        flat cohort curves, 45% of PDP drop-offs are decision friction.
        Same root cause, two data sources confirming it.
        The fix: stronger value messaging + increased promotional investment.
        → Price elasticity –3.1 means promotions have real impact.""", 'green')


# TAB 3 — A/B TEST
# ══════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("## A/B Test Results")
    st.info("Note: All data in this project is synthetically generated and calibrated to realistic personal care DTC and CPG retail benchmarks. No real customer or sales data is used.")
    callout("""<strong>Experiment:</strong> luminos_pdp_value_messaging_q1_2024 —
    90-day RCT testing value messaging + ingredient-benefit features on Luminos.com PDP.
    Jan 8 – Apr 7, 2024 · 50/50 split · 4,000 per group.""")

    ctrl = exp[exp['variant']=='control'].copy()
    trtm = exp[exp['variant']=='treatment'].copy()
    ctrl_pdp = ctrl[ctrl['reached_pdp']==True]
    trtm_pdp = trtm[trtm['reached_pdp']==True]

    pc,pt,p_atc,sig_atc = two_prop_z(
        len(ctrl_pdp),ctrl_pdp['reached_atc'].sum(),
        len(trtm_pdp),trtm_pdp['reached_atc'].sum()
    )
    lift_atc = (pt-pc)/pc

    ctrl_aov = ctrl.loc[ctrl['purchased']==True,'order_revenue'].dropna()
    trtm_aov = trtm.loc[trtm['purchased']==True,'order_revenue'].dropna()
    _,p_aov  = stats.ttest_ind(ctrl_aov, trtm_aov)
    lift_aov = (trtm_aov.mean()-ctrl_aov.mean())/ctrl_aov.mean()
    sig_aov  = p_aov < 0.05

    ctrl_atc_s = ctrl[ctrl['reached_atc']==True]
    trtm_atc_s = trtm[trtm['reached_atc']==True]
    pc2,pt2,p_chk,sig_chk = two_prop_z(
        len(ctrl_atc_s),ctrl_atc_s['reached_checkout'].sum(),
        len(trtm_atc_s),trtm_atc_s['reached_checkout'].sum()
    )
    lift_chk = (pt2-pc2)/pc2

    ctrl_ret = ctrl.loc[ctrl['purchased']==True,'is_returned'].dropna()
    trtm_ret = trtm.loc[trtm['purchased']==True,'is_returned'].dropna()
    lift_ret     = (trtm_ret.mean()-ctrl_ret.mean())/ctrl_ret.mean()
    guardrail_ok = lift_ret <= 0.10

    ctrl_rev_pu = ctrl.loc[ctrl['purchased']==True,'order_revenue'].sum()/len(ctrl)
    trtm_rev_pu = trtm.loc[trtm['purchased']==True,'order_revenue'].sum()/len(trtm)
    lift_rev     = (trtm_rev_pu-ctrl_rev_pu)/ctrl_rev_pu
    guardrail_rev= lift_rev >= 0

    ship = sig_atc and lift_atc>0 and guardrail_ok and guardrail_rev

    section("Section 1 — Experiment Design")
    d1,d2 = st.columns(2)
    with d1:
        st.markdown("""<div class="finding-card">
            <div class="finding-title">Control — Original PDP</div>
            <div class="finding-body">Standard product description · No changes
            · <strong>4,000 users · 90 days</strong></div>
        </div>""", unsafe_allow_html=True)
    with d2:
        st.markdown("""<div class="finding-card">
            <div class="finding-title">Treatment — Enhanced PDP</div>
            <div class="finding-body">① Ingredient benefit callouts ② Social proof
            ③ Free shipping threshold banner · <strong>4,000 users · 90 days</strong></div>
        </div>""", unsafe_allow_html=True)

    callout("""<strong>Hypothesis:</strong> Adding ingredient value messaging and
    decision-support features will increase PDP→ATC rate by ≥10% relative
    by reducing decision friction for Luminos.com shoppers.""", 'blue')

    section("Section 2 — Results Overview")
    if ship:
        st.markdown("""<div style="background:#d4edda;border:2px solid #2ec4b6;
        border-radius:12px;padding:20px;text-align:center;margin-bottom:16px;">
            <div style="font-size:26px;font-weight:800;color:#155724;">SHIP</div>
            <div style="font-size:14px;color:#155724;margin-top:4px;">
            Primary metric significant · Guardrails passed · Recommend full rollout
            </div></div>""", unsafe_allow_html=True)
    else:
        st.markdown("""<div style="background:#fff3cd;border:2px solid #f9a825;
        border-radius:12px;padding:20px;text-align:center;margin-bottom:16px;">
            <div style="font-size:26px;font-weight:800;color:#856404;">INVESTIGATE</div>
        </div>""", unsafe_allow_html=True)

    r1,r2,r3,r4 = st.columns(4)
    for col,lbl,val,sub,dlt,badge,card in [
        (r1,"PDP→ATC Rate",f"{pt:.2%}",f"Control: {pc:.2%}",
         f"+{lift_atc:.1%} lift",sig_badge(sig_atc,p_atc),
         "kpi-card-green" if sig_atc and lift_atc>0 else "kpi-card"),
        (r2,"Avg Order Value",f"${trtm_aov.mean():.2f}",
         f"Control: ${ctrl_aov.mean():.2f}",f"+{lift_aov:.1%} lift",
         sig_badge(sig_aov,p_aov),
         "kpi-card-green" if sig_aov and lift_aov>0 else "kpi-card-orange"),
        (r3,"Checkout Rate",f"{pt2:.2%}",f"Control: {pc2:.2%}",
         f"+{lift_chk:.1%} lift",sig_badge(sig_chk,p_chk),
         "kpi-card-green" if sig_chk and lift_chk>0 else "kpi-card-orange"),
        (r4,"Return Rate",f"{trtm_ret.mean():.2%}",
         f"Control: {ctrl_ret.mean():.2%}",f"{lift_ret:+.1%} vs control",
         guardrail_badge(guardrail_ok),
         "kpi-card-green" if guardrail_ok else "kpi-card-red"),
    ]:
        with col:
            st.markdown(f"""<div class="{card}">
                <div class="kpi-label">{lbl}</div>
                <div class="kpi-value">{val}</div>
                <div style="font-size:12px;color:#6c757d;margin-bottom:3px;">{sub}</div>
                <div style="font-size:12px;font-weight:600;margin-bottom:5px;">{dlt}</div>
                {badge}
            </div>""", unsafe_allow_html=True)

    st.divider()

    section("Section 3 — Subgroup: Price Tier")
    tier_rows = []
    for tier in ['low','mid','high']:
        c = ctrl_pdp[ctrl_pdp['price_tier']==tier]
        t = trtm_pdp[trtm_pdp['price_tier']==tier]
        if len(c)==0 or len(t)==0: continue
        pc_t,pt_t,p_t,s_t = two_prop_z(
            len(c),c['reached_atc'].sum(),
            len(t),t['reached_atc'].sum()
        )
        tier_rows.append({'Tier':tier.capitalize(),
            'Control':pc_t,'Treatment':pt_t,
            'Lift':(pt_t-pc_t)/pc_t,'p_val':p_t,'sig':s_t})
    tier_df = pd.DataFrame(tier_rows)

    tl, tr = st.columns([1,1.2])
    with tl:
        ddf = tier_df.copy()
        ddf['Control']   = ddf['Control'].map('{:.2%}'.format)
        ddf['Treatment'] = ddf['Treatment'].map('{:.2%}'.format)
        ddf['Lift']      = ddf['Lift'].map('{:+.1%}'.format)
        ddf['p-value']   = tier_df['p_val'].map('{:.4f}'.format)
        ddf['Sig']       = tier_df['sig'].map(lambda x:'✓' if x else '~')
        st.dataframe(ddf[['Tier','Control','Treatment','Lift','p-value','Sig']],
                     use_container_width=True, hide_index=True)
        callout("Lift increases with price tier — validates decision friction hypothesis for bundles and value packs.",
                'green')
    with tr:
        fig_tier = px.bar(tier_df, x='Tier', y='Lift', color='Tier',
            text=tier_df['Lift'].map(lambda x:f'{x:+.1%}'),
            color_discrete_sequence=[COLORS['secondary'],
                                     COLORS['warning'],COLORS['danger']],
            title='Relative Lift by Price Tier')
        fig_tier.update_traces(textposition='outside')
        fig_tier.update_yaxes(tickformat='.0%', title='Relative Lift')
        fig_tier.update_layout(showlegend=False)
        clean_layout(fig_tier, height=300)
        st.plotly_chart(fig_tier, use_container_width=True)

    st.divider()

    section("Section 4 — Subgroup: Drop-off Reason (Mechanism Validation)")
    callout("""<strong>The most important subgroup analysis.</strong>
    If the treatment works via the hypothesized mechanism, lift should be highest
    for decision_friction and comparison_intent users — and near zero for
    impulse_faded and wrong_audience users who were never interested.""", 'blue')

    reason_rows = []
    for reason in ['decision_friction','comparison_intent','price_barrier',
                   'price_shock','impulse_faded','distraction',
                   'wrong_audience','ad_pdp_mismatch']:
        c = ctrl_pdp[ctrl_pdp['dropoff_reason']==reason]
        t = trtm_pdp[trtm_pdp['dropoff_reason']==reason]
        if len(c)<20 or len(t)<20: continue
        pc_r,pt_r,p_r,s_r = two_prop_z(
            len(c),c['reached_atc'].sum(),
            len(t),t['reached_atc'].sum()
        )
        reason_rows.append({'reason':reason,'n_ctrl':len(c),
            'ctrl':pc_r,'trtm':pt_r,
            'lift':(pt_r-pc_r)/pc_r,'p_val':p_r,'sig':s_r})

    reason_df = pd.DataFrame(reason_rows).sort_values('lift',ascending=False)

    rr1, rr2 = st.columns(2)
    with rr1:
        rd = reason_df.sort_values('lift', ascending=True)
        bar_colors_r = [
            '#2ec4b6' if v>=0.20 else
            '#4361ee' if v>=0.10 else
            '#f9a825' if v>=0.03 else '#e63946'
            for v in rd['lift']
        ]
        fig_r = go.Figure(go.Bar(
            x=rd['lift']*100, y=rd['reason'],
            orientation='h', marker_color=bar_colors_r, opacity=0.85,
            text=[f'{v:+.1%}{"✓" if s else ""}' for v,s in zip(rd['lift'],rd['sig'])],
            textposition='outside',
        ))
        fig_r.add_vline(x=0, line_color='#212529', line_width=1)
        fig_r.update_layout(
            title='Relative Lift by Drop-off Reason',
            xaxis=dict(title='Relative Lift (%)',
                       range=[rd['lift'].min()*130,
                               rd['lift'].max()*140]),
        )
        clean_layout(fig_r, height=360)
        st.plotly_chart(fig_r, use_container_width=True)

    with rr2:
        top5 = reason_df[reason_df['reason'].isin([
            'decision_friction','comparison_intent',
            'price_barrier','impulse_faded','wrong_audience'
        ])].copy()
        fig_r2 = go.Figure()
        fig_r2.add_trace(go.Bar(
            name='Control', x=top5['reason'],
            y=top5['ctrl']*100,
            marker_color=COLORS['muted'], opacity=0.80,
        ))
        fig_r2.add_trace(go.Bar(
            name='Treatment', x=top5['reason'],
            y=top5['trtm']*100,
            marker_color=COLORS['primary'], opacity=0.80,
            text=[f'{l:+.0%}{"✓" if s else ""}'
                  for l,s in zip(top5['lift'],top5['sig'])],
            textposition='outside',
        ))
        fig_r2.update_layout(
            barmode='group', title='ATC Rate by Drop-off Reason',
            yaxis_title='ATC Rate (%)',
            xaxis=dict(tickangle=-20),
            legend=dict(orientation='h',y=1.1),
        )
        clean_layout(fig_r2, height=360)
        st.plotly_chart(fig_r2, use_container_width=True)

    df_lift = reason_df[reason_df['reason']=='decision_friction']['lift'].values
    ci_lift = reason_df[reason_df['reason']=='comparison_intent']['lift'].values
    pb_lift = reason_df[reason_df['reason']=='price_barrier']['lift'].values
    if_lift = reason_df[reason_df['reason']=='impulse_faded']['lift'].values

    if len(df_lift) and len(ci_lift) and len(pb_lift) and len(if_lift):
        pattern = df_lift[0]>ci_lift[0]>pb_lift[0]>if_lift[0]
        callout(f"""<strong>Mechanism confirmed: {'✓ YES' if pattern else '~ Partial'}.</strong>
        decision_friction: {df_lift[0]:+.1%} →
        comparison_intent: {ci_lift[0]:+.1%} →
        price_barrier: {pb_lift[0]:+.1%} →
        impulse_faded: {if_lift[0]:+.1%}.
        Treatment helps users who were genuinely interested but lacked confidence.
        It does NOT help users who were never interested.""", 'green')

    st.divider()

    section("Section 5 — Weekly Lift Stability")

    exp_pdp = exp[exp['reached_pdp']==True].copy()
    exp_pdp['week_num'] = (
        (exp_pdp['session_start']-EXP_START).dt.days//7+1
    ).clip(1,13)

    weekly_rows = []
    for week in sorted(exp_pdp['week_num'].unique()):
        wd = exp_pdp[exp_pdp['week_num']==week]
        cw = wd[wd['variant']=='control']
        tw = wd[wd['variant']=='treatment']
        if len(cw)<10 or len(tw)<10: continue
        pcw,ptw,pw,sw = two_prop_z(
            len(cw),cw['reached_atc'].sum(),
            len(tw),tw['reached_atc'].sum()
        )
        weekly_rows.append({'week':int(week),'ctrl':pcw,'trtm':ptw,
            'lift':(ptw-pcw)/pcw if pcw>0 else 0,'sig':sw})
    wdf   = pd.DataFrame(weekly_rows)
    early = wdf[wdf['week']<=4]['lift'].mean()
    late  = wdf[wdf['week']>8]['lift'].mean()
    avg_w = wdf['lift'].mean()

    w1,w2 = st.columns(2)
    with w1:
        fig_w1 = go.Figure()
        fig_w1.add_trace(go.Scatter(x=wdf['week'],y=wdf['ctrl']*100,
            mode='lines+markers',name='Control',
            line=dict(color=COLORS['muted'],width=2),marker=dict(size=6)))
        fig_w1.add_trace(go.Scatter(x=wdf['week'],y=wdf['trtm']*100,
            mode='lines+markers',name='Treatment',
            line=dict(color=COLORS['primary'],width=2),marker=dict(size=6),
            fill='tonexty',fillcolor='rgba(67,97,238,0.08)'))
        fig_w1.update_layout(title='CVR by Variant — Weekly',
            xaxis_title='Week',yaxis_title='PDP→ATC Rate (%)',
            legend=dict(orientation='h',y=1.1))
        clean_layout(fig_w1,height=300)
        st.plotly_chart(fig_w1,use_container_width=True)

    with w2:
        fig_w2 = go.Figure(go.Bar(
            x=wdf['week'],y=wdf['lift']*100,
            marker_color=[COLORS['secondary'] if s else COLORS['warning']
                          for s in wdf['sig']],
            opacity=0.85,width=0.65))
        fig_w2.add_hline(y=avg_w*100,line_dash='dash',
            line_color=COLORS['danger'],
            annotation_text=f'Avg: {avg_w:.2%}',
            annotation_position='right')
        fig_w2.update_layout(title='Weekly Relative Lift',
            xaxis_title='Week',yaxis_title='Relative Lift (%)')
        clean_layout(fig_w2,height=300)
        st.plotly_chart(fig_w2,use_container_width=True)

    novelty = late > early*0.85
    callout(f"""{'No novelty effect detected' if novelty else 'Possible novelty effect'}.
    Early (wks 1-4): {early:+.2%} · Late (wks 9-13): {late:+.2%}.
    {'Lift is stable — genuine behavior change.' if novelty else 'Monitor post-launch for 30 days.'}""",
    'green' if novelty else '')

    st.divider()

    section("Section 6 — Business Impact")
    monthly_pdp = pre[pre['reached_pdp']==True].shape[0] / 9
    incr_atc    = monthly_pdp * (pt-pc)
    ctrl_chk_r  = ctrl_pdp['reached_checkout'].mean()
    ctrl_pur_r  = ctrl[ctrl['reached_checkout']==True]['purchased'].mean()
    incr_orders = incr_atc * ctrl_chk_r * ctrl_pur_r
    incr_rev    = incr_orders * trtm_aov.mean()

    b1,b2,b3 = st.columns(3)
    for col,lbl,val,sub in [
        (b1,"Incremental Orders/Month",f"{incr_orders:,.0f}",
         f"From {monthly_pdp*pc:,.0f} → {monthly_pdp*pt:,.0f}"),
        (b2,"Incremental Revenue/Month",f"${incr_rev:,.0f}",
         f"At ${trtm_aov.mean():.2f} avg order value"),
        (b3,"Annualised Revenue Impact",f"${incr_rev*12:,.0f}",
         "If lift holds across full traffic"),
    ]:
        kpi(col, lbl, val, sub, 'kpi-card-green')



# ══════════════════════════════════════════════════════════════════════
# TAB 4 — CHURN  (unchanged from v11)
# ══════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════
# TAB 4 — CHURN  (v13 — slimmed to 4 sections)
# ══════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("## Churn Analysis")
    st.info("Note: All data in this project is synthetically generated and calibrated to realistic personal care DTC and CPG retail benchmarks. No real customer or sales data is used.")
    callout("""<strong>Purpose:</strong> Score Luminos.com users for churn risk
    using behavioral and transactional features from April–October 2023.
    Churn is defined as no activity in the subsequent 60-day window (October 2023–January 2024).
    Note: the 77% overall churn rate reflects a broad definition that includes
    one-time visitors and non-purchasers — among users who completed at least
    one purchase, the churn rate is materially lower.
    <br><br>
    Three approaches were used.
    <strong>Logistic Regression</strong> (AUC 0.915): interpretable baseline —
    purchase frequency and session engagement are the strongest protective factors;
    days since last visit is the strongest risk factor.
    <strong>XGBoost</strong> (AUC 0.916): gradient-boosted ensemble —
    near-identical AUC to logistic regression, confirming the signal is largely
    linear and feature engineering matters more than model complexity.
    <strong>BG/NBD</strong>: probabilistic model for purchasers that estimates
    P(alive) without requiring a churn label; output feeds into CLV prediction.""", 'blue')

    pre_events_ch = events[
        (events['event_timestamp']>=OBS_START) &
        (events['event_timestamp']<OBS_CUTOFF)
    ].copy()
    label_events_ch = events[
        (events['event_timestamp']>=OBS_CUTOFF) &
        (events['event_timestamp']<EXP_START)
    ].copy()
    obs_users    = set(pre_events_ch['user_pseudo_id'].unique())
    active_users = set(label_events_ch['user_pseudo_id'].unique())
    churn_labels_dict = {u: 0 if u in active_users else 1 for u in obs_users}

    churn_labeled = churn_scores.copy()
    churn_labeled['churned'] = churn_labeled['user_pseudo_id'].map(
        churn_labels_dict).fillna(1)

    overall_churn = churn_labeled['churned'].mean()
    total_users   = len(churn_scores)
    high_risk     = (churn_scores['churn_risk_tier']=='High Risk').sum()
    medium_risk   = (churn_scores['churn_risk_tier']=='Medium Risk').sum()
    low_risk      = (churn_scores['churn_risk_tier']=='Low Risk').sum()

    # ── Section 1: Overview ───────────────────────────────────────────
    section("Section 1 — Churn Overview")

    cv1,cv2,cv3,cv4 = st.columns(4)
    for col,lbl,val,dlt,card in [
        (cv1,"Users Analyzed",    f"{total_users:,}","observation window","kpi-card"),
        (cv2,"Overall Churn Rate",f"{overall_churn:.1%}","no return in 60 days","kpi-card-red"),
        (cv3,"High Risk Users",   f"{high_risk:,}",f"{high_risk/total_users:.1%} of users","kpi-card-red"),
        (cv4,"Low Risk Users",    f"{low_risk:,}",f"{low_risk/total_users:.1%} of users","kpi-card-green"),
    ]:
        kpi(col, lbl, val, dlt, card)

    callout("""<strong>Churn Definition:</strong> Active Apr–Oct 2023 but
    <strong>no activity in Oct–Jan 2024</strong>.
    Features from observation window only — no data leakage.""")

    ch1, ch2 = st.columns([1, 1.4])

    with ch1:
        fig_donut = go.Figure(go.Pie(
            labels=['High Risk','Medium Risk','Low Risk'],
            values=[high_risk,medium_risk,low_risk], hole=0.55,
            marker=dict(colors=[COLORS['danger'],COLORS['warning'],COLORS['secondary']]),
            textinfo='label+percent', textfont=dict(size=13),
        ))
        fig_donut.update_layout(title='Churn Risk Distribution',
            annotations=[dict(text=f"{overall_churn:.0%}<br>churn rate",
                x=0.5,y=0.5,font_size=16,showarrow=False,
                font=dict(color='#212529',weight='bold'))])
        clean_layout(fig_donut, height=320)
        st.plotly_chart(fig_donut, use_container_width=True)

    with ch2:
        # Days since last visit — single histogram, churned vs retained
        churned_days  = churn_labeled[churn_labeled['churned']==1]['days_since_last_visit'].dropna()
        retained_days = churn_labeled[churn_labeled['churned']==0]['days_since_last_visit'].dropna()

        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=churned_days, name='Churned',
            nbinsx=40, marker_color=COLORS['danger'],
            opacity=0.70, histnorm='percent'))
        fig_hist.add_trace(go.Histogram(
            x=retained_days, name='Retained',
            nbinsx=40, marker_color=COLORS['secondary'],
            opacity=0.70, histnorm='percent'))
        fig_hist.update_layout(
            barmode='overlay',
            title='Days Since Last Visit — Churned vs Retained',
            xaxis_title='Days Since Last Visit',
            yaxis_title='% of Users',
            legend=dict(orientation='h', y=1.1))
        clean_layout(fig_hist, height=320)
        st.plotly_chart(fig_hist, use_container_width=True)

    callout(f"""<strong>Key signal:</strong> Churned users median
    <strong>{churned_days.median():.0f} days</strong> since last visit vs retained at
    <strong>{retained_days.median():.0f} days</strong>.
    Flag users inactive <strong>30+ days</strong> for win-back —
    {(1-((churned_days<=30).mean())):.0%} of eventual churners are already past this threshold.""", 'blue')

    st.divider()

    # ── Section 2: Feature Importance ────────────────────────────────
    section("Section 2 — What Drives Churn?")
    st.markdown("""XGBoost feature importance reveals the strongest predictors.
    Engagement frequency and recency dominate — behavioral signals like
    scroll depth and drop-off reason add incremental lift.""")

    xgb_f = pd.DataFrame({
        'Feature'   :['avg_sessions_per_week','days_since_last_visit',
                      'total_sessions','total_orders','total_revenue',
                      'avg_order_value','dropoff_reason_score',
                      'pdp_per_session','avg_scroll_depth',
                      'intent_level_score'],
        'Importance':[0.3836,0.2329,0.0605,0.0512,0.0398,
                      0.0287,0.0090,0.0085,0.0081,0.0073],
    }).sort_values('Importance', ascending=True)

    new_feats  = ['avg_scroll_depth','intent_level_score','dropoff_reason_score']
    fi_colors  = [
        COLORS['danger']  if f in ['avg_sessions_per_week','days_since_last_visit'] else
        COLORS['warning'] if f in new_feats else COLORS['primary']
        for f in xgb_f['Feature']
    ]
    fig_fi = go.Figure(go.Bar(
        x=xgb_f['Importance'], y=xgb_f['Feature'],
        orientation='h', marker_color=fi_colors, opacity=0.85,
        text=[f'{v:.2%}' for v in xgb_f['Importance']],
        textposition='outside'))
    fig_fi.update_layout(
        title='XGBoost Feature Importance (AUC 0.92)<br>'
              '<sup>Red = top signals · Orange = new behavioral features · Blue = transactional</sup>',
        xaxis=dict(title='Importance', range=[0, 0.48]))
    clean_layout(fig_fi, height=360)
    st.plotly_chart(fig_fi, use_container_width=True)

    callout("""<strong>Top drivers:</strong>
    ① <strong>avg_sessions_per_week</strong> (38%) — engagement frequency is the strongest predictor.
    ② <strong>days_since_last_visit</strong> (23%) — recency signal.
    ③ New behavioral features (scroll depth, drop-off reason, intent level) add 2.4% combined —
    small but meaningful lift over standard RFM features alone.""")

    st.divider()

    # ── Section 3: Users by Risk Tier ────────────────────────────────
    section("Section 3 — Users by Risk Tier")

    def make_risk_table(tier_label, n=10):
        subset = churn_scores[churn_scores['churn_risk_tier']==tier_label].copy()
        if tier_label=='High Risk':   subset = subset.nlargest(n,'xgb_churn_prob')
        elif tier_label=='Low Risk':  subset = subset.nsmallest(n,'xgb_churn_prob')
        else:                         subset = subset.sample(min(n,len(subset)),random_state=42)
        cols = ['user_pseudo_id','xgb_churn_prob','churn_risk_tier',
                'total_sessions','total_orders','total_revenue',
                'days_since_last_visit','favorite_device','favorite_source']
        subset = subset[[c for c in cols if c in subset.columns]].copy()
        subset['user_pseudo_id'] = [f"User_{i+1:03d}" for i in range(len(subset))]
        subset['xgb_churn_prob'] = subset['xgb_churn_prob'].map('{:.1%}'.format)
        if 'total_revenue' in subset.columns:
            subset['total_revenue'] = subset['total_revenue'].map('${:.0f}'.format)
        subset = subset.rename(columns={
            'user_pseudo_id':'User','xgb_churn_prob':'Churn Prob',
            'churn_risk_tier':'Risk Tier','total_sessions':'Sessions',
            'total_orders':'Orders','total_revenue':'Revenue',
            'days_since_last_visit':'Days Since Visit',
            'favorite_device':'Device','favorite_source':'Source'})
        return subset

    rt1,rt2,rt3 = st.tabs(["High Risk","Medium Risk","Low Risk"])
    with rt1:
        st.caption(f"{high_risk:,} total high-risk users ({high_risk/total_users:.1%})")
        st.dataframe(make_risk_table('High Risk'),use_container_width=True,hide_index=True)
    with rt2:
        st.caption(f"{medium_risk:,} total medium-risk users ({medium_risk/total_users:.1%})")
        st.dataframe(make_risk_table('Medium Risk'),use_container_width=True,hide_index=True)
    with rt3:
        st.caption(f"{low_risk:,} total low-risk users ({low_risk/total_users:.1%})")
        st.dataframe(make_risk_table('Low Risk'),use_container_width=True,hide_index=True)

    st.divider()

    # ── Section 4: Recommendations ───────────────────────────────────
    section("Section 4 — Retention Recommendations")

    rc1,rc2,rc3 = st.columns(3)
    for col,title,body in [
        (rc1,"High Risk — Win-back",
         f"<strong>{high_risk:,} users</strong><br>Win-back email within 7 days.<br>"
         f"Personalised offer on viewed Luminos SKUs.<br>Non-buyers, 60+ days inactive."),
        (rc2,"Medium Risk — Re-engagement",
         f"<strong>{medium_risk:,} users</strong><br>Browse-abandonment + value messaging.<br>"
         f"First purchase converts to low-risk.<br>Channel: email (highest CVR)."),
        (rc3,"Low Risk — Loyalty",
         f"<strong>{low_risk:,} users</strong><br>Subscription discount, early access.<br>"
         f"Goal: increase repurchase frequency and AOV."),
    ]:
        with col:
            st.markdown(f"""<div class="finding-card">
                <div class="finding-title">{title}</div>
                <div class="finding-body">{body}</div>
            </div>""", unsafe_allow_html=True)



# ══════════════════════════════════════════════════════════════════════
# TAB 5 — RETENTION  (v13 — slimmed to 4 sections)
# ══════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown("## Retention Modeling")
    st.info("Note: All data in this project is synthetically generated and calibrated to realistic personal care DTC and CPG retail benchmarks. No real customer or sales data is used.")
    callout("""<strong>Purpose:</strong> Building on churn scores from the previous tab,
    this analysis answers: what is each Luminos customer worth,
    and where should retention investment be prioritized?
    <br><br>
    <strong>RFM Segmentation:</strong> every user scored 1–4 on Recency, Frequency,
    and Monetary value and assigned to a named segment that maps directly to a CRM action.
    <strong>CLV Prediction (6-month horizon):</strong> BG/NBD predicted purchase frequency
    combined with Gamma-Gamma predicted AOV and a gross margin assumption —
    identifies which customers will generate the most value over the next 6 months.
    <strong>Purchase Propensity (30-day):</strong> logistic regression scoring each
    user's probability of purchasing in the next 30 days, used to rank users
    for campaign targeting.""", 'blue')

    @st.cache_data
    def build_retention_data():
        ret = pd.read_csv('data/retention_master.csv')
        seg_summary = (
            ret.groupby('segment')
            .agg(users=('user_pseudo_id','count'),
                 avg_clv=('clv_6month','mean'),
                 total_clv=('clv_6month','sum'),
                 avg_monetary=('monetary','mean'))
            .reset_index()
            .sort_values('total_clv', ascending=False)
        )
        clv_data  = ret[ret['clv_6month']>0].copy()
        prop_data = ret[ret['propensity_score'].notna()].copy()
        return ret, seg_summary, clv_data, prop_data

    ret_master, seg_summary, clv_data, prop_data = build_retention_data()

    total_ret  = len(ret_master)
    total_clv  = ret_master['clv_6month'].sum()
    avg_clv    = ret_master[ret_master['clv_6month']>0]['clv_6month'].mean()
    high_prop  = (ret_master['propensity_tier']=='High Propensity').sum()

    # ── Section 1: Overview ───────────────────────────────────────────
    section("Section 1 — Overview")

    ov1,ov2,ov3,ov4 = st.columns(4)
    for col,lbl,val,dlt,card in [
        (ov1,"Users Modeled",        f"{total_ret:,}","retention master","kpi-card"),
        (ov2,"Total Predicted CLV",  f"${total_clv:,.0f}","6-month horizon","kpi-card-green"),
        (ov3,"Avg CLV (Purchasers)", f"${avg_clv:.0f}","per buying customer","kpi-card-green"),
        (ov4,"High Propensity Users",f"{high_prop:,}",
         f"{high_prop/total_ret:.1%} likely to buy in 30d","kpi-card-orange"),
    ]:
        kpi(col,lbl,val,dlt,card)

    st.divider()

    # ── Section 2: RFM Action Table ───────────────────────────────────
    section("️ Section 2 — RFM Segments & Actions")
    st.markdown("""Every Luminos customer scored 1–4 on Recency, Frequency, and Monetary value.
    The table below is the CRM team's playbook — segment, size, predicted value, and action.""")

    action_map = {
        'Champion'               :('4-4-4','VIP rewards, early access, ask for reviews'),
        'Loyal Customer'         :('3-4-x','Loyalty program, upsell, cross-sell'),
        'New Customer'           :('4-1-x','Onboarding sequence, nudge first repeat purchase'),
        'Potential Loyalist'     :('3-2-x','Targeted emails, encourage 2nd purchase'),
        'At Risk'                :('2-3-x','Win-back before they lapse'),
        'Hibernating (High Value)':('1-x-4','High-value win-back — personalised discount'),
        'Hibernating'            :('1-2-x','Re-engagement discount or accept as churned'),
        'Cannot Lose Them'       :('1-4-x','Urgent win-back — high frequency gone quiet'),
        'Lost / Non-Buyer'       :('1-1-1','Minimal investment — accept as churned'),
    }
    action_rows = []
    for _, row in seg_summary.iterrows():
        seg  = row['segment']
        info = action_map.get(seg, ('—','—'))
        action_rows.append({
            'Segment'           : seg,
            'Users'             : f"{int(row['users']):,}",
            'Avg 6mo CLV'       : f"${row['avg_clv']:.0f}",
            'Total CLV'         : f"${row['total_clv']:,.0f}",
            'RFM Profile'       : info[0],
            'Recommended Action': info[1],
        })
    st.dataframe(pd.DataFrame(action_rows), use_container_width=True, hide_index=True)

    hibernating_hv = (ret_master['segment']=='Hibernating (High Value)').sum()
    callout(f"""<strong>Hibernating (High Value)</strong> — {hibernating_hv:,} users —
    is the highest ROI win-back target. They demonstrated strong purchasing capacity
    but have gone quiet. Re-acquiring them costs far less than acquiring a new
    customer of equivalent value.""")

    st.divider()

    # ── Section 3: CLV × Propensity ───────────────────────────────────
    section("Section 3 — CLV × Propensity: The Full Customer Picture")
    st.markdown("""**Top-right = highest priority:** high CLV and high propensity to buy soon.
    **Top-left = Hibernating High Value:** worth a lot but gone quiet — urgent win-back.
    **Bottom = low priority:** low value, low likelihood to buy.""")

    sc1, sc2 = st.columns([1.6, 1])

    with sc1:
        scatter_data = ret_master[
            (ret_master['clv_6month']>0) &
            (ret_master['propensity_score'].notna())
        ].sample(min(800, len(ret_master)), random_state=42)

        fig_scatter = px.scatter(
            scatter_data, x='propensity_score', y='clv_6month',
            color='segment', size='clv_6month', size_max=18, opacity=0.60,
            title='CLV vs Purchase Propensity by RFM Segment',
            labels={'propensity_score':'Purchase Propensity (30-day)',
                    'clv_6month':'6-Month CLV ($)', 'segment':'RFM Segment'},
            color_discrete_sequence=px.colors.qualitative.Bold,
        )
        fig_scatter.add_vline(
            x=0.60, line_dash='dash', line_color=COLORS['muted'],
            annotation_text='High propensity')
        fig_scatter.add_hline(
            y=scatter_data['clv_6month'].median(), line_dash='dash',
            line_color=COLORS['muted'], annotation_text='Median CLV')
        fig_scatter.update_layout(
            xaxis=dict(tickformat='.0%'),
            legend=dict(orientation='h', y=-0.25, font=dict(size=9)))
        clean_layout(fig_scatter, height=420)
        st.plotly_chart(fig_scatter, use_container_width=True)

    with sc2:
        clv_vals = clv_data['clv_6month']
        fig_clv  = go.Figure(go.Histogram(
            x=clv_vals, nbinsx=40,
            marker_color=COLORS['primary'], opacity=0.80))
        fig_clv.add_vline(x=clv_vals.mean(), line_dash='dash',
            line_color=COLORS['danger'],
            annotation_text=f"Mean: ${clv_vals.mean():.0f}")
        fig_clv.add_vline(x=clv_vals.median(), line_dash='dash',
            line_color=COLORS['warning'],
            annotation_text=f"Median: ${clv_vals.median():.0f}")
        fig_clv.update_layout(
            title='6-Month CLV Distribution',
            xaxis_title='CLV ($)', yaxis_title='Users', showlegend=False)
        clean_layout(fig_clv, height=280)
        st.plotly_chart(fig_clv, use_container_width=True)

        # Propensity by segment bar
        prop_seg = (
            prop_data.groupby('segment')['propensity_score']
            .mean().reset_index()
            .rename(columns={'propensity_score':'avg_prop'})
            .sort_values('avg_prop', ascending=True)
        )
        prop_colors = [
            COLORS['danger']  if v>=0.60 else
            COLORS['warning'] if v>=0.25 else COLORS['muted']
            for v in prop_seg['avg_prop']
        ]
        fig_ps = go.Figure(go.Bar(
            x=prop_seg['avg_prop']*100, y=prop_seg['segment'],
            orientation='h', marker_color=prop_colors, opacity=0.85,
            text=[f'{v:.1%}' for v in prop_seg['avg_prop']],
            textposition='outside'))
        fig_ps.update_layout(
            title='Avg Purchase Propensity by Segment',
            xaxis=dict(title='Propensity (%)',
                       range=[0, prop_seg['avg_prop'].max()*140]))
        clean_layout(fig_ps, height=280)
        st.plotly_chart(fig_ps, use_container_width=True)

    callout(f"""<strong>Campaign targeting:</strong> Send next Luminos campaign to
    <strong>{high_prop:,} high-propensity users</strong> ({high_prop/total_ret:.1%} of all users)
    — most likely to repurchase in the next 30 days.
    Cross-reference with CLV: prioritise high-propensity + high-CLV users first.""", 'green')

    st.divider()

    # ── Section 4: Recommendations ───────────────────────────────────
    section("Section 4 — Retention Recommendations")

    champions_n  = (ret_master['segment']=='Champion').sum()
    potential_n  = (ret_master['segment']=='Potential Loyalist').sum()
    lost_n       = (ret_master['segment']=='Lost / Non-Buyer').sum()

    rr1,rr2,rr3,rr4 = st.columns(4)
    for col,title,body in [
        (rr1,"Champions",
         f"<strong>{champions_n:,} users</strong><br>VIP rewards + early access.<br>"
         f"Ask for reviews and referrals.<br>Protect at all costs."),
        (rr2,"Hibernating — High Value",
         f"<strong>{hibernating_hv:,} users</strong><br>Personalised win-back discount.<br>"
         f"Reference past Luminos purchase.<br>Highest ROI re-engagement."),
        (rr3,"Potential Loyalists",
         f"<strong>{potential_n:,} users</strong><br>Encourage 2nd purchase.<br>"
         f"Show PDP value messaging (A/B treatment).<br>First repeat = long-term retention."),
        (rr4,"Lost / Non-Buyers",
         f"<strong>{lost_n:,} users</strong><br>Minimal budget — low ROI.<br>"
         f"Awareness only if budget allows.<br>Accept as churned."),
    ]:
        with col:
            st.markdown(f"""<div class="finding-card">
                <div class="finding-title">{title}</div>
                <div class="finding-body">{body}</div>
            </div>""", unsafe_allow_html=True)