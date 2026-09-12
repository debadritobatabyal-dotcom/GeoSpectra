import os
import json
import logging
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import theme
from i18n import t

from .production_predict import production_predict, load_production_model_bundle, RISK_THRESHOLDS
from .scenarios import simulate_scenario, list_scenarios, SCENARIOS
from .recovery import simulate_recovery_scenario, list_recovery_scenarios, RECOVERY_SCENARIOS
from .recommendations import generate_recommendations

LOGGER = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "production", "synthetic_production_dataset_fixed.csv")
PRIMARY_METRICS_PATH = os.path.join(BASE_DIR, "models", "production", "production_metrics.json")
FALLBACK_METRICS_PATH = os.path.join(BASE_DIR, "models", "production_metrics.json")

@st.cache_data(show_spinner=False)
def load_production_dataframe() -> pd.DataFrame:
    if not os.path.exists(DATA_PATH):
        return pd.DataFrame()
    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df

@st.cache_data(show_spinner=False)
def load_model_metrics() -> dict:
    for p in [PRIMARY_METRICS_PATH, FALLBACK_METRICS_PATH]:
        if os.path.exists(p):
            try:
                with open(p) as f:
                    return json.load(f)
            except Exception:
                pass
    return {}

def render_html(html_str: str):
    lines = [line.strip() for line in html_str.splitlines()]
    cleaned = "\n".join(line for line in lines if line)
    st.markdown(cleaned, unsafe_allow_html=True)

def render_production_tab():
    curr_theme = theme.get_active_theme()
    T = theme.get_theme_tokens(curr_theme)
    df_prod = load_production_dataframe()
    metrics = load_model_metrics()

    if df_prod.empty:
        st.error("Production dataset not found. Please ensure data/production/synthetic_production_dataset_fixed.csv exists.")
        return

    all_mines = sorted(df_prod["mine_id"].unique())
    col_c1, col_c2, col_c3 = st.columns([1.1, 1.6, 1.3])

    with col_c1:
        selected_mine = st.selectbox(
            t("monitored_pit_site"),
            options=all_mines,
            index=0,
            help="Select one of the 8 monitored mining operations across the belt."
        )

    mine_df = df_prod[df_prod["mine_id"] == selected_mine].sort_values("date").reset_index(drop=True)
    holdout_2024 = mine_df[mine_df["date"] >= "2024-01-01"]
    display_df = holdout_2024 if not holdout_2024.empty else mine_df
    available_dates = display_df["date"].dt.strftime("%Y-%m-%d").tolist()

    with col_c2:
        default_idx = min(len(available_dates) - 1, max(0, 120))
        selected_date_str = st.selectbox(
            t("operational_shift"),
            options=available_dates,
            index=default_idx,
            help="Select an unseen operational shift from the forward 2024 holdout evaluation period."
        )

    snapshot_row = display_df[display_df["date"].dt.strftime("%Y-%m-%d") == selected_date_str].iloc[0].to_dict()

    with col_c3:
        target_override = st.number_input(
            t("planned_shift_quota"),
            min_value=500.0,
            max_value=6000.0,
            value=float(snapshot_row.get("planned_production_tpd", 2400.0)),
            step=50.0,
            help="Planned daily mining quota to evaluate target variance and shortfall against."
        )

    current_inputs = dict(snapshot_row)
    current_inputs["planned_production_tpd"] = target_override

    bundle = load_production_model_bundle()
    baseline_res = production_predict(current_inputs, bundle=bundle, include_explain=True)

    expected_tpd = baseline_res["expected_production_tpd"]
    target_tpd = baseline_res["planned_production_tpd"]
    target_variance_tpd = baseline_res["target_variance_tpd"]
    shortfall_tonnes = baseline_res["shortfall_tonnes"]
    shortfall_pct = baseline_res["shortfall_pct"]
    on_target = baseline_res["on_target"]
    risk_level = baseline_res["risk_level"]
    risk_color = baseline_res["risk_color"]
    risk_summary = baseline_res["risk_summary"]
    shap_drivers = baseline_res["shap_drivers"]
    recommendations = baseline_res["recommendations"]

    shift_dt = pd.to_datetime(selected_date_str)
    formatted_date = shift_dt.strftime("%d %b %Y").upper()

    render_html(f"""
    <div style="padding: 24px 0 16px 0; border-bottom: 1px solid var(--border); margin-bottom: 28px;">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 16px;">
            <div>
                <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 6px;">
                    <span style="font-size: 0.95rem; font-weight: 800; color: var(--accent-copper);">05</span>
                    <span style="font-size: 0.72rem; font-weight: 800; letter-spacing: 2px; text-transform: uppercase; color: var(--accent-blue);">
                        {t("prod_kicker")}
                    </span>
                </div>
                <div style="font-size: 2.8rem; font-weight: 800; color: var(--text-primary); line-height: 1.05; letter-spacing: -1.5px; margin-bottom: 6px;">
                    {t("prod_subtitle")}
                </div>
                <div style="font-size: 0.95rem; font-weight: 400; color: var(--text-muted); max-width: 720px;">
                    {t("prod_desc")}
                </div>
            </div>
            <div style="text-align: right; border-left: 2px solid var(--accent-copper); padding-left: 20px;">
                <div style="font-size: 1.15rem; font-weight: 800; color: var(--text-primary); letter-spacing: -0.5px;">{selected_mine}</div>
                <div style="font-size: 0.78rem; font-weight: 600; color: var(--text-muted); margin-top: 2px;">{formatted_date}</div>
                <div style="display: inline-block; margin-top: 8px; padding: 2px 10px; background: var(--surface); border: 1px solid var(--border); border-radius: 4px; font-size: 0.65rem; font-weight: 800; color: var(--text-muted); letter-spacing: 1px;">
                    {t("demo_synthetic_badge")}
                </div>
            </div>
        </div>
    </div>
    """)

    col_kpi_dom, col_kpi_status = st.columns([1.35, 1.0], gap="large")

    with col_kpi_dom:
        render_html(f"""
        <div style="padding: 10px 0;">
            <div style="font-size: 0.75rem; font-weight: 800; letter-spacing: 1.5px; text-transform: uppercase; color: var(--accent-copper); margin-bottom: 4px;">
                {t("prod_kpi_expected")}
            </div>
            <div style="display: flex; align-items: baseline; gap: 12px; margin-bottom: 8px;">
                <span style="font-size: 4.8rem; font-weight: 800; color: var(--text-primary); line-height: 0.95; letter-spacing: -2.5px;">
                    {expected_tpd:,.0f}
                </span>
                <span style="font-size: 1.15rem; font-weight: 700; color: var(--text-muted); text-transform: uppercase; letter-spacing: 1px;">
                    {t("prod_unit_tpd")}
                </span>
            </div>
            <div style="font-size: 0.82rem; color: var(--text-muted); margin-bottom: 16px;">
                {t("forecast_explainer")}
            </div>
            <div style="display: flex; gap: 32px; padding-top: 14px; border-top: 1px solid var(--border);">
                <div>
                    <div style="font-size: 0.68rem; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; color: var(--text-muted);">{t("prod_kpi_planned")}</div>
                    <div style="font-size: 1.25rem; font-weight: 800; color: var(--text-primary);">{target_tpd:,.0f} <span style="font-size: 0.78rem; color: var(--text-muted);">t/d</span></div>
                </div>
                <div>
                    <div style="font-size: 0.68rem; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; color: var(--text-muted);">{t("prod_kpi_variance")}</div>
                    <div style="font-size: 1.25rem; font-weight: 800; color: var(--text-primary);">
                        {target_variance_tpd:+,.0f} <span style="font-size: 0.75rem; font-weight: 700; color: {'var(--accent-green)' if on_target else 'var(--accent-red)'};">{'▲' if on_target else '▼'} t/d</span>
                    </div>
                </div>
                <div>
                    <div style="font-size: 0.68rem; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; color: var(--text-muted);">{t("shortfall_status")}</div>
                    <div style="font-size: 1.25rem; font-weight: 800; color: var(--text-primary);">
                        {shortfall_tonnes:,.0f} <span style="font-size: 0.75rem; font-weight: 700; color: {'var(--accent-green)' if on_target else 'var(--accent-amber)'};">t/d ({shortfall_pct:.1f}%)</span>
                    </div>
                </div>
            </div>
        </div>
        """)

    with col_kpi_status:

        if on_target:
            status_bg = T["risk_low_bg"]
            status_border = T["risk_low_border"]
            status_accent = T["risk_low_text"]
            status_headline = f"+{abs(target_variance_tpd):,.0f} t/day"
            status_sub = t("above_target_sub", pct=abs(target_variance_tpd) / target_tpd * 100)
            status_risk_text = t("risk_low")
        else:
            if risk_level == "HIGH":
                status_bg = T["risk_high_bg"]
                status_border = T["risk_high_border"]
                status_accent = T["risk_high_text"]
                status_risk_text = t("risk_high")
            elif risk_level == "MEDIUM":
                status_bg = T["risk_med_bg"]
                status_border = T["risk_med_border"]
                status_accent = T["risk_med_text"]
                status_risk_text = t("risk_medium")
            else:
                status_bg = T["risk_low_bg"]
                status_border = T["risk_low_border"]
                status_accent = T["risk_low_text"]
                status_risk_text = t("risk_low")

            status_headline = f"−{shortfall_tonnes:,.0f} t/day"
            status_sub = t("below_target_sub", pct=shortfall_pct)

        render_html(f"""
        <div style="background: {status_bg}; border: 1px solid {status_border}; border-radius: 12px; padding: 22px 24px; height: 100%; display: flex; flex-direction: column; justify-content: space-between; box-shadow: var(--card-shadow);">
            <div>
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <span style="font-size: 0.72rem; font-weight: 800; letter-spacing: 1.5px; text-transform: uppercase; color: {status_accent};">
                        {t("target_status")}
                    </span>
                    <span style="background: {status_accent}; color: #ffffff; padding: 2px 10px; border-radius: 9999px; font-size: 0.68rem; font-weight: 800; letter-spacing: 0.5px;">
                        {status_risk_text}
                    </span>
                </div>
                <div style="font-size: 2.8rem; font-weight: 800; color: var(--text-primary); line-height: 1.0; letter-spacing: -1px; margin: 6px 0;">
                    {status_headline}
                </div>
                <div style="font-size: 0.88rem; font-weight: 600; color: var(--text-primary); margin-bottom: 4px;">
                    {status_sub}
                </div>
                <div style="font-size: 0.75rem; color: var(--text-secondary);">
                    {risk_summary}
                </div>
            </div>
            <div style="font-size: 0.68rem; color: var(--text-muted); border-top: 1px solid var(--border); padding-top: 8px; margin-top: 12px;">
                {t("demo_thresholds_note")}
            </div>
        </div>
        """)

    render_html(f"""
    <div style="margin-top: 36px; margin-bottom: 8px;">
        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 4px;">
            <span style="font-size: 1.15rem; font-weight: 800; color: var(--accent-copper);">01</span>
            <span style="font-size: 0.72rem; font-weight: 800; letter-spacing: 1.5px; text-transform: uppercase; color: var(--accent-blue);">
                {t("temporal_trajectory")}
            </span>
        </div>
        <div style="font-size: 1.35rem; font-weight: 800; color: var(--text-primary); letter-spacing: -0.5px;">
            {t("trajectory_title")}
        </div>
    </div>
    """)

    @st.cache_data(show_spinner=False)
    def compute_mine_trend(mine_id_val: str):
        sub_df = df_prod[df_prod["mine_id"] == mine_id_val].sort_values("date").reset_index(drop=True)
        sub_2024 = sub_df[sub_df["date"] >= "2024-01-01"].copy()
        if sub_2024.empty:
            sub_2024 = sub_df.tail(120).copy()
        preds = []
        for _, row in sub_2024.iterrows():
            r_dict = row.to_dict()
            res = production_predict(r_dict, bundle=bundle, include_explain=False)
            preds.append(res["expected_production_tpd"])
        sub_2024["expected_production_tpd"] = preds
        return sub_2024

    mine_predictions_df = compute_mine_trend(selected_mine)

    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(
        x=mine_predictions_df["date"],
        y=mine_predictions_df["planned_production_tpd"],
        mode="lines",
        name=t("planned_target_legend"),
        line=dict(color=T["plotly_line_target"], width=1.8, dash="dash"),
        hovertemplate="<b>Date:</b> %{x|%d %b %Y}<br><b>Planned Target:</b> %{y:,.0f} t/d<extra></extra>"
    ))
    fig_trend.add_trace(go.Scatter(
        x=mine_predictions_df["date"],
        y=mine_predictions_df["expected_production_tpd"],
        mode="lines",
        name=t("expected_capacity_legend"),
        line=dict(color=T["plotly_line_primary"], width=2.6),
        hovertemplate="<b>Date:</b> %{x|%d %b %Y}<br><b>Expected Capacity:</b> %{y:,.0f} t/d<extra></extra>"
    ))

    fig_trend.add_vline(
        x=shift_dt,
        line_width=1.5,
        line_dash="dot",
        line_color=T["accent_copper"],
        annotation_text=t("active_shift_marker"),
        annotation_position="top right"
    )

    fig_trend.update_layout(
        template=T["plotly_template"],
        height=280,
        margin=dict(l=30, r=20, t=10, b=30),
        plot_bgcolor=T["plotly_bg"],
        paper_bgcolor=T["plotly_paper_bg"],
        hovermode="x unified",
        font=dict(family="Space Grotesk", color=T["text_primary"]),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=11, color=T["text_secondary"])
        ),
        xaxis=dict(showgrid=True, gridcolor=T["plotly_grid"], tickformat="%b %Y", tickfont=dict(size=10, color=T["text_muted"])),
        yaxis=dict(title=t("prod_unit_tpd"), title_font=dict(size=11, color=T["text_secondary"]), showgrid=True, gridcolor=T["plotly_grid"], tickfont=dict(size=10, color=T["text_muted"]))
    )
    st.plotly_chart(fig_trend, use_container_width=True, config={"displayModeBar": False})

    render_html(f"""
    <div style="margin-top: 40px; margin-bottom: 8px;">
        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 4px;">
            <span style="font-size: 1.15rem; font-weight: 800; color: var(--accent-copper);">02</span>
            <span style="font-size: 0.72rem; font-weight: 800; letter-spacing: 1.5px; text-transform: uppercase; color: var(--accent-blue);">
                {t("why_kicker")}
            </span>
        </div>
        <div style="font-size: 1.55rem; font-weight: 800; color: var(--text-primary); letter-spacing: -0.5px;">
            {t("why_title")}
        </div>
        <div style="font-size: 0.85rem; color: var(--text-secondary); max-width: 820px; margin-top: 4px;">
            {t("why_desc")}
        </div>
    </div>
    """)

    col_shap_vis, col_shap_list = st.columns([1.15, 1.0], gap="large")

    with col_shap_vis:
        all_contribs = shap_drivers.get("all_contributions", [])
        if all_contribs:
            top_drivers = all_contribs[:8]
            labels = [d["label"] for d in reversed(top_drivers)]
            impacts = [d["shap_impact_tonnes"] for d in reversed(top_drivers)]
            colors = [T["plotly_bar_negative"] if imp < 0 else T["plotly_bar_positive"] for imp in impacts]

            fig_shap = go.Figure(go.Bar(
                x=impacts,
                y=labels,
                orientation="h",
                marker=dict(color=colors),
                text=[f"{imp:+.0f} t" for imp in impacts],
                textposition="auto",
                textfont=dict(size=10, color=T["text_primary"]),
                hovertemplate="<b>%{y}</b><br>Model-Associated Impact: %{x:+.1f} t/day<extra></extra>"
            ))

            fig_shap.update_layout(
                template=T["plotly_template"],
                height=260,
                margin=dict(l=10, r=10, t=10, b=20),
                plot_bgcolor=T["plotly_bg"],
                paper_bgcolor=T["plotly_paper_bg"],
                xaxis=dict(title=f"{t('prod_unit_tpd')} (vs. Baseline)", title_font=dict(size=10, color=T["text_secondary"]), zeroline=True, zerolinecolor=T["border"], showgrid=True, gridcolor=T["plotly_grid"], tickfont=dict(color=T["text_muted"])),
                yaxis=dict(tickfont=dict(size=10, color=T["text_primary"]))
            )
            st.plotly_chart(fig_shap, use_container_width=True, config={"displayModeBar": False})

    with col_shap_list:
        neg_drivers = shap_drivers.get("top_negative_drivers", [])
        pos_drivers = shap_drivers.get("top_positive_drivers", [])

        render_html(f"<div style='font-size: 0.72rem; font-weight: 800; letter-spacing: 1px; text-transform: uppercase; color: {T['risk_high']}; margin-bottom: 8px;'>{t('drag_factors')}</div>")
        if neg_drivers:
            for d in neg_drivers[:4]:
                val_str = f"{d['feature_value']:.1f} {d['unit']}".strip()
                ctrl_badge = f"<span style='background: rgba(16, 185, 129, 0.15); color: {T['accent_green']}; padding: 1px 6px; border-radius: 4px; font-size: 0.65rem; font-weight: 700;'>{t('badge_controllable')}</span>" if d.get("is_controllable") else f"<span style='background: rgba(120, 120, 120, 0.1); color: {T['text_muted']}; padding: 1px 6px; border-radius: 4px; font-size: 0.65rem;'>{t('badge_external')}</span>"
                render_html(f"""
                <div style="display: flex; justify-content: space-between; align-items: center; padding: 8px 12px; background: {T['surface']}; border: 1px solid rgba(239, 68, 68, 0.3); border-radius: 8px; margin-bottom: 6px;">
                    <div>
                        <span style="font-size: 0.82rem; font-weight: 700; color: {T['text_primary']};">{d['label']}</span>
                        <span style="font-size: 0.72rem; color: {T['text_muted']}; margin-left: 6px;">({val_str})</span>
                        <span style="margin-left: 6px;">{ctrl_badge}</span>
                    </div>
                    <span style="font-size: 0.88rem; font-weight: 800; color: {T['risk_high']};">{d['shap_impact_tonnes']:+.0f} t/d</span>
                </div>
                """)
        else:
            render_html(f"<div style='font-size: 0.80rem; color: {T['accent_green']};'>{t('no_drag')}</div>")

        render_html(f"<div style='font-size: 0.72rem; font-weight: 800; letter-spacing: 1px; text-transform: uppercase; color: {T['accent_green']}; margin: 12px 0 8px 0;'>{t('boost_factors')}</div>")
        if pos_drivers:
            for d in pos_drivers[:2]:
                val_str = f"{d['feature_value']:.1f} {d['unit']}".strip()
                render_html(f"""
                <div style="display: flex; justify-content: space-between; align-items: center; padding: 6px 12px; background: {T['surface']}; border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 8px; margin-bottom: 4px;">
                    <div>
                        <span style="font-size: 0.80rem; font-weight: 700; color: {T['text_primary']};">{d['label']}</span>
                        <span style="font-size: 0.72rem; color: {T['text_muted']}; margin-left: 6px;">({val_str})</span>
                    </div>
                    <span style="font-size: 0.84rem; font-weight: 800; color: {T['accent_green']};">{d['shap_impact_tonnes']:+.0f} t/d</span>
                </div>
                """)

    render_html(f"""
    <div style="margin-top: 40px; margin-bottom: 12px;">
        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 4px;">
            <span style="font-size: 1.15rem; font-weight: 800; color: var(--accent-copper);">03</span>
            <span style="font-size: 0.72rem; font-weight: 800; letter-spacing: 1.5px; text-transform: uppercase; color: var(--accent-blue);">
                {t("briefing_kicker")}
            </span>
        </div>
        <div style="font-size: 1.55rem; font-weight: 800; color: var(--text-primary); letter-spacing: -0.5px;">
            {t("briefing_title")}
        </div>
        <div style="font-size: 0.85rem; color: var(--text-secondary); margin-top: 4px;">
            {t("briefing_desc")}
        </div>
    </div>
    """)

    for idx, rec in enumerate(recommendations[:4], 1):
        num_str = f"{idx:02d}"
        prio = rec["priority"]
        prio_color = T["risk_high"] if prio == "High" else (T["risk_med"] if prio == "Medium" else T["accent_blue"])
        prio_label = t("prio_high") if prio == "High" else (t("prio_medium") if prio == "Medium" else t("prio_low"))

        render_html(f"""
        <div style="display: flex; gap: 20px; align-items: flex-start; padding: 18px 20px; background: {T['surface']}; border: 1px solid {T['border']}; border-radius: 10px; margin-bottom: 10px; box-shadow: {T['card_shadow']};">
            <div style="font-size: 1.45rem; font-weight: 800; color: var(--accent-copper); line-height: 1;">
                {num_str}
            </div>
            <div style="flex: 1;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                    <span style="font-size: 0.92rem; font-weight: 800; color: var(--text-primary); text-transform: uppercase; letter-spacing: 0.5px;">
                        {rec['title']}
                    </span>
                    <span style="font-size: 0.68rem; font-weight: 800; color: {prio_color}; text-transform: uppercase; letter-spacing: 1px;">
                        {prio_label}
                    </span>
                </div>
                <div style="font-size: 0.82rem; color: var(--text-secondary); line-height: 1.45; margin-bottom: 4px;">
                    {rec['action']}
                </div>
                <div style="font-size: 0.72rem; color: var(--text-muted); font-style: italic;">
                    {t("telemetry_shap_trigger")} {rec['rationale']}
                </div>
            </div>
        </div>
        """)

    render_html(f"""
    <div style="margin-top: 44px; margin-bottom: 12px; padding-top: 24px; border-top: 1px solid {T['border']};">
        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 4px;">
            <span style="font-size: 1.15rem; font-weight: 800; color: var(--accent-copper);">04</span>
            <span style="font-size: 0.72rem; font-weight: 800; letter-spacing: 1.5px; text-transform: uppercase; color: var(--accent-green);">
                {t("recov_kicker")}
            </span>
        </div>
        <div style="font-size: 1.85rem; font-weight: 800; color: var(--text-primary); letter-spacing: -0.8px;">
            {t("recov_title")}
        </div>
        <div style="font-size: 0.88rem; color: var(--text-secondary); max-width: 820px; margin-top: 4px;">
            {t("recov_desc")}
        </div>
    </div>
    """)

    recov_list = list_recovery_scenarios()
    recov_cols = st.columns(len(recov_list))
    if "active_recovery_scenario" not in st.session_state:
        st.session_state.active_recovery_scenario = "improve_shovel_availability"

    for idx, r_scen in enumerate(recov_list):
        with recov_cols[idx]:
            is_active = (st.session_state.active_recovery_scenario == r_scen["key"])
            btn_label = f"{r_scen['icon']} {r_scen['category']}"
            if st.button(btn_label, key=f"btn_recov_{r_scen['key']}", type="primary" if is_active else "secondary", use_container_width=True):
                st.session_state.active_recovery_scenario = r_scen["key"]
                st.rerun()

    active_recov_key = st.session_state.active_recovery_scenario

    recov_result = simulate_recovery_scenario(
        baseline_inputs=current_inputs,
        scenario_key=active_recov_key,
        predict_fn=lambda x: production_predict(x, bundle=bundle, include_explain=False),
        baseline_prediction=baseline_res
    )

    r_scen_expected = recov_result["scenario_expected_tpd"]
    r_recovery_tonnes = recov_result["recovery_tonnes"]
    r_recovery_pct = recov_result["recovery_pct"]
    r_remaining_shortfall = recov_result["remaining_shortfall_tonnes"]
    r_remaining_pct = recov_result["remaining_shortfall_pct"]
    r_new_risk = recov_result["new_risk_level"]
    r_new_risk_color = recov_result["new_risk_color"]
    r_improved = recov_result["risk_improved"]

    improved_str = f" {t('improved_tag')}" if r_improved else ""
    render_html(f"""
    <div style="background: {T['recovery_box_bg']}; border: 1px solid {T['recovery_box_border']}; border-left: 6px solid var(--accent-green); border-radius: 12px; padding: 22px 24px; margin-top: 14px; box-shadow: {T['card_shadow']};">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 14px; margin-bottom: 16px;">
            <div>
                <span style="font-size: 1.15rem; font-weight: 800; color: var(--text-primary);">
                    {recov_result['scenario_icon']} {t("active_intervention")} {recov_result['scenario_name']}
                </span>
                <div style="font-size: 0.80rem; color: var(--text-secondary); margin-top: 2px;">
                    {recov_result['scenario_description']}
                </div>
            </div>
            <div style="display: inline-block; background: {r_new_risk_color}; color: #ffffff; padding: 3px 12px; border-radius: 9999px; font-size: 0.76rem; font-weight: 800; letter-spacing: 0.5px;">
                {t("revised_risk")}: {r_new_risk}{improved_str}
            </div>
        </div>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 16px; padding-top: 14px; border-top: 1px solid {T['border']};">
            <div>
                <div style="font-size: 0.68rem; font-weight: 700; text-transform: uppercase; color: var(--text-muted); letter-spacing: 1px;">{t("baseline_col")}</div>
                <div style="font-size: 1.85rem; font-weight: 800; color: var(--text-primary);">{expected_tpd:,.0f} <span style="font-size: 0.85rem; color: var(--text-muted);">t/d</span></div>
            </div>
            <div style="border-left: 1px solid {T['border']}; padding-left: 16px;">
                <div style="font-size: 0.68rem; font-weight: 700; text-transform: uppercase; color: var(--text-muted); letter-spacing: 1px;">{t("simulated_col")}</div>
                <div style="font-size: 1.85rem; font-weight: 800; color: var(--text-primary);">{r_scen_expected:,.0f} <span style="font-size: 0.85rem; color: var(--accent-green); font-weight: 700;">▲ t/d</span></div>
            </div>
            <div style="border-left: 1px solid {T['border']}; padding-left: 16px;">
                <div style="font-size: 0.68rem; font-weight: 700; text-transform: uppercase; color: var(--text-muted); letter-spacing: 1px;">{t("recovery_col")}</div>
                <div style="font-size: 1.85rem; font-weight: 800; color: var(--text-primary);">+{r_recovery_tonnes:,.0f} <span style="font-size: 0.85rem; color: var(--accent-green); font-weight: 700;">t/d (+{r_recovery_pct:.1f}%)</span></div>
            </div>
            <div style="border-left: 1px solid {T['border']}; padding-left: 16px;">
                <div style="font-size: 0.68rem; font-weight: 700; text-transform: uppercase; color: var(--text-muted); letter-spacing: 1px;">{t("shortfall_col")}</div>
                <div style="font-size: 1.85rem; font-weight: 800; color: var(--text-primary);">
                    {r_remaining_shortfall:,.0f} <span style="font-size: 0.85rem; color: {'var(--accent-green)' if r_remaining_shortfall <= 0 else 'var(--accent-red)'}; font-weight: 700;">t/d ({r_remaining_pct:.1f}%)</span>
                </div>
            </div>
        </div>
    </div>
    """)

    r_mods = recov_result.get("applied_modifications", [])
    if r_mods:
        with st.expander(f"🔍 {t('inspect_controllable', count=len(r_mods))}", expanded=False):
            st.dataframe(pd.DataFrame([
                {
                    t("col_lever"): m["parameter"],
                    t("col_base_val"): m["original_value"],
                    t("col_sim_setting"): m["modified_value"],
                    t("col_feasibility"): m["reason"]
                }
                for m in r_mods
            ]), use_container_width=True, hide_index=True)

    render_html(f"""
    <div style="margin-top: 44px; margin-bottom: 12px; padding-top: 24px; border-top: 1px solid {T['border']};">
        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 4px;">
            <span style="font-size: 1.15rem; font-weight: 800; color: var(--accent-copper);">05</span>
            <span style="font-size: 0.72rem; font-weight: 800; letter-spacing: 1.5px; text-transform: uppercase; color: var(--accent-blue);">
                {t("hazard_kicker")}
            </span>
        </div>
        <div style="font-size: 1.85rem; font-weight: 800; color: var(--text-primary); letter-spacing: -0.8px;">
            {t("hazard_title")}
        </div>
        <div style="font-size: 0.88rem; color: var(--text-secondary); max-width: 820px; margin-top: 4px;">
            {t("hazard_desc")}
        </div>
    </div>
    """)

    if "active_hazard_scenario" not in st.session_state:
        st.session_state.active_hazard_scenario = "baseline"

    col_h1, col_h2, col_h3, col_h4, col_h5 = st.columns(5)
    with col_h1:
        if st.button(t("btn_base_shift"), key="hz_btn_base", use_container_width=True):
            st.session_state.active_hazard_scenario = "baseline"
            st.rerun()
    with col_h2:
        if st.button(t("btn_heavy_rain"), key="hz_btn_rain", use_container_width=True):
            st.session_state.active_hazard_scenario = "heavy_rainfall"
            st.rerun()
    with col_h3:
        if st.button(t("btn_shovel_break"), key="hz_btn_shovel", use_container_width=True):
            st.session_state.active_hazard_scenario = "shovel_breakdown"
            st.rerun()
    with col_h4:
        if st.button(t("btn_multi_hazard"), key="hz_btn_multi", use_container_width=True):
            st.session_state.active_hazard_scenario = "multi_hazard"
            st.rerun()
    with col_h5:
        if st.button(t("btn_reset_base"), key="hz_btn_reset", use_container_width=True):
            st.session_state.active_hazard_scenario = "baseline"
            st.rerun()

    active_hazard_key = st.session_state.active_hazard_scenario

    hazard_sim = simulate_scenario(
        base_inputs=current_inputs,
        scenario_key=active_hazard_key,
        predict_fn=lambda x: production_predict(x, bundle=bundle, include_explain=False),
        baseline_prediction=baseline_res
    )

    h_scen_expected = hazard_sim["scenario_prediction"]["expected_production_tpd"]
    h_delta_tonnes = hazard_sim["delta_tonnes"]
    h_delta_pct = hazard_sim["delta_pct"]
    h_risk = hazard_sim["scenario_prediction"]["risk_level"]
    h_risk_color = hazard_sim["scenario_prediction"]["risk_color"]
    h_shortfall = hazard_sim["scenario_prediction"]["shortfall_tonnes"]

    render_html(f"""
    <div style="background: {T['hazard_box_bg']}; border: 1px solid {T['hazard_box_border']}; border-left: 6px solid {h_risk_color}; border-radius: 12px; padding: 22px 24px; margin-top: 14px; box-shadow: {T['card_shadow']};">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 14px; margin-bottom: 16px;">
            <div>
                <span style="font-size: 1.15rem; font-weight: 800; color: var(--text-primary);">
                    {hazard_sim['scenario_icon']} {t("active_hazard_sim")} {hazard_sim['scenario_name']}
                </span>
                <div style="font-size: 0.80rem; color: var(--text-secondary); margin-top: 2px;">
                    {hazard_sim['scenario_description']}
                </div>
            </div>
            <div style="display: inline-block; background: {h_risk_color}; color: #ffffff; padding: 3px 12px; border-radius: 9999px; font-size: 0.76rem; font-weight: 800; letter-spacing: 0.5px;">
                {t("hazard_risk_tag")}: {h_risk}
            </div>
        </div>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 16px; padding-top: 14px; border-top: 1px solid {T['border']};">
            <div>
                <div style="font-size: 0.68rem; font-weight: 700; text-transform: uppercase; color: var(--text-muted); letter-spacing: 1px;">{t("baseline_col")}</div>
                <div style="font-size: 1.85rem; font-weight: 800; color: var(--text-primary);">{expected_tpd:,.0f} <span style="font-size: 0.85rem; color: var(--text-muted);">t/d</span></div>
            </div>
            <div style="border-left: 1px solid {T['border']}; padding-left: 16px;">
                <div style="font-size: 0.68rem; font-weight: 700; text-transform: uppercase; color: var(--text-muted); letter-spacing: 1px;">{t("stress_col")}</div>
                <div style="font-size: 1.85rem; font-weight: 800; color: var(--text-primary);">{h_scen_expected:,.0f} <span style="font-size: 0.85rem; color: {h_risk_color}; font-weight: 700;">▼ t/d</span></div>
            </div>
            <div style="border-left: 1px solid {T['border']}; padding-left: 16px;">
                <div style="font-size: 0.68rem; font-weight: 700; text-transform: uppercase; color: var(--text-muted); letter-spacing: 1px;">{t("capacity_change_col")}</div>
                <div style="font-size: 1.85rem; font-weight: 800; color: var(--text-primary);">{h_delta_tonnes:+,.0f} <span style="font-size: 0.85rem; color: {'var(--accent-green)' if h_delta_tonnes >= 0 else 'var(--accent-red)'}; font-weight: 700;">t/d ({h_delta_pct:+.1f}%)</span></div>
            </div>
            <div style="border-left: 1px solid {T['border']}; padding-left: 16px;">
                <div style="font-size: 0.68rem; font-weight: 700; text-transform: uppercase; color: var(--text-muted); letter-spacing: 1px;">{t("projected_deficit_col")}</div>
                <div style="font-size: 1.85rem; font-weight: 800; color: var(--text-primary);">
                    {h_shortfall:,.0f} <span style="font-size: 0.85rem; color: var(--accent-red); font-weight: 700;">t/d</span>
                </div>
            </div>
        </div>
    </div>
    """)

    with st.expander(t("benchmark_expander"), expanded=False):
        render_html(f"""
        <div style="font-size: 0.82rem; color: var(--text-secondary); line-height: 1.5; margin-bottom: 12px;">
            {t("benchmark_body")}
        </div>
        """)

        xgb_m = metrics.get("models", {}).get("xgboost", {}).get("metrics", {"mae": 118.09, "rmse": 162.54, "r2": 0.9574})
        rf_m = metrics.get("models", {}).get("random_forest_benchmark", {}).get("metrics", {"mae": 141.16, "rmse": 192.85, "r2": 0.9400})

        col_b1, col_b2, col_b3 = st.columns(3)
        with col_b1:
            st.metric(t("mae_label"), f"{xgb_m['mae']} t/d", delta=t("better_than_rf", diff=rf_m['mae'] - xgb_m['mae']), delta_color="normal")
        with col_b2:
            st.metric(t("rmse_label"), f"{xgb_m['rmse']} t/d", delta=t("better_than_rf", diff=rf_m['rmse'] - xgb_m['rmse']), delta_color="normal")
        with col_b3:
            st.metric(t("r2_label"), f"{xgb_m['r2']:.4f}", delta=t("vs_rf_benchmark", diff=xgb_m['r2'] - rf_m['r2']), delta_color="normal")

        render_html(f"""
        <div style="font-size: 0.75rem; color: var(--text-muted); margin-top: 10px; border-top: 1px solid var(--border); padding-top: 8px;">
            {t("split_strategy_note")}
        </div>
        """)
