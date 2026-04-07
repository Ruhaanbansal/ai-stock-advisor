import streamlit as st
import plotly.graph_objects as go
from src.explainability import compute_feature_importance, shap_chart, get_technical_signals

def show_prediction_page(advisor, model, scaler, close_prices):
    """Displays the AI Prediction & Market Intelligence page."""
    st.markdown("""
        <div style='margin-bottom:24px'>
            <h2 style='margin:0;font-size:1.6rem;font-weight:700;
                       background:linear-gradient(135deg,#00d4aa,#6c63ff);
                       -webkit-background-clip:text;-webkit-text-fill-color:transparent'>
                🧠 AI Investment Intelligence
            </h2>
            <p style='color:#7a8299;margin:4px 0 0;font-size:0.9rem'>
                Multi-factor signal decomposition · Institutional-grade consensus · Risk-adjusted verdict
            </p>
        </div>
    """, unsafe_allow_html=True)

    top_l, top_r = st.columns([1, 1.2])

    with top_l:
        # Decision Badge
        badge_class = {
            "Strong Buy": "badge-strong-buy", "Buy": "badge-buy",
            "Hold": "badge-hold", "Sell": "badge-sell",
        }.get(advisor["recommendation"], "badge-hold")
        
        st.markdown(f"""
        <div style="background:var(--surface);border:1px solid var(--border);border-radius:14px;padding:24px;text-align:center;height:100%">
          <div style="color:var(--muted);font-size:12px;margin-bottom:12px;letter-spacing:1px;font-weight:600">PREDICTED VERDICT</div>
          <span class="rec-badge {badge_class}" style="font-size:32px;padding:12px 36px">{advisor['recommendation']}</span>
          <div style="margin-top:20px;display:flex;justify-content:center;gap:32px">
            <div>
                <div style="color:var(--muted);font-size:10px">CONFIDENCE</div>
                <div style="font-size:18px;font-weight:700;color:var(--accent)">{advisor['confidence']}%</div>
            </div>
            <div style="border-left:1px solid var(--border);height:30px"></div>
            <div>
                <div style="color:var(--muted);font-size:10px">RISK LEVEL</div>
                <div style="font-size:18px;font-weight:700;color:#ffb347">{advisor['risk']}</div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    with top_r:
        # Decision Factor Radar
        radar_fig = go.Figure(data=go.Scatterpolar(
            r=list(advisor["factor_scores"].values()),
            theta=list(advisor["factor_scores"].keys()),
            fill='toself',
            fillcolor='rgba(0, 212, 170, 0.25)',
            line=dict(color='#00d4aa', width=3),
        ))
        radar_fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 100], showticklabels=False, gridcolor="rgba(122,130,153,0.1)"),
                angularaxis=dict(gridcolor="rgba(122,130,153,0.1)", linecolor="rgba(122,130,153,0.1)", tickfont=dict(size=10, color="#7a8299")),
                bgcolor="rgba(0,0,0,0)",
            ),
            showlegend=False, height=220,
            margin=dict(l=40, r=40, t=20, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(radar_fig, width='stretch')

    st.markdown("---")

    col_l, col_r = st.columns([1.1, 0.9])

    with col_l:
        st.markdown("### 🔍 Model Explainability")
        imp = compute_feature_importance(model, scaler, close_prices)
        st.plotly_chart(shap_chart(imp), width='stretch')
        
        # Technical Signal Capsules (Symmetric Grid)
        signals = get_technical_signals(close_prices)
        if signals:
            st.markdown("<div style='margin-top:16px'></div>", unsafe_allow_html=True)
            s_rows = st.columns(3)
            # We map signals to these slots
            for i, (name, (val, label, desc)) in enumerate(signals.items()):
                target_col = s_rows[i % 3]
                color = "#00d4aa" if "🟢" in label else "#ff5c7c" if "🔴" in label else "#ffb347"
                icon  = "●"
                target_col.markdown(f"""
                <div style="background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:12px;text-align:center;margin-bottom:12px">
                  <div style="font-size:9px;color:var(--muted);margin-bottom:6px;font-weight:600">{name.upper()}</div>
                  <div style="font-size:14px;font-weight:700;color:{color};line-height:1.2">{label}</div>
                  <div style="font-size:18px;color:{color if icon == '●' else '#fff'}">{icon}</div>
                  <div style="font-size:10px;color:var(--muted);margin-top:2px">{val}</div>
                </div>
                """, unsafe_allow_html=True)

    with col_r:
        st.markdown("### Detailed Analysis & Reasoning")
        
        reasoning_icons = ["📈", "📰", "🛡️", "📊", "🧭"]
        for i, reason in enumerate(advisor["reasons"]):
            icon = reasoning_icons[i % len(reasoning_icons)]
            style = "alert-info"
            if "📈" in reason or "🟢" in reason or "Bullish" in reason: style = "alert-ok"
            elif "📉" in reason or "🔴" in reason or "Bearish" in reason: style = "alert-danger"
            elif "⚠️" in reason or "🟡" in reason: style = "alert-warning"

            st.markdown(f"""
            <div class="{style}" style="display:flex;align-items:flex-start;gap:12px;padding:16px;border-radius:12px;margin-bottom:12px">
                <span style="font-size:20px;margin-top:-2px">{icon}</span>
                <div style="font-size:14px;line-height:1.5">{reason}</div>
            </div>
            """, unsafe_allow_html=True)
