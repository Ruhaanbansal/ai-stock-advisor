import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime, timedelta
from src.features import create_extended_features
from src.dataloader import load_stock_data, get_close_prices
from src.insight import generate_chart_insights
from src.ui.components import show_insight
import os

def show_dashboard(data, close_prices, stock, period, articles=None):
    """Displays the main price dashboard with technical indicators and news overlays."""
    st.subheader("Price Chart & Technical Indicators", help="Historical price trends overlayed with Exponential Moving Averages (20 & 50 day).")

    extended = create_extended_features(data)

    chart_tab, bb_tab, macd_tab = st.tabs(["Price & MAs", "Bollinger Bands", "MACD"])

    with chart_tab:
        # Google-style logic: color based on net performance in period
        is_up      = close_prices.iloc[-1] >= close_prices.iloc[0]
        chart_color = "#00d4aa" if is_up else "#ff5c7c"
        fill_color  = "rgba(0, 212, 170, 0.08)" if is_up else "rgba(255, 92, 124, 0.08)"

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=close_prices.index, y=close_prices,
            name="Close",
            line=dict(color=chart_color, width=2),
            fill="tozeroy",
            fillcolor=fill_color,
            hoverinfo="x+y",
        ))

        # ── News Overlay ─────────────────────────────────────────
        if articles:
            news_dates = []
            news_prices = []
            news_texts = []
            
            for art in articles[:15]: # Max 15 markers
                try:
                    dt = pd.to_datetime(art['publishedAt']).tz_localize(None).normalize()
                    if dt in close_prices.index:
                        news_dates.append(dt)
                        news_prices.append(close_prices.loc[dt])
                        news_texts.append(f"<b>{art['source']}</b>: {art['title'][:60]}...")
                except: continue
            
            if news_dates:
                fig.add_trace(go.Scatter(
                    x=news_dates, y=news_prices,
                    mode='markers', name='News',
                    marker=dict(symbol='diamond', size=10, color='#6C63FF', line=dict(width=1, color='white')),
                    text=news_texts, hoverinfo='text'
                ))

        # Thinner SMAs with coordinated colors
        if "SMA20" in extended:
            fig.add_trace(go.Scatter(
                x=extended.index, y=extended["SMA20"],
                name="SMA 20", line=dict(color="rgba(108, 99, 255, 0.5)", width=1.2, dash="dash")
            ))
        if "SMA50" in extended:
            fig.add_trace(go.Scatter(
                x=extended.index, y=extended["SMA50"],
                name="SMA 50", line=dict(color="rgba(255, 179, 71, 0.5)", width=1.2, dash="dash")
            ))

        fig.update_layout(
            template="plotly_dark",
            height=450,
            showlegend=True,
            legend=dict(orientation="h", y=1.02, x=0, font=dict(size=10)),
            margin=dict(l=0, r=0, t=30, b=0),
            xaxis=dict(
                showgrid=False,
                zeroline=False,
                showline=False,
                tickfont=dict(color="#7a8299", size=10),
            ),
            yaxis=dict(
                gridcolor="rgba(122, 130, 153, 0.1)",
                zeroline=False,
                showline=False,
                tickfont=dict(color="#7a8299", size=10),
                side="right",
            ),
            hovermode="x unified",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, width="stretch")

        # AI Insights Snackbar for Price
        insights = generate_chart_insights(data, extended)
        for ins in insights:
            show_insight(ins)

        # Volume chart
        if "Volume" in data.columns:
            st.subheader("Trading Volume", help="Daily trading volume. Massive spikes often indicate institutional entry or exit.")
            vol_fig = go.Figure(go.Bar(
                x=data.index, y=data["Volume"],
                marker_color="rgba(122, 130, 153, 0.3)",
                name="Volume"
            ))
            vol_fig.update_layout(template="plotly_dark", height=200, margin=dict(l=0,r=0,t=0,b=0))
            st.plotly_chart(vol_fig, width="stretch")

        # ── Comparison Chart (if active) ──────────────────────────
        if st.session_state.get("compare_ticker"):
            st.markdown("---")
            st.subheader(f"📊 Relative Performance: {stock} vs {st.session_state.compare_ticker}")
            
            with st.status("Fetching comparison data...", expanded=False):
                comp_data = load_stock_data(st.session_state.compare_ticker, period=period)
            
            if comp_data is not None and not comp_data.empty:
                c_close = get_close_prices(comp_data)
                
                # Re-index to match primary data dates
                c_close = c_close.reindex(close_prices.index, method='ffill')
                
                # Normalize to 0% at start
                norm_main = ((close_prices / close_prices.iloc[0]) - 1) * 100
                norm_comp = ((c_close / c_close.iloc[0]) - 1) * 100
                
                fig_comp = go.Figure()
                fig_comp.add_trace(go.Scatter(
                    x=norm_main.index, y=norm_main, name=stock,
                    line=dict(color="#00d4aa", width=2)
                ))
                fig_comp.add_trace(go.Scatter(
                    x=norm_comp.index, y=norm_comp, name=st.session_state.compare_ticker,
                    line=dict(color="#6c63ff", width=2)
                ))
                
                fig_comp.update_layout(
                    template="plotly_dark", height=400,
                    yaxis_title="Percent Change (%)",
                    hovermode="x unified",
                    margin=dict(l=0,r=0,t=20,b=0),
                    legend=dict(orientation="h", y=1.1)
                )
                st.plotly_chart(fig_comp, width="stretch")
                st.caption("Performance is normalized to 0% at the beginning of the period.")

    with bb_tab:
        if "BB_Upper" in extended:
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=extended.index, y=extended["BB_Upper"],
                                      name="Upper Band", line=dict(color="#ff5c7c", dash="dot", width=1)))
            fig2.add_trace(go.Scatter(x=extended.index, y=extended["BB_Mid"],
                                      name="Middle (SMA20)", line=dict(color="#7a8299", width=1)))
            fig2.add_trace(go.Scatter(x=extended.index, y=extended["BB_Lower"],
                                      name="Lower Band", line=dict(color="#00d4aa", dash="dot", width=1),
                                      fill="tonexty", fillcolor="rgba(0,212,170,0.04)"))
            fig2.add_trace(go.Scatter(x=close_prices.index, y=close_prices,
                                      name="Close", line=dict(color="white", width=1.5)))
            fig2.update_layout(template="plotly_dark", height=400,
                               margin=dict(l=0,r=0,t=10,b=0))
            st.plotly_chart(fig2, width='stretch')
        else:
            st.info("Not enough data for Bollinger Bands.")

    with macd_tab:
        if "MACD" in extended:
            fig3 = go.Figure()
            macd_hist = extended["MACD"] - extended["MACD_Signal"]
            fig3.add_trace(go.Bar(x=extended.index, y=macd_hist, name="Histogram",
                                  marker_color=["#00d4aa" if v >= 0 else "#ff5c7c" for v in macd_hist]))
            fig3.add_trace(go.Scatter(x=extended.index, y=extended["MACD"],
                                      name="MACD", line=dict(color="#6c63ff")))
            fig3.add_trace(go.Scatter(x=extended.index, y=extended["MACD_Signal"],
                                      name="Signal", line=dict(color="#ffb347", dash="dot")))
            fig3.update_layout(template="plotly_dark", height=380,
                               margin=dict(l=0,r=0,t=10,b=0))
            st.plotly_chart(fig3, width='stretch')
        else:
            st.info("Not enough data for MACD.")

    # 5-day forecast strip
    st.subheader("5-Day Price Forecast")
    today  = datetime.today()
    fdates = [(today + timedelta(days=i+1)).strftime("%a %d %b") for i in range(5)]
    fcols  = st.columns(5)
    
    # We expect forecast_5d to be passed in or calculated
    from src.model import forecast_prices, predict_next_price
    # For now, let's assume it's passed or recalculated to keep it stable
    # Actually, we can just pass it directly from app.py
