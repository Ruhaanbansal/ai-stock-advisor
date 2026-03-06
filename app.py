import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go


# Project Modules
from src.dataloader import load_stock_data, get_close_prices, load_portfolio_data, calculate_returns
from src.model import train_lstm_model
from src.sentiment import get_news_sentiment, get_stock_news
from src.risk import calculate_risk_metrics
from src.explainability import generate_explanation, shap_explanation, lime_explanation
from src.portfolio import recommend_portfolio
from src.backtest import run_backtest
from src.evaluation import evaluate_model
from src.advisor import run_ai_advisor
from src.alerts import detect_market_alerts
from src.insight import generate_ai_insight

# --------------------------------------------------
# CUSTOM DASHBOARD STYLING
# --------------------------------------------------

st.markdown("""
<style>

body {
    background-color: #0e1117;
}

.metric-card {
    background-color: #1c1f26;
    padding: 15px;
    border-radius: 12px;
    text-align: center;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.3);
}

.section-header {
    font-size: 26px;
    font-weight: bold;
    margin-top: 20px;
}

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="AI Stock Advisor",
    page_icon="📈",
    layout="wide"
)

st.title("📈 AI-Powered Stock Advisor")

st.markdown(
"""
This platform combines **Deep Learning (LSTM)**, **Financial News Sentiment**,  
**Risk Analytics**, and **Explainable AI** to provide intelligent investment insights.
"""
)

# --------------------------------------------------
# SIDEBAR NAVIGATION
# --------------------------------------------------

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go to",
    [
        "Dashboard",
        "Market Intelligence",
        "AI Prediction",
        "Portfolio Optimizer",
        "Backtesting",
        "Model Evaluation"
    ]
)

stock = st.sidebar.selectbox(
    "Select NSE Stock",
    ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS"]
)

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

with st.spinner("Loading stock data..."):

    data = load_stock_data(stock)

    if data is None or data.empty:
        st.error("Unable to fetch stock data.")
        st.stop()

    close_prices = get_close_prices(data)

# --------------------------------------------------
# TRAIN MODEL
# --------------------------------------------------

with st.spinner("Training AI model..."):

    model, scaler, last_sequence = train_lstm_model(close_prices)

sequence_length = 60

X_input = last_sequence.reshape(1, sequence_length, 4)

predicted_scaled = model.predict(X_input, verbose=0)

predicted_price = scaler.inverse_transform(
    np.concatenate([predicted_scaled, np.zeros((1,3))], axis=1)
)[:,0]

predicted_price = float(predicted_price[0])

current_price = float(close_prices.iloc[-1])

# --------------------------------------------------
# SENTIMENT
# --------------------------------------------------

sentiment_score, sentiment_label, headlines = get_news_sentiment(stock)

# --------------------------------------------------
# RISK
# --------------------------------------------------

volatility, sharpe_ratio, max_drawdown = calculate_risk_metrics(close_prices)

# -----------------------------------
# AI Advisor Decision
# -----------------------------------

advisor = run_ai_advisor(
    current_price,
    predicted_price,
    volatility,
    sentiment_score
)

# -----------------------------------
# MARKET OVERVIEW (ADD HERE)
# -----------------------------------

st.markdown("## Market Overview")

k1, k2, k3, k4 = st.columns(4)

k1.metric("Current Price", f"₹{current_price:.2f}")
k2.metric("Predicted Price", f"₹{predicted_price:.2f}")
k3.metric("Expected Change", f"{advisor['price_change']:.2f}%")
k4.metric("Sentiment", sentiment_label)

# -----------------------------------
# STOCK PRICE CHART
# -----------------------------------

st.subheader("Stock Price Chart")

st.line_chart(close_prices)

# ==================================================
# DASHBOARD PAGE
# ==================================================

if page == "Dashboard":

    st.subheader("Stock Price Trend")

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=close_prices.index,
            y=close_prices,
            mode="lines",
            name="Stock Price"
        )
    )

    fig.update_layout(
        template="plotly_dark",
        height=450,
        xaxis_title="Date",
        yaxis_title="Price"
    )

    st.plotly_chart(fig, use_container_width=True)

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Current Price", f"₹{current_price:.2f}")
    col2.metric("Predicted Price", f"₹{predicted_price:.2f}")
    col3.metric("Expected Change", f"{advisor['price_change']:.2f}%")
    col4.metric("Recommendation", advisor["recommendation"])

    st.progress(int(advisor["confidence"]))

    st.write(f"AI Confidence Score: **{advisor['confidence']}%**")

# -----------------------------------
# AI Confidence
# -----------------------------------

    # -----------------------------------
# AI Confidence
# -----------------------------------

    st.subheader("AI Confidence")

    import plotly.graph_objects as go

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=advisor["confidence"],
        title={'text': "AI Confidence"},
        gauge={
            'axis': {'range': [0,100]},
            'bar': {'color': "green"},
            'steps': [
                {'range': [0,40], 'color': "#ff4d4d"},
                {'range': [40,70], 'color': "#ffa500"},
                {'range': [70,100], 'color': "#00cc66"}
            ]
        }
    ))

    fig.update_layout(height=300)

    st.plotly_chart(fig, use_container_width=True)   

# ==================================================
# MARKET INTELLIGENCE
# ==================================================

elif page == "Market Intelligence":

    st.header("Market Intelligence")

    col1, col2 = st.columns(2)

    col1.metric("Sentiment Score", f"{sentiment_score:.2f}")
    col2.metric("Market Sentiment", sentiment_label)

    st.subheader("Latest News")

    news = get_stock_news(stock)

    if len(news) == 0:
        st.warning("No recent news found.")
    else:
        for n in news:
            st.write("•", n)

    st.subheader("Market Risk Alerts")

    alerts = detect_market_alerts(close_prices, sentiment_score)

    if len(alerts) == 0:
        st.success("No major risk signals detected.")
    else:
        for alert in alerts:
            st.error(alert)

# ==================================================
# AI PREDICTION
# ==================================================

elif page == "AI Prediction":

    st.header("AI Investment Recommendation")

    col1, col2, col3 = st.columns(3)

    col1.metric("Current Price", f"₹{current_price:.2f}")
    col2.metric("Predicted Price", f"₹{predicted_price:.2f}")
    col3.metric("Expected Change", f"{advisor['price_change']:.2f}%")

    st.success(f"Recommendation: **{advisor['recommendation']}**")

    st.subheader("Risk Dashboard")

    col1, col2, col3 = st.columns(3)

    col1.metric("Volatility", f"{volatility:.2f}")
    col2.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
    col3.metric("Max Drawdown", f"{max_drawdown:.2%}")

    st.subheader("AI Explanation")

    reasons = advisor["reasons"]

    with st.expander("AI Investment Insight"):

        insight = generate_ai_insight(
            advisor["recommendation"],
            advisor["price_change"],
            sentiment_label,
            volatility
        )

        for i in insight:
            st.write("•", i)

    st.subheader("AI Advisor Insight")

    insight = generate_ai_insight(
        advisor["recommendation"],
        advisor["price_change"],
        sentiment_label,
        volatility
    )

    for i in insight:
        st.write("•", i)

    # ---------------- EXPLAINABILITY ----------------

    st.subheader("Model Explainability")

    tab1, tab2 = st.tabs(["SHAP", "LIME"])

    with tab1:

        feature_names = ["Close", "SMA20", "SMA50", "RSI"]

        try:

            importance = shap_explanation(model, X_input, feature_names)

            importance_df = pd.DataFrame({
                "Feature": importance.keys(),
                "Importance": importance.values()
            })

            st.bar_chart(importance_df.set_index("Feature"))

        except:
            st.warning("SHAP explanation unavailable")

    with tab2:

        try:

            lime_results = lime_explanation(model, X_input, feature_names)

            for feature, weight in lime_results:

                if weight > 0:
                    st.write(f"🟢 {feature} increases prediction")
                else:
                    st.write(f"🔴 {feature} decreases prediction")

        except:
            st.warning("LIME explanation unavailable")

# ==================================================
# PORTFOLIO OPTIMIZER
# ==================================================

elif page == "Portfolio Optimizer":

    st.header("Portfolio Optimization")

    portfolio_stocks = st.multiselect(
        "Select Portfolio Stocks",
        ["RELIANCE.NS","TCS.NS","INFY.NS","HDFCBANK.NS","ICICIBANK.NS"],
        default=["RELIANCE.NS","TCS.NS"]
    )

    if portfolio_stocks:

        portfolio_data = load_portfolio_data(portfolio_stocks)

        returns = calculate_returns(portfolio_data)

        investment = st.number_input(
            "Investment Amount",
            min_value=1000,
            value=100000
        )

        risk_level = st.selectbox(
            "Risk Level",
            ["Low","Medium","High"]
        )

        portfolio = recommend_portfolio(
            returns,
            investment,
            risk_level
        )

        st.write(portfolio)

        st.subheader("Portfolio Allocation")

        fig = go.Figure(
            data=[
                go.Pie(
                    labels=portfolio["Stock"],
                    values=portfolio["Investment"],
                    hole=0.4
                )
            ]
        )

    fig.update_layout(template="plotly_dark")

    st.plotly_chart(fig)

# ==================================================
# BACKTESTING
# ==================================================

elif page == "Backtesting":

    st.header("Strategy Backtesting")

    comparison, performance = run_backtest(stock)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=comparison.index,
            y=comparison["Buy & Hold"],
            name="Buy & Hold"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=comparison.index,
            y=comparison["AI Strategy"],
            name="AI Strategy"
        )
    )

    fig.update_layout(template="plotly_dark")

    st.plotly_chart(fig)

    col1, col2 = st.columns(2)

    col1.metric(
        "Buy & Hold Return",
        f"{performance['buy_hold_return']:.2f}%"
    )

    col2.metric(
        "AI Strategy Return",
        f"{performance['ai_return']:.2f}%"
    )

# ==================================================
# MODEL EVALUATION
# ==================================================

elif page == "Model Evaluation":

    st.header("Model Evaluation")

    evaluation = evaluate_model(
        close_prices,
        model,
        scaler
    )

    st.pyplot(evaluation["chart"])

    col1, col2 = st.columns(2)

    col1.metric("MAE", f"{evaluation['mae']:.2f}")
    col2.metric("RMSE", f"{evaluation['rmse']:.2f}")

    st.subheader("Model Confidence")

    confidence_eval = int(evaluation["confidence"])

    st.progress(confidence_eval)

    st.write(f"Model Confidence Score: **{confidence_eval:.2f}%**")