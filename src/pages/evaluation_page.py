import streamlit as st
import plotly.graph_objects as go
from src.evaluation import evaluate_model

def show_evaluation_page(model, scaler, close_prices, train_mae, test_mae):
    """Displays the AI Model Evaluation metrics and performance charts."""
    st.markdown("""
        <div style='margin-bottom:24px'>
            <h2 style='margin:0;font-size:1.6rem;font-weight:700;
                       background:linear-gradient(135deg,#00d4aa,#6c63ff);
                       -webkit-background-clip:text;-webkit-text-fill-color:transparent'>
                🔬 AI Model Performance Analytics
            </h2>
            <p style='color:#7a8299;margin:4px 0 0;font-size:0.9rem'>
                Back-testing accuracy · Loss transformation · Predictive delta
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1.5])
    
    with col1:
        st.markdown("### 📊 Error Metrics")
        st.metric("Training Median Absolute Error (MAE)", f"₹{train_mae:,.2f}")
        st.metric("Test Median Absolute Error (MAE)", f"₹{test_mae:,.2f}")
        
        # Performance Indicators
        st.markdown("---")
        st.markdown("#### Model Reliability Indicator")
        reliability = "High" if test_mae < (close_prices.mean() * 0.05) else "Moderate"
        st.info(f"Model Reliability: **{reliability}**")
        st.caption("Determined by the ratio of MAE to price volatility.")

    with col2:
        st.markdown("### 📈 Predicted vs. Actual")
        # In a real app, evaluate_model would return the plotting data
        eval_data = evaluate_model(model, scaler, close_prices)
        
        fig_eval = go.Figure()
        fig_eval.add_trace(go.Scatter(
            x=eval_data['dates'], y=eval_data['actual'],
            name="Actual Price", line=dict(color="#e4e8f0", width=1.5)
        ))
        fig_eval.add_trace(go.Scatter(
            x=eval_data['dates'], y=eval_data['pred'],
            name="AI Prediction", line=dict(color="#00d4aa", dash="dot")
        ))
        fig_eval.update_layout(template="plotly_dark", height=400, margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig_eval, width="stretch")
