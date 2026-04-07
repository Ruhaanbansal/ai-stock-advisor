with open('app.py', encoding='utf-8') as f:
    content = f.read()

# Fix deprecated Streamlit param
content = content.replace("width='stretch'", "use_container_width=True")

# Fix risk metrics key names (sharpe_ratio → sharpe in new risk.py)
content = content.replace('risk_metrics["sharpe_ratio"]', 'risk_metrics.get("sharpe", 0.0)')
content = content.replace("risk_metrics['sharpe_ratio']", 'risk_metrics.get("sharpe", 0.0)')

# Fix Model Evaluation page title
content = content.replace(
    'st.subheader("LSTM Model Evaluation")',
    'st.subheader("GradientBoosting Model Evaluation")'
)

# Fix model evaluation metrics display (test_mae, train_mae)
content = content.replace(
    'e1.metric("MAE",        f"₹{evaluation[\'mae\']:.2f}")',
    'e1.metric("Train MAE",  f"₹{evaluation.get(\'train_mae\', evaluation.get(\'mae\', 0)):.2f}")'
)
content = content.replace(
    'e2.metric("RMSE",       f"₹{evaluation[\'rmse\']:.2f}")',
    'e2.metric("Test MAE",   f"₹{evaluation.get(\'test_mae\', evaluation.get(\'rmse\', 0)):.2f}")'
)
content = content.replace(
    'e3.metric("MAPE",       f"{evaluation[\'mape\']:.2f}%")',
    'e3.metric("MAPE",       f"{evaluation.get(\'mape\', 0):.2f}%")'
)
content = content.replace(
    'e4.metric("Confidence", f"{evaluation[\'confidence\']:.1f}%")',
    'e4.metric("Dir. Accuracy", f"{evaluation.get(\'dir_acc\', 0):.1f}%")'
)

with open('app.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Done")
