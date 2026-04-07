import streamlit as st
from PIL import Image
import io
from src.chart_analyzer import (analyse_chart, get_api_key, rec_style, trend_icon, pattern_type_color)

def show_analyzer_page():
    """Displays the AI Chart Analyzer section."""
    st.markdown("""
        <div style='margin-bottom:24px'>
            <h2 style='margin:0;font-size:1.6rem;font-weight:700;
                       background:linear-gradient(135deg,#00d4aa,#6c63ff);
                       -webkit-background-clip:text;-webkit-text-fill-color:transparent'>
                📸 AI Vision Chart Analyzer
            </h2>
            <p style='color:#7a8299;margin:4px 0 0;font-size:0.9rem'>
                Technical Pattern Recognition · Candlestick Logic · Multi-Timeframe Vision
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    col_up, col_res = st.columns([1, 1.5])
    
    with col_up:
        st.markdown("### 🖼️ Upload Chart Screenshot")
        uploaded_file = st.file_uploader("Upload PNG/JPG", type=["png", "jpg", "jpeg"])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Chart", width="stretch")
            
            if st.button("🔍 AI Analyze Chart", type="primary", width="stretch"):
                with st.status("AI Vision Engine scanning chart...", expanded=True):
                    analysis = analyse_chart(image)
                
                st.session_state.chart_analysis = analysis
                st.rerun()

    if "chart_analysis" in st.session_state:
        with col_res:
            st.markdown("### 🧠 AI Vision Analysis Output")
            analysis = st.session_state.chart_analysis
            
            # Recommendation Badge
            rec = analysis.get('recommendation', 'Hold')
            style = rec_style(rec)
            st.markdown(f"""
            <div style="background:#111520;border:1px solid #1e2538;border-radius:14px;padding:24px;text-align:center">
                <div style="font-size:12px;color:#7a8299;margin-bottom:8px">AI VISION RECOMMENDATION</div>
                <div style="{style}">{rec.upper()}</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Detailed Findings
            st.markdown("#### Patterns & Trends Detected")
            for pattern in analysis.get('patterns', []):
                st.markdown(f"""
                <div style="background:rgba(108, 99, 255, 0.1);border-left:4px solid #6c63ff;padding:12px;border-radius:6px;margin-bottom:10px">
                    <span style="font-weight:700;color:#e4e8f0">{pattern['type']}</span>: 
                    <span style="font-size:13px;color:#7a8299">{pattern['description']}</span>
                </div>
                """, unsafe_allow_html=True)
            
            # Final Verdict
            st.markdown("---")
            st.info(f"<b>Final Verdict:</b> {analysis.get('summary', 'No summary available.')}")
