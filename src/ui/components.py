import streamlit as st
import base64
import os

def inject_custom_css():
    """Injects assets/style.css into the Streamlit app."""
    css_path = os.path.join("assets", "style.css")
    if os.path.exists(css_path):
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def get_base64_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def show_brand():
    """Displays the NiftyMind logo and brand name."""
    logo_path = os.path.join("assets", "logo.svg")
    if os.path.exists(logo_path):
        logo_base64 = get_base64_bin_file(logo_path)
        st.markdown(f"""
        <div style="display: flex; align-items: center; gap: 12px; padding: 10px 0 20px;">
            <img src="data:image/svg+xml;base64,{logo_base64}" width="32" height="32">
            <div>
                <span style="font-family:'Space Grotesk',sans-serif; font-size:22px; font-weight:700; color:#00D4AA; letter-spacing:-0.5px;">
                    NiftyMind
                </span>
                <span style="font-size:10px; color:#7a8299; display:block; margin-top:-2px; letter-spacing:0.5px; font-weight:500;">
                    AI STOCK INTELLIGENCE
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("### NiftyMind")

def show_status_heartbeat(status="live"):
    """Displays a live status indicator in the sidebar with dynamic coloring."""
    configs = {
        "live":    {"color": "#00d4aa", "text": "SYSTEM LIVE",    "class": "heartbeat"},
        "delayed": {"color": "#ffb347", "text": "SYSTEM DELAYED", "class": "heartbeat-delayed"},
        "outage":  {"color": "#ff5c7c", "text": "SYSTEM OUTAGE",  "class": "heartbeat-outage"},
    }
    cfg = configs.get(status, configs["live"])
    
    st.sidebar.markdown(f"""
    <div style="display: flex; align-items: center; padding: 10px 0; border-top: 1px solid rgba(30,37,56,0.6); margin-top: 15px;">
        <span class="{cfg['class']}"></span>
        <span style="font-size: 11px; color: {cfg['color']}; font-weight: 600; letter-spacing: 0.5px;">{cfg['text']}</span>
        <span style="font-size: 10px; color: #7a8299; margin-left: auto;">NSE / YF</span>
    </div>
    """, unsafe_allow_html=True)

def show_insight(text, level="info"):
    """Displays a professional insight snackbar."""
    colors = {
        "info": "#6C63FF",
        "success": "#00D4AA",
        "warning": "#FFB347",
        "danger": "#FF5C7C"
    }
    st.markdown(f"""
    <div class="insight-card" style="border-left-color: {colors.get(level, '#6C63FF')};">
        <div style="font-size: 10px; color: #7a8299; font-weight: 700; margin-bottom: 4px; letter-spacing: 1px;">AI INSIGHT</div>
        <div style="font-size: 14px; line-height: 1.5; color: #e4e8f0;">{text}</div>
    </div>
    """, unsafe_allow_html=True)
