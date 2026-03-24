import streamlit as st
import os
from typing import List
from typing_extensions import TypedDict
from typing import Literal

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LangGraph AI Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }
    .stApp { background: #0f0f13; }
    .main-title {
        font-family: 'Space Mono', monospace;
        font-size: 2rem;
        font-weight: 700;
        color: #e2ff5d;
        letter-spacing: -1px;
    }
    .sub-title { color: #888; font-size: 0.9rem; margin-top: -10px; }
    .chat-bubble-user {
        background: #1e1e28;
        border: 1px solid #333;
        border-radius: 12px 12px 2px 12px;
        padding: 12px 16px;
        margin: 8px 0;
        color: #f0f0f0;
        text-align: right;
    }
    .chat-bubble-bot {
        background: #191924;
        border: 1px solid #2a2a3a;
        border-left: 3px solid #e2ff5d;
        border-radius: 2px 12px 12px 12px;
        padding: 12px 16px;
        margin: 8px 0;
        color: #d4d4d4;
    }
    .source-badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 0.7rem;
        font-family: 'Space Mono', monospace;
        font-weight: 700;
        letter-spacing: 0.5px;
        margin-bottom: 8px;
    }
    .badge-wiki { background: #1a3a5c; color: #60b4ff; border: 1px solid #2a5a8c; }
    .badge-vector { background: #1a3a1a; color: #60ff90; border: 1px solid #2a7a2a; }
    .stTextInput > div > div > input {
        background: #1e1e28 !important;
        border: 1px solid #333 !important;
        color: #f0f0f0 !important;
        border-radius: 8px !important;
    }
    .stButton > button {
        background: #e2ff5d !important;
        color: #0f0f13 !important;
        font-family: 'Space Mono', monospace !important;
        font-weight: 700 !important;
        border: none !important;
        border-radius: 8px !important;
    }
    .stButton > button:hover {
        background: #f0ff8a !important;
    }
    .sidebar-card {
        background: #1a1a24;
        border: 1px solid #2a2a3a;
        border-radius: 10px;
        padding: 12px;
        margin: 8px 0;
        font-size: 0.82rem;
        color: #aaa;
    }
    div[data-testid="stSidebar"] { background: #0d0d17; }
</style>
""", unsafe_allow_html=True)