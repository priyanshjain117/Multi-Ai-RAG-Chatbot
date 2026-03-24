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
# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuration")

    groq_api_key = st.text_input("Groq API Key", type="password", placeholder="gsk_...")
    astra_token   = st.text_input("AstraDB Token", type="password", placeholder="AstraCS:...")
    astra_db_id   = st.text_input("AstraDB ID", placeholder="fca479ba-...")

    st.divider()
    st.markdown("### 📚 Data Sources")
    default_urls = [
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
        "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
    ]
    urls_input = st.text_area(
        "Vectorstore URLs (one per line)",
        value="\n".join(default_urls),
        height=120,
    )

    load_btn = st.button("🚀 Initialize Agent", use_container_width=True)

    st.divider()
    st.markdown("""
    <div class="sidebar-card">
    <b>🔀 Routing Logic</b><br><br>
    Questions about <b>agents, prompts, adversarial attacks</b> → Vectorstore (AstraDB)<br><br>
    All other questions → Wikipedia Search
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="sidebar-card">
    <b>📦 Stack</b><br><br>
    LangGraph · LangChain · AstraDB · Groq (Gemma2-9b) · HuggingFace Embeddings · Wikipedia
    </div>
    """, unsafe_allow_html=True)
    # ── Main Area ─────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">🤖 LangGraph Agent</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Agentic RAG · AstraDB Vectorstore · Wikipedia Fallback</div>', unsafe_allow_html=True)
st.write("")

# ── Session State ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "agent_ready" not in st.session_state:
    st.session_state.agent_ready = False
if "app" not in st.session_state:
    st.session_state.app = None