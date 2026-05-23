import html as _html
import streamlit as st


def load_css():

    st.markdown("""
    <style>

    .stApp {
        background: #0f0f13;
        color: white;
    }

    .main-title {
        font-size: 2rem;
        font-weight: bold;
        color: #e2ff5d;
    }

    .sub-title {
        color: #888;
        margin-bottom: 20px;
    }

    .chat-user {
        background: #1e1e28;
        padding: 12px;
        border-radius: 12px;
        margin: 10px 0;
        text-align: right;
    }

    .chat-bot {
        background: #191924;
        padding: 12px;
        border-radius: 12px;
        margin: 10px 0;
        border-left: 3px solid #e2ff5d;
    }

    </style>
    """, unsafe_allow_html=True)


def render_user_message(message):

    safe_msg = _html.escape(message)
    st.markdown(
        f'<div class="chat-user">🧑 {safe_msg}</div>',
        unsafe_allow_html=True,
    )


def render_bot_message(message, source):

    safe_msg = _html.escape(message)

    badge = (
        "📖 WIKIPEDIA"
        if source == "wiki_search"
        else "🗄️ VECTORSTORE"
    )

    st.markdown(
        f"""
        <div class="chat-bot">
            <b>{badge}</b><br><br>
            🤖 {safe_msg}
        </div>
        """,
        unsafe_allow_html=True,
    )