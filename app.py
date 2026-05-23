"""app.py — LangGraph RAG Agent · Production UI"""

import os
import streamlit as st
from datetime import datetime
from dotenv import load_dotenv
from urllib.parse import urlparse

load_dotenv()
os.environ.setdefault("USER_AGENT", "LangGraph-RAG-Agent/1.0")

from ui import (
    load_css,
    parse_env_file,
    render_sidebar_brand,
    render_status,
    render_section,
    render_welcome,
    render_setup_intro,
    render_chat_header,
    render_empty_chat,
    render_user_message,
    render_bot_message,
    render_typing,
    render_routing_info,
)
from agent import initialize_agent
from vectorstore import VectorstoreInitError


# ── Page config ───────────────────────────────────────────────

st.set_page_config(
    page_title="RAG Agent · LangGraph",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

load_css()


# ── Session state ─────────────────────────────────────────────

def _init():
    defaults = {
        "messages":      [],
        "agent_ready":   False,
        "app":           None,
        "groq_key":      os.getenv("GROQ_API_KEY", ""),
        "astra_token":   os.getenv("ASTRA_DB_TOKEN", ""),
        "astra_db_id":   os.getenv("ASTRA_DB_ID", ""),
        "_quick":        "",          # queued from suggestion button
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init()


def _ts() -> str:
    return datetime.now().strftime("%H:%M")


def _reset_agent() -> None:
    st.session_state.agent_ready = False
    st.session_state.app = None
    st.session_state.messages = []


def _apply_env_values(parsed: dict) -> list[str]:
    loaded = []
    field_map = {
        "GROQ_API_KEY": "groq_key",
        "ASTRA_DB_TOKEN": "astra_token",
        "ASTRA_DB_ID": "astra_db_id",
    }
    for env_key, state_key in field_map.items():
        if env_key in parsed:
            if parsed[env_key] != st.session_state[state_key]:
                _reset_agent()
            st.session_state[state_key] = parsed[env_key]
            loaded.append(env_key)
    return loaded


def _friendly_error(exc: Exception) -> str:
    if isinstance(exc, VectorstoreInitError):
        return str(exc)
    text = str(exc).lower()
    if "authentication" in text or "unauthorized" in text or "forbidden" in text:
        return "Authentication failed. Check your API keys and AstraDB credentials."
    if "collection" in text and "does not exist" in text:
        return "The previous vector collection was missing. Reinitialize the agent to create a fresh collection."
    if "groq" in text or "api key" in text:
        return "Groq could not be reached. Check your API key and try again."
    if "connect" in text or "connection" in text or "timeout" in text or "network" in text:
        return "A network connection failed. Check your internet connection and that AstraDB/Groq are reachable."
    return "Something went wrong during initialization. Check your credentials, URLs, and network connection."


def _friendly_chat_error(exc: Exception) -> str:
    text = str(exc).lower()
    if "groq" in text or "api key" in text or "authentication" in text:
        return "The chat model could not be reached. Check your Groq API key and try again."
    if "wikipedia" in text:
        return "Wikipedia could not be reached. Try again or ask the question more specifically."
    if "connect" in text or "connection" in text or "timeout" in text or "network" in text:
        return "A network connection failed while answering. Check your internet connection and try again."
    return "Something went wrong while answering. Please try again."


def _valid_url(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


# ── Consume queued quick-query (from suggestion buttons) ──────
#   Must happen before rendering so we can add user msg to state
#   before the message loop below runs.

_quick = st.session_state.pop("_quick", "")

# Chat input appears only after initialization. A disabled sticky input
# looks broken, so the onboarding screen shows setup controls instead.
user_input = ""
if st.session_state.agent_ready:
    user_input = st.chat_input("Ask anything…")

new_query = (user_input or _quick or "").strip()

# Add user message to session state NOW so it's included in
# the render loop below (which runs top-to-bottom).
if new_query and st.session_state.agent_ready:
    st.session_state.messages.append({
        "role":    "user",
        "content": new_query,
        "time":    _ts(),
    })


# ── Sidebar ───────────────────────────────────────────────────

with st.sidebar:

    render_sidebar_brand()
    render_status(st.session_state.agent_ready)
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    if st.session_state.agent_ready:
        if st.button("← Back to Setup", use_container_width=True):
            _reset_agent()
            st.rerun()
        st.divider()

    # ── .env Quick-Setup ─────────────────────────────────────
    render_section("Credential File")

    uploaded = st.file_uploader(
        "Upload .env file",
        type=None,
        label_visibility="visible",
        help="Drag & drop your .env file to auto-fill credentials",
    )
    st.markdown(
        '<div style="font-size:.72rem;color:var(--muted);text-align:center;'
        'margin:-4px 0 6px">Use a .env with GROQ_API_KEY, ASTRA_DB_TOKEN, and ASTRA_DB_ID.</div>',
        unsafe_allow_html=True,
    )

    if uploaded is not None:
        try:
            parsed = parse_env_file(uploaded.read().decode("utf-8"))
            loaded = _apply_env_values(parsed)
            if loaded:
                st.toast(f"✅ Loaded: {', '.join(loaded)}")
            else:
                st.toast("⚠️ No recognized keys found in .env")
        except Exception as e:
            st.toast(f"❌ Could not parse .env: {e}")

    st.divider()

    # ── Credentials ──────────────────────────────────────────
    render_section("Credentials")

    groq_key = st.text_input(
        "Groq API Key",
        value=st.session_state.groq_key,
        type="password",
        placeholder="gsk_…",
        help="Get your free key at console.groq.com",
    )
    if groq_key != st.session_state.groq_key:
        _reset_agent()
    st.session_state.groq_key = groq_key

    astra_token = st.text_input(
        "AstraDB Token",
        value=st.session_state.astra_token,
        type="password",
        placeholder="AstraCS:…",
        help="Your DataStax Astra DB application token",
    )
    if astra_token != st.session_state.astra_token:
        _reset_agent()
    st.session_state.astra_token = astra_token

    astra_db_id = st.text_input(
        "AstraDB ID",
        value=st.session_state.astra_db_id,
        type="password",
        placeholder="xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
        help="Your Astra DB database ID (UUID)",
    )
    if astra_db_id != st.session_state.astra_db_id:
        _reset_agent()
    st.session_state.astra_db_id = astra_db_id

    st.divider()

    # ── Knowledge-base URLs ───────────────────────────────────
    render_section("Knowledge Base URLs")

    urls_raw = st.text_area(
        "One URL per line",
        value="\n".join([
            "https://lilianweng.github.io/posts/2023-06-23-agent/",
            "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
            "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
        ]),
        height=110,
        label_visibility="collapsed",
        help="Pages that will be scraped and embedded into AstraDB.",
    )

    st.divider()

    # ── Init button ───────────────────────────────────────────
    init_btn = st.button(
        "🚀  Initialize Agent",
        use_container_width=True,
        type="primary",
    )

    render_routing_info()

    # ── Clear chat ────────────────────────────────────────────
    if st.session_state.agent_ready and st.session_state.messages:
        st.divider()
        if st.button("🗑️  Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()


# ── Init handler ──────────────────────────────────────────────

def _initialize_agent(key: str, astra_token: str, astra_db_id: str, urls_raw: str, *, surface):
    key = key.strip()
    astra_token = astra_token.strip()
    astra_db_id = astra_db_id.strip()
    urls = [u.strip() for u in urls_raw.split("\n") if u.strip()]

    if not key:
        surface.error("⚠️ Please enter your Groq API Key.")
    elif not astra_token:
        surface.error("⚠️ Please enter your AstraDB token.")
    elif not astra_db_id:
        surface.error("⚠️ Please enter your AstraDB database ID.")
    elif not urls:
        surface.error("⚠️ Add at least one URL.")
    elif invalid_urls := [url for url in urls if not _valid_url(url)]:
        surface.error(f"⚠️ Invalid knowledge-base URL: {invalid_urls[0]}")
    else:
        if hasattr(surface, "__enter__"):
            ctx = surface
        else:
            ctx = st.container()
        with ctx:
            with st.spinner("Scraping pages & rebuilding AstraDB vectors…"):
                try:
                    st.session_state.app = initialize_agent(
                        groq_api_key=key,
                        astra_token=astra_token,
                        astra_db_id=astra_db_id,
                        urls=urls,
                    )
                    st.session_state.agent_ready = True
                    st.session_state.messages = []
                    st.toast("🎉 Agent ready! Start chatting.")
                    st.rerun()
                except Exception as e:
                    _reset_agent()
                    st.error(f"Init failed: {_friendly_error(e)}")


if init_btn:
    key = st.session_state.groq_key.strip()
    astra_token = st.session_state.astra_token.strip()
    astra_db_id = st.session_state.astra_db_id.strip()
    _initialize_agent(key, astra_token, astra_db_id, urls_raw, surface=st.sidebar)


# ── Main content ──────────────────────────────────────────────

if not st.session_state.agent_ready:

    # ── Welcome / onboarding screen ───────────────────────────
    render_welcome()
    render_setup_intro()

    main_uploaded = st.file_uploader(
        "Upload .env file",
        type=None,
        key="main_env_upload",
        help="Drag & drop your .env file to auto-fill credentials.",
    )

    if main_uploaded is not None:
        try:
            parsed = parse_env_file(main_uploaded.read().decode("utf-8"))
            loaded = _apply_env_values(parsed)
            if loaded:
                st.toast(f"✅ Loaded: {', '.join(loaded)}")
            else:
                st.toast("⚠️ No recognized keys found in .env")
        except Exception as e:
            st.error(f"Could not parse .env: {e}")

    c1, c2 = st.columns(2)
    with c1:
        main_groq_key = st.text_input(
            "Groq API Key",
            value=st.session_state.groq_key,
            type="password",
            placeholder="gsk_…",
            key="main_groq_key",
        )
        main_astra_db_id = st.text_input(
            "AstraDB ID",
            value=st.session_state.astra_db_id,
            type="password",
            placeholder="xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
            key="main_astra_db_id",
        )
    with c2:
        main_astra_token = st.text_input(
            "AstraDB Token",
            value=st.session_state.astra_token,
            type="password",
            placeholder="AstraCS:…",
            key="main_astra_token",
        )

    main_urls_raw = st.text_area(
        "Knowledge Base URLs",
        value="\n".join([
            "https://lilianweng.github.io/posts/2023-06-23-agent/",
            "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
            "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
        ]),
        height=108,
        key="main_urls_raw",
        help="One URL per line.",
    )

    if (
        main_groq_key != st.session_state.groq_key
        or main_astra_token != st.session_state.astra_token
        or main_astra_db_id != st.session_state.astra_db_id
    ):
        _reset_agent()
    st.session_state.groq_key = main_groq_key
    st.session_state.astra_token = main_astra_token
    st.session_state.astra_db_id = main_astra_db_id

    if st.button("🚀  Initialize Agent", type="primary", use_container_width=True, key="main_init_btn"):
        _initialize_agent(
            st.session_state.groq_key,
            st.session_state.astra_token,
            st.session_state.astra_db_id,
            main_urls_raw,
            surface=st,
        )

else:

    # ── Chat view ─────────────────────────────────────────────
    nav_col, _ = st.columns([1, 4])
    with nav_col:
        if st.button("Back to Setup", use_container_width=True, key="chat_back_to_setup"):
            _reset_agent()
            st.rerun()

    render_chat_header(len(st.session_state.messages))

    msgs = st.session_state.messages

    if not msgs:
        # ── Empty state + suggestion chips ────────────────────
        render_empty_chat()

        st.markdown(
            '<div style="text-align:center;margin-top:2px;">'
            '<span style="font-size:.73rem;color:var(--muted)">Try asking:</span></div>',
            unsafe_allow_html=True,
        )

        suggestions = [
            "What is prompt engineering?",
            "Explain AI agent memory",
            "Who invented the internet?",
            "Hello! 👋",
        ]
        c1, c2 = st.columns(2)
        for i, sug in enumerate(suggestions):
            col = c1 if i % 2 == 0 else c2
            if col.button(sug, key=f"sug_{i}", use_container_width=True):
                st.session_state["_quick"] = sug
                st.rerun()

    else:
        # ── Message history ────────────────────────────────────
        for msg in msgs:
            if msg["role"] == "user":
                render_user_message(msg["content"], msg.get("time", ""))
            else:
                render_bot_message(
                    msg["content"],
                    msg.get("source", "vectorstore"),
                    msg.get("time", ""),
                )

    # ── Stream agent for new query ─────────────────────────────
    if new_query and st.session_state.agent_ready:

        typing_ph = st.empty()
        with typing_ph:
            render_typing()

        try:
            final_output = None
            for chunk in st.session_state.app.stream({"question": new_query}):
                for _, value in chunk.items():
                    final_output = value

            typing_ph.empty()

            if final_output is None or not final_output.get("generation"):
                st.error("The agent returned no response. Please try again.")
            else:
                answer = final_output["generation"]
                source = final_output.get("source", "vectorstore")
                now    = _ts()

                st.session_state.messages.append({
                    "role":    "bot",
                    "content": answer,
                    "source":  source,
                    "time":    now,
                })

                # Rerun so bot message appears cleanly in the
                # history loop and the header count is accurate.
                st.rerun()

        except Exception as e:
            typing_ph.empty()
            st.error(f"Error: {_friendly_chat_error(e)}")
