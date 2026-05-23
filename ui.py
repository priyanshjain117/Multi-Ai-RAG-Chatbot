"""Streamlit UI helpers for the RAG Agent app."""

import html as _html
import re

import streamlit as st


def _md(text: str) -> str:
    """Convert a small, escaped Markdown subset to HTML."""
    t = _html.escape(text)
    t = re.sub(
        r"```(\w*)\n?(.*?)```",
        lambda m: (
            f'<pre><code class="lang-{m.group(1) or "text"}">'
            f"{m.group(2).strip()}</code></pre>"
        ),
        t,
        flags=re.DOTALL,
    )
    t = re.sub(r"`([^`\n]+)`", r"<code>\1</code>", t)
    t = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", t)
    t = re.sub(r"\*(.+?)\*", r"<em>\1</em>", t)

    lines, out, in_list = t.split("\n"), [], False
    for line in lines:
        m = re.match(r"^[-*•] (.+)$", line)
        if m:
            if not in_list:
                out.append("<ul>")
                in_list = True
            out.append(f"<li>{m.group(1)}</li>")
        else:
            if in_list:
                out.append("</ul>")
                in_list = False
            out.append(line)
    if in_list:
        out.append("</ul>")

    segs = re.split(r"(<pre>.*?</pre>)", "\n".join(out), flags=re.DOTALL)
    return "".join(
        s if i % 2 == 1 else s.replace("\n", "<br>")
        for i, s in enumerate(segs)
    )


def parse_env_file(content: str) -> dict:
    """Parse recognized .env keys without logging or displaying values."""
    aliases = {
        "ASTRA_DB_TOKEN": "ASTRA_DB_TOKEN",
        "ASTRADB_TOKEN": "ASTRA_DB_TOKEN",
        "ASTRA_DB_APPLICATION_TOKEN": "ASTRA_DB_TOKEN",
        "ASTRA_DB_ID": "ASTRA_DB_ID",
        "ASTRADB_ID": "ASTRA_DB_ID",
        "ASTRA_DB_DATABASE_ID": "ASTRA_DB_ID",
        "GROQ_API_KEY": "GROQ_API_KEY",
    }
    result = {}
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line.removeprefix("export ").strip()
        if "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        canonical_key = aliases.get(key)
        if canonical_key:
            result[canonical_key] = val.strip().strip('"').strip("'")
    return result


def _chip(source: str) -> str:
    return {
        "vectorstore": '<span class="src-chip src-vec">Knowledge Base</span>',
        "wiki_search": '<span class="src-chip src-wiki">Wikipedia</span>',
        "general_chat": '<span class="src-chip src-chat">Chat</span>',
    }.get(source, '<span class="src-chip src-vec">Knowledge Base</span>')


_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
  --bg: #fffaf5;
  --panel: #ffffff;
  --panel-soft: #fff7ed;
  --input: #fffdf9;
  --border: rgba(234, 88, 12, 0.16);
  --border-strong: rgba(234, 88, 12, 0.34);
  --primary: #f97316;
  --primary-dark: #ea580c;
  --amber: #f59e0b;
  --sky: #0284c7;
  --green: #059669;
  --text: #1f2933;
  --muted: #6b7280;
  --faint: #9ca3af;
  --shadow: 0 18px 45px rgba(124, 45, 18, 0.10);
  --radius: 8px;
}

*, *::before, *::after { box-sizing: border-box; }

html, body, .stApp {
  font-family: 'Inter', system-ui, sans-serif !important;
  background: linear-gradient(180deg, #fffaf5 0%, #ffffff 42%, #fff7ed 100%) !important;
  color: var(--text) !important;
}

#MainMenu, footer { visibility: hidden !important; }
header {
  background: transparent !important;
  box-shadow: none !important;
}
[data-testid="stToolbar"],
[data-testid="stDeployButton"],
[data-testid="stDecoration"] { display: none !important; }

.main .block-container {
  max-width: 980px !important;
  padding: 1.35rem 2rem 5.75rem !important;
}

[data-testid="stSidebar"] {
  background: #ffffff !important;
  border-right: 1px solid var(--border) !important;
  box-shadow: 10px 0 35px rgba(124, 45, 18, 0.04);
}
[data-testid="stSidebar"] .block-container {
  padding: 1.25rem 1rem 1.5rem !important;
}

h1, h2, h3, h4 {
  color: var(--text) !important;
  letter-spacing: 0 !important;
}

.stTextInput input,
.stTextArea textarea {
  background: var(--input) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
  color: var(--text) !important;
  font-size: .9rem !important;
  box-shadow: none !important;
}
.stTextInput input:focus,
.stTextArea textarea:focus {
  border-color: var(--primary) !important;
  box-shadow: 0 0 0 3px rgba(249, 115, 22, .14) !important;
}
.stTextInput label,
.stTextArea label,
.stFileUploader label {
  color: var(--muted) !important;
  font-size: .68rem !important;
  font-weight: 700 !important;
  letter-spacing: .06em !important;
  text-transform: uppercase !important;
}

button[kind="primary"],
.stButton > button[kind="primary"] {
  background: linear-gradient(135deg, var(--primary), var(--primary-dark)) !important;
  border: 0 !important;
  border-radius: var(--radius) !important;
  color: white !important;
  font-weight: 700 !important;
  padding: .68rem 1.1rem !important;
  box-shadow: 0 10px 26px rgba(234, 88, 12, .22) !important;
}
.stButton > button:not([kind="primary"]) {
  background: #ffffff !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
  color: var(--text) !important;
  font-weight: 600 !important;
}
.stButton > button:hover {
  border-color: var(--border-strong) !important;
  transform: translateY(-1px);
}

[data-testid="stFileUploader"] section {
  background: var(--panel-soft) !important;
  border: 1.5px dashed var(--border-strong) !important;
  border-radius: var(--radius) !important;
}
[data-testid="stFileUploader"] section *,
[data-testid="stFileUploader"] small,
[data-testid="stFileUploader"] span,
[data-testid="stFileUploader"] p {
  color: var(--muted) !important;
}
[data-testid="stFileUploader"] button {
  background: #ffffff !important;
  border: 1px solid var(--border-strong) !important;
  color: var(--text) !important;
  border-radius: var(--radius) !important;
}
[data-testid="stFileUploader"] svg {
  color: var(--primary-dark) !important;
  fill: none !important;
  stroke: currentColor !important;
}

hr {
  border: 0 !important;
  border-top: 1px solid var(--border) !important;
  margin: .75rem 0 !important;
}

[data-testid="stAlert"],
[data-testid="stToast"] {
  background: #ffffff !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
  color: var(--text) !important;
}
[data-testid="stSpinner"] svg { color: var(--primary) !important; }

[data-testid="stBottom"],
[data-testid="stBottomBlockContainer"],
[data-testid="stChatInputContainer"] {
  background: linear-gradient(180deg, rgba(255, 250, 245, 0), #fffaf5 34%, #fff7ed 100%) !important;
  border-top: 1px solid rgba(234, 88, 12, 0.10) !important;
  box-shadow: 0 -14px 34px rgba(124, 45, 18, 0.06) !important;
}
[data-testid="stBottom"] > div,
[data-testid="stBottomBlockContainer"] > div,
[data-testid="stChatInputContainer"] > div {
  background: transparent !important;
}
[data-testid="stChatInput"] {
  background: #ffffff !important;
  border: 1px solid var(--border-strong) !important;
  border-radius: 10px !important;
  box-shadow: 0 12px 34px rgba(124, 45, 18, 0.10) !important;
  max-width: 980px !important;
  margin: 0 auto .55rem !important;
}
[data-testid="stChatInput"] *,
[data-testid="stChatInput"] [data-baseweb="textarea"],
[data-testid="stChatInput"] [data-baseweb="base-input"] {
  background-color: transparent !important;
}
[data-testid="stChatInput"]:focus-within {
  border-color: var(--primary) !important;
  box-shadow: 0 0 0 3px rgba(249, 115, 22, .13), 0 12px 34px rgba(124, 45, 18, 0.10) !important;
}
[data-testid="stChatInput"] textarea {
  background: transparent !important;
  color: var(--text) !important;
  font-family: 'Inter', system-ui, sans-serif !important;
}
[data-testid="stChatInput"] textarea:disabled {
  -webkit-text-fill-color: var(--faint) !important;
  opacity: 1 !important;
}
[data-testid="stChatInput"] textarea::placeholder {
  color: var(--faint) !important;
}
[data-testid="stChatInput"] button {
  background: linear-gradient(135deg, var(--primary), var(--primary-dark)) !important;
  border: 1px solid rgba(234, 88, 12, .22) !important;
  color: #ffffff !important;
  border-radius: 8px !important;
  min-width: 38px !important;
  min-height: 38px !important;
  display: inline-grid !important;
  place-items: center !important;
  opacity: 1 !important;
}
[data-testid="stChatInput"] button::before {
  content: "\\2191";
  color: #ffffff !important;
  font-size: 1rem !important;
  font-weight: 800 !important;
  line-height: 1 !important;
}
[data-testid="stChatInput"] button svg {
  color: #ffffff !important;
  fill: none !important;
  stroke: currentColor !important;
  width: 18px !important;
  height: 18px !important;
  opacity: 0 !important;
  position: absolute !important;
}
[data-testid="stChatInput"] button:disabled {
  background: #f9fafb !important;
  border-color: var(--border) !important;
  color: var(--faint) !important;
}
[data-testid="stChatInput"] button:disabled::before {
  color: var(--faint) !important;
}
[data-testid="stChatInput"] button:disabled svg {
  color: var(--faint) !important;
}

@keyframes fadeInUp {
  from { opacity: 0; transform: translateY(10px); }
  to { opacity: 1; transform: translateY(0); }
}
@keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
@keyframes pulseDot {
  0%, 80%, 100% { transform: scale(.55); opacity: .35; }
  40% { transform: scale(1); opacity: 1; }
}
@keyframes statusPulse {
  0%, 100% { box-shadow: 0 0 4px var(--green); }
  50% { box-shadow: 0 0 14px rgba(5, 150, 105, .45); }
}

.sb-brand {
  display: flex;
  align-items: center;
  gap: 11px;
  padding: 8px 0 18px;
  border-bottom: 1px solid var(--border);
  margin-bottom: 16px;
}
.sb-icon, .w-logo, .msg-av.bot {
  background: linear-gradient(135deg, var(--primary), var(--primary-dark));
  color: #fff;
}
.sb-icon {
  width: 38px;
  height: 38px;
  border-radius: 8px;
  display: grid;
  place-items: center;
  font-size: 18px;
}
.sb-name { font-size: .96rem; font-weight: 800; color: var(--text); letter-spacing: 0; }
.sb-tag { font-size: .68rem; color: var(--muted); margin-top: 1px; }
.sb-lbl {
  color: var(--muted);
  font-size: .68rem;
  font-weight: 800;
  letter-spacing: .06em;
  text-transform: uppercase;
  margin: 15px 0 8px;
}

.st-badge {
  display: inline-flex;
  align-items: center;
  gap: 7px;
  border-radius: 999px;
  padding: 5px 10px;
  font-size: .75rem;
  font-weight: 700;
  border: 1px solid var(--border);
  background: var(--panel-soft);
  color: var(--muted);
}
.st-badge.ready { color: var(--green); background: rgba(5, 150, 105, .08); border-color: rgba(5, 150, 105, .18); }
.st-dot { width: 7px; height: 7px; border-radius: 50%; background: var(--faint); }
.st-dot.ready { background: var(--green); animation: statusPulse 2s infinite; }

.w-wrap {
  display: flex;
  flex-direction: column;
  align-items: center;
  min-height: 56vh;
  padding: 56px 20px 32px;
  text-align: center;
  animation: fadeInUp .45s ease;
}
.w-logo {
  width: 82px;
  height: 82px;
  border-radius: 18px;
  display: grid;
  place-items: center;
  font-size: 38px;
  margin-bottom: 24px;
  box-shadow: 0 18px 45px rgba(234, 88, 12, .22);
}
.w-title {
  font-size: 2.2rem;
  line-height: 1.12;
  font-weight: 800;
  color: var(--text);
  letter-spacing: 0;
  margin-bottom: 10px;
}
.w-sub {
  max-width: 560px;
  color: var(--muted);
  font-size: .98rem;
  line-height: 1.75;
  margin-bottom: 26px;
}
.feat-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 12px;
  width: min(100%, 560px);
  margin-bottom: 22px;
}
.feat-card {
  background: #ffffff;
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 18px 12px;
  box-shadow: 0 8px 24px rgba(124, 45, 18, .05);
}
.feat-icon { font-size: 24px; margin-bottom: 8px; }
.feat-lbl { color: var(--muted); font-size: .72rem; font-weight: 800; text-transform: uppercase; letter-spacing: .04em; }
.w-hint {
  background: var(--panel-soft);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  color: var(--muted);
  padding: 13px 16px;
  font-size: .86rem;
  line-height: 1.65;
}
.w-hint strong, .w-hint em { color: var(--primary-dark); font-style: normal; }

.setup-panel {
  width: min(100%, 760px);
  margin: 4px auto 34px;
  padding: 18px;
  background: #ffffff;
  border: 1px solid var(--border);
  border-radius: var(--radius);
  box-shadow: 0 14px 36px rgba(124, 45, 18, 0.07);
}
.setup-title {
  color: var(--text);
  font-size: 1rem;
  font-weight: 800;
  margin-bottom: 3px;
}
.setup-sub {
  color: var(--muted);
  font-size: .82rem;
  line-height: 1.55;
  margin-bottom: 14px;
}
.setup-panel [data-testid="stVerticalBlock"] {
  gap: .7rem;
}

.chat-hdr {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 10px 0 16px;
  border-bottom: 1px solid var(--border);
  margin-bottom: 18px;
}
.chat-hdr-name { font-size: 1rem; font-weight: 800; color: var(--text); }
.chat-hdr-sub { font-size: .72rem; color: var(--muted); margin-top: 3px; }

.msg-row {
  display: flex;
  gap: 12px;
  margin-bottom: 10px;
  animation: fadeInUp .25s ease;
}
.msg-row.user { flex-direction: row-reverse; }
.msg-av {
  width: 34px;
  height: 34px;
  border-radius: 50%;
  display: grid;
  place-items: center;
  flex-shrink: 0;
  margin-top: 3px;
  font-size: 14px;
}
.msg-av.user { background: #ffffff; border: 1px solid var(--border); }
.msg-av.chat { background: linear-gradient(135deg, var(--sky), #38bdf8); color: #fff; }
.msg-body { flex: 1; min-width: 0; max-width: calc(100% - 48px); }
.msg-row.user .msg-body { display: flex; flex-direction: column; align-items: flex-end; }
.msg-meta {
  display: flex;
  align-items: center;
  gap: 7px;
  margin-bottom: 5px;
  color: var(--faint);
  font-size: .7rem;
}
.msg-meta.user { flex-direction: row-reverse; }
.msg-who { color: var(--muted); font-size: .66rem; font-weight: 800; text-transform: uppercase; letter-spacing: .04em; }
.msg-time { opacity: .75; }
.src-chip {
  border-radius: 999px;
  padding: 2px 7px;
  font-size: .62rem;
  font-weight: 800;
  text-transform: uppercase;
  letter-spacing: .04em;
}
.src-vec { color: var(--green); background: rgba(5, 150, 105, .08); border: 1px solid rgba(5, 150, 105, .16); }
.src-wiki { color: var(--sky); background: rgba(2, 132, 199, .08); border: 1px solid rgba(2, 132, 199, .16); }
.src-chat { color: var(--primary-dark); background: rgba(249, 115, 22, .10); border: 1px solid rgba(249, 115, 22, .18); }
.msg-bubble {
  padding: 12px 15px;
  border-radius: 12px;
  font-size: .91rem;
  line-height: 1.68;
  word-wrap: break-word;
}
.msg-bubble.user {
  background: linear-gradient(135deg, #ffedd5, #fff7ed);
  border: 1px solid var(--border-strong);
  border-top-right-radius: 4px;
}
.msg-bubble.bot {
  background: #ffffff;
  border: 1px solid var(--border);
  border-top-left-radius: 4px;
  box-shadow: 0 8px 24px rgba(124, 45, 18, .05);
}
.msg-bubble.bot code {
  font-family: 'JetBrains Mono', monospace;
  background: #fff7ed;
  color: var(--primary-dark);
  border-radius: 4px;
  padding: 2px 5px;
  font-size: .84em;
}
.msg-bubble.bot pre {
  background: #1f2933;
  color: #f9fafb;
  border-radius: var(--radius);
  padding: 13px;
  overflow-x: auto;
}
.msg-bubble.bot pre code { background: transparent; color: inherit; padding: 0; }

.typing-row {
  display: flex;
  gap: 12px;
  align-items: center;
  animation: fadeIn .25s ease;
}
.typing-dots {
  display: flex;
  gap: 5px;
  background: #ffffff;
  border: 1px solid var(--border);
  border-radius: 12px;
  border-top-left-radius: 4px;
  padding: 12px 15px;
}
.t-dot {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: var(--primary);
  animation: pulseDot 1.4s infinite;
}
.t-dot:nth-child(2) { animation-delay: .2s; }
.t-dot:nth-child(3) { animation-delay: .4s; }

.empty-st {
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 56px 20px 24px;
  text-align: center;
}
.empty-icon { font-size: 42px; opacity: .45; margin-bottom: 12px; }
.empty-txt { color: var(--muted); font-size: .9rem; max-width: 300px; line-height: 1.65; }
.info-card {
  background: var(--panel-soft);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  color: var(--muted);
  font-size: .78rem;
  line-height: 1.72;
  margin-top: 10px;
  padding: 12px 13px;
}
.info-card b { color: var(--text); }
.ic-chip { display: inline-block; border-radius: 999px; padding: 1px 7px; font-size: .66rem; font-weight: 800; }
.ic-v { background: rgba(5, 150, 105, .08); color: var(--green); }
.ic-w { background: rgba(2, 132, 199, .08); color: var(--sky); }
.ic-c { background: rgba(249, 115, 22, .10); color: var(--primary-dark); }

@media (max-width: 760px) {
  .main .block-container { padding: 1rem 1rem 5.5rem !important; }
  .feat-grid { grid-template-columns: 1fr; }
  .w-title { font-size: 1.75rem; }
  .msg-body { max-width: calc(100% - 44px); }
}
</style>
"""


def load_css():
    st.markdown(_CSS, unsafe_allow_html=True)


def render_sidebar_brand():
    st.markdown(
        """
<div class="sb-brand">
  <div class="sb-icon">AI</div>
  <div>
    <div class="sb-name">RAG Agent</div>
    <div class="sb-tag">LangGraph + AstraDB</div>
  </div>
</div>""",
        unsafe_allow_html=True,
    )


def render_status(ready: bool):
    cls = "ready" if ready else "idle"
    text = "Agent Ready" if ready else "Not Initialized"
    st.markdown(
        f'<div class="st-badge {cls}"><div class="st-dot {cls}"></div>{text}</div>',
        unsafe_allow_html=True,
    )


def render_section(label: str):
    st.markdown(f'<div class="sb-lbl">{_html.escape(label)}</div>', unsafe_allow_html=True)


def render_welcome():
    st.markdown(
        """
<div class="w-wrap">
  <div class="w-logo">AI</div>
  <div class="w-title">LangGraph RAG Agent</div>
  <div class="w-sub">
    A polished multi-source assistant that routes each question to AstraDB retrieval,
    Wikipedia, or direct conversation.
  </div>
  <div class="feat-grid">
    <div class="feat-card"><div class="feat-icon">DB</div><div class="feat-lbl">Astra Vectors</div></div>
    <div class="feat-card"><div class="feat-icon">WK</div><div class="feat-lbl">Wikipedia</div></div>
    <div class="feat-card"><div class="feat-icon">LLM</div><div class="feat-lbl">Groq Chat</div></div>
  </div>
  <div class="w-hint">
    Add credentials below, upload an optional <strong>.env</strong>,
    then <em>Initialize Agent</em>. The chat unlocks after setup.
  </div>
</div>""",
        unsafe_allow_html=True,
    )


def render_setup_intro():
    st.markdown(
        """
<div class="setup-panel">
  <div class="setup-title">Setup</div>
  <div class="setup-sub">
    Paste your keys here or upload a .env file. Keep one knowledge-base URL per line.
  </div>
</div>""",
        unsafe_allow_html=True,
    )


def render_chat_header(n: int):
    label = f"{n} message{'s' if n != 1 else ''}"
    st.markdown(
        f"""
<div class="chat-hdr">
  <div>
    <div class="chat-hdr-name">RAG Assistant</div>
    <div class="chat-hdr-sub">{label} this session</div>
  </div>
</div>""",
        unsafe_allow_html=True,
    )


def render_empty_chat():
    st.markdown(
        """
<div class="empty-st">
  <div class="empty-icon">Ask</div>
  <div class="empty-txt">Start a conversation or pick a suggestion below.</div>
</div>""",
        unsafe_allow_html=True,
    )


def render_user_message(text: str, ts: str = ""):
    safe = _html.escape(text)
    t = f'<span class="msg-time">{_html.escape(ts)}</span>' if ts else ""
    st.markdown(
        f"""
<div class="msg-row user">
  <div class="msg-av user">You</div>
  <div class="msg-body">
    <div class="msg-meta user"><span class="msg-who">You</span>{t}</div>
    <div class="msg-bubble user">{safe}</div>
  </div>
</div>""",
        unsafe_allow_html=True,
    )


def render_bot_message(text: str, source: str, ts: str = ""):
    body = _md(text)
    chip = _chip(source)
    t = f'<span class="msg-time">{_html.escape(ts)}</span>' if ts else ""
    av_cls = "chat" if source == "general_chat" else "bot"
    av_icon = "AI" if source != "general_chat" else "Hi"
    st.markdown(
        f"""
<div class="msg-row">
  <div class="msg-av {av_cls}">{av_icon}</div>
  <div class="msg-body">
    <div class="msg-meta"><span class="msg-who">Assistant</span>{chip}{t}</div>
    <div class="msg-bubble bot">{body}</div>
  </div>
</div>""",
        unsafe_allow_html=True,
    )


def render_typing():
    st.markdown(
        """
<div class="typing-row">
  <div class="msg-av bot">AI</div>
  <div class="typing-dots">
    <div class="t-dot"></div><div class="t-dot"></div><div class="t-dot"></div>
  </div>
  <span style="font-size:.72rem;color:var(--muted)">Thinking...</span>
</div>""",
        unsafe_allow_html=True,
    )


def render_routing_info():
    st.markdown(
        """
<div class="info-card">
  <b>Smart Query Routing</b><br><br>
  <span class="ic-chip ic-v">Vector</span> Knowledge-base topics<br>
  <span class="ic-chip ic-w">Wiki</span> General factual questions<br>
  <span class="ic-chip ic-c">Chat</span> Greetings and small talk
</div>""",
        unsafe_allow_html=True,
    )
