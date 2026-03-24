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

    # ── Agent Initialization ──────────────────────────────────────────────────────
def initialize_agent(groq_key, astra_tok, astra_id, urls):
    """Build and cache the LangGraph agent."""
    import cassio
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.document_loaders import WebBaseLoader
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores.cassandra import Cassandra
    from langchain_core.prompts import ChatPromptTemplate
    from pydantic import BaseModel, Field
    from langchain_groq import ChatGroq
    from langchain_community.utilities import WikipediaAPIWrapper
    from langchain_community.tools import WikipediaQueryRun
    from langchain_core.documents import Document
    from langgraph.graph import END, StateGraph, START

    # AstraDB init
    cassio.init(token=astra_tok, database_id=astra_id)

    # Load & split docs
    docs = [WebBaseLoader(url).load() for url in urls]
    docs_list = [item for sublist in docs for item in sublist]
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=500, chunk_overlap=0)
    doc_splits = splitter.split_documents(docs_list)

    # Embeddings + vectorstore
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    astra_vs = Cassandra(embedding=embeddings, table_name="qa_mini_demo", session=None, keyspace=None)
    astra_vs.delete_collection()
    astra_vs.add_documents(doc_splits)
    retriever = astra_vs.as_retriever()

    # Router LLM
    os.environ["GROQ_API_KEY"] = groq_key
    llm = ChatGroq(groq_api_key=groq_key, model_name="llama-3.1-8b-instant")

    class RouteQuery(BaseModel):
        datasource: Literal["vectorstore", "wiki_search"] = Field(
            ..., description="Route to wikipedia or vectorstore."
        )

    structured_router = llm.with_structured_output(RouteQuery)
    sys_prompt = (
        "You are an expert at routing a user question to a vectorstore or wikipedia. "
        "The vectorstore contains documents related to agents, prompt engineering, and adversarial attacks. "
        "Use the vectorstore for questions on these topics. Otherwise, use wiki-search."
    )
    route_prompt = ChatPromptTemplate.from_messages([("system", sys_prompt), ("human", "{question}")])
    question_router = route_prompt | structured_router

    # Wikipedia tool
    api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=500)
    wiki = WikipediaQueryRun(api_wrapper=api_wrapper)

    # Graph state
    class GraphState(TypedDict):
        question: str
        generation: str
        documents: List[str]
        source: str

    def retrieve(state):
        question = state["question"]
        documents = retriever.invoke(question)
        return {"documents": documents, "question": question, "source": "vectorstore"}

    def wiki_search(state):
        question = state["question"]
        docs = wiki.invoke({"query": question})
        wiki_results = Document(page_content=docs)
        return {"documents": [wiki_results], "question": question, "source": "wiki_search"}

    def route_question(state):
        question = state["question"]
        source = question_router.invoke({"question": question})
        return source.datasource

    workflow = StateGraph(GraphState)
    workflow.add_node("wiki_search", wiki_search)
    workflow.add_node("retrieve", retrieve)
    workflow.add_conditional_edges(
        START,
        route_question,
        {"wiki_search": "wiki_search", "vectorstore": "retrieve"},
    )
    workflow.add_edge("retrieve", END)
    workflow.add_edge("wiki_search", END)

    return workflow.compile()

if load_btn:
        if not groq_api_key or not astra_token or not astra_db_id:
            st.error("Please fill in all API credentials in the sidebar.")
        else:
            urls = [u.strip() for u in urls_input.strip().split("\n") if u.strip()]
            with st.spinner("🔧 Initializing agent — loading docs, building vectorstore..."):
                try:
                    st.session_state.app = initialize_agent(groq_api_key, astra_token, astra_db_id, urls)
                    st.session_state.agent_ready = True
                    st.success("✅ Agent ready! Start chatting below.")
                except Exception as e:
                    st.error(f"Initialization failed: {e}")

# ── Chat Interface ────────────────────────────────────────────────────────────
if not st.session_state.agent_ready:
    st.info("👈 Enter your API credentials in the sidebar and click **Initialize Agent** to begin.")
else:
    # Render history
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f'<div class="chat-bubble-user">🧑 {msg["content"]}</div>', unsafe_allow_html=True)
        else:
            badge_class = "badge-wiki" if msg.get("source") == "wiki_search" else "badge-vector"
            badge_label = "📖 WIKIPEDIA" if msg.get("source") == "wiki_search" else "🗄️ VECTORSTORE"
            st.markdown(f"""
            <div class="chat-bubble-bot">
              <span class="source-badge {badge_class}">{badge_label}</span><br>
              🤖 {msg["content"]}
            </div>""", unsafe_allow_html=True)
# Input row
    col1, col2 = st.columns([5, 1])
    with col1:
        user_input = st.text_input("Ask anything...", key="user_query", label_visibility="collapsed",
                                   placeholder="e.g. What is agent memory? / Who is Elon Musk?")
    with col2:
        send = st.button("Send →", use_container_width=True)

    if send and user_input.strip():
        query = user_input.strip()
        st.session_state.messages.append({"role": "user", "content": query})

        with st.spinner("🔍 Routing & retrieving..."):
            try:
                final_output = None
                used_source = "vectorstore"
                for output in st.session_state.app.stream({"question": query}):
                    for key, value in output.items():
                        final_output = value
                        used_source = value.get("source", "vectorstore")

                # Extract answer text
                docs = final_output.get("documents", [])
                if docs:
                    if hasattr(docs[0], "page_content"):
                        answer = docs[0].page_content[:800]
                    elif isinstance(docs[0], dict):
                        answer = docs[0].get("page_content", str(docs[0]))[:800]
                    else:
                        answer = str(docs[0])[:800]
                else:
                    answer = "No relevant information found."

                st.session_state.messages.append({"role": "bot", "content": answer, "source": used_source})
                st.rerun()

            except Exception as e:
                st.error(f"Error: {e}")

    # Clear chat button
    if st.session_state.messages:
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()