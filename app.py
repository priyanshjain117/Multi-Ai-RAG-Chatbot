import streamlit as st

from ui import (
    load_css,
    render_user_message,
    render_bot_message,
)

from agent import initialize_agent


st.set_page_config(
    page_title="LangGraph AI Agent",
    page_icon="🤖",
    layout="wide",
)

load_css()


if "messages" not in st.session_state:
    st.session_state.messages = []

if "agent_ready" not in st.session_state:
    st.session_state.agent_ready = False

if "app" not in st.session_state:
    st.session_state.app = None


with st.sidebar:
    st.markdown("## ⚙️ Configuration")   

    groq_api_key = st.text_input(
        "Groq API Key",
        type="password",
    )

    astra_token = st.text_input(
        "AstraDB Token",
        type="password",
    )

    astra_db_id = st.text_input(
        "AstraDB ID",
    )

    st.divider()

    default_urls = [
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
        "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
    ]

    urls_input = st.text_area(
        "Vectorstore URLs",
        value="\n".join(default_urls),
        height=150,
    )

    load_btn = st.button(
        "🚀 Initialize Agent",
        use_container_width=True,
    )


st.markdown(
    '<div class="main-title">🤖 LangGraph Agentic RAG</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="sub-title">Vectorstore + Wikipedia + Generation</div>',
    unsafe_allow_html=True,
)


if load_btn:

    if not groq_api_key or not astra_token or not astra_db_id:
        st.error("Please fill all credentials.")

    else:
        urls = [
            u.strip()
            for u in urls_input.split("\n")
            if u.strip()
        ]

        with st.spinner("Initializing Agent..."):

            try:
                st.session_state.app = initialize_agent(
                    groq_api_key,
                    astra_token,
                    astra_db_id,
                    urls,
                )

                st.session_state.agent_ready = True

                st.success("Agent initialized successfully!")

            except Exception as e:
                st.error(f"Initialization failed: {e}")


if not st.session_state.agent_ready:

    st.info("Initialize the agent from sidebar.")

else:

    for msg in st.session_state.messages:

        if msg["role"] == "user":
            render_user_message(msg["content"])

        else:
            render_bot_message(
                msg["content"],
                msg["source"],
            )

    col1, col2 = st.columns([5, 1])

    with col1:
        user_input = st.text_input(
            "Ask anything...",
            label_visibility="collapsed",
        )

    with col2:
        send = st.button(
            "Send →",
            use_container_width=True,
        )

    if send and user_input.strip():

        query = user_input.strip()

        st.session_state.messages.append({
            "role": "user",
            "content": query,
        })

        with st.spinner("Thinking..."):

            try:
                final_output = None

                for output in st.session_state.app.stream({
                    "question": query
                }):
                    for _, value in output.items():
                        final_output = value

                answer = final_output["generation"]

                source = final_output["source"]

                st.session_state.messages.append({
                    "role": "bot",
                    "content": answer,
                    "source": source,
                })

                st.rerun()

            except Exception as e:
                st.error(f"Error: {e}")

    if st.session_state.messages:

        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()