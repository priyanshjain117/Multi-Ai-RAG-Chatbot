from typing import List
from typing_extensions import TypedDict
import re

import os
os.environ.setdefault("USER_AGENT", "LangGraph-RAG-Agent/1.0")

from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langgraph.graph import START, END, StateGraph


class GraphState(TypedDict):
    question:   str
    generation: str
    documents:  List[Document]
    source:     str


def build_graph(retriever, wiki_tool, question_router, groq_api_key: str):

    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name="llama-3.1-8b-instant",
    )

    # ── Nodes ────────────────────────────────────────────────

    def retrieve(state: GraphState) -> dict:
        docs = retriever.invoke(state["question"]) or []
        return {
            "documents": docs,
            "question":  state["question"],
            "source":    "vectorstore",
        }

    def wiki_search(state: GraphState) -> dict:
        try:
            docs = wiki_tool.invoke({"query": state["question"]}) or ""
        except Exception:
            prompt = (
                "Answer this factual question directly and concisely. "
                "If there are multiple accepted contributors, mention the main names.\n\n"
                f"Question: {state['question']}\nAnswer:"
            )
            response = llm.invoke(prompt)
            return {
                "documents":  [],
                "question":   state["question"],
                "generation": response.content,
                "source":     "general_chat",
            }

        return {
            "documents": [Document(page_content=docs)],
            "question":  state["question"],
            "source":    "wiki_search",
        }

    def general_chat(state: GraphState) -> dict:
        """Handle greetings, small talk, and casual conversation directly."""
        prompt = (
            "You are a warm, friendly, and helpful AI assistant. "
            "Respond naturally and concisely to the user's message. "
            "Be engaging and personable — no need to look anything up.\n\n"
            f"User: {state['question']}\nAssistant:"
        )
        response = llm.invoke(prompt)
        return {
            "documents":  [],
            "question":   state["question"],
            "generation": response.content,
            "source":     "general_chat",
        }

    def generate(state: GraphState) -> dict:
        if state.get("generation"):
            return state

        context = "\n\n".join(doc.page_content for doc in state["documents"])
        if state["source"] == "wiki_search":
            prompt = (
                "You are a precise factual assistant.\n"
                "Use the Wikipedia context when it answers the question. "
                "If the Wikipedia context is irrelevant or too narrow, answer from general knowledge instead. "
                "Keep the answer concise and do not mention the context.\n\n"
                f"Wikipedia context:\n{context}\n\n"
                f"Question: {state['question']}\nAnswer:"
            )
        else:
            prompt = (
                "You are a precise and helpful AI assistant.\n"
                "Answer using the provided knowledge-base context.\n"
                "Synthesize the best answer from relevant passages.\n"
                "If the context truly does not contain the answer, say you don't know.\n\n"
                f"Knowledge-base context:\n{context}\n\n"
                f"Question: {state['question']}\nAnswer:"
            )
        response = llm.invoke(prompt)
        return {
            "documents":  state["documents"],
            "question":   state["question"],
            "generation": response.content,
            "source":     state["source"],
        }

    def route_question(state: GraphState) -> str:
        question = state["question"].strip()
        if re.fullmatch(r"(?i)(hi|hello|hey|thanks|thank you|bye|goodbye|good morning|good afternoon|good evening)[!. ]*", question):
            return "general_chat"

        try:
            result = question_router.invoke({"question": question})
        except Exception:
            return "wiki_search"

        datasource = getattr(result, "datasource", None)
        if datasource in {"vectorstore", "wiki_search", "general_chat"}:
            return datasource
        return "wiki_search"

    # ── Graph ────────────────────────────────────────────────

    workflow = StateGraph(GraphState)

    workflow.add_node("retrieve",     retrieve)
    workflow.add_node("wiki_search",  wiki_search)
    workflow.add_node("generate",     generate)
    workflow.add_node("general_chat", general_chat)

    workflow.add_conditional_edges(
        START,
        route_question,
        {
            "vectorstore":  "retrieve",
            "wiki_search":  "wiki_search",
            "general_chat": "general_chat",
        },
    )

    workflow.add_edge("retrieve",     "generate")
    workflow.add_edge("wiki_search",  "generate")
    workflow.add_edge("general_chat", END)          # no generate step needed
    workflow.add_edge("generate",     END)

    return workflow.compile()
