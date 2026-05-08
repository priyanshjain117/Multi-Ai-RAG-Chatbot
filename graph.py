from typing import List
from typing_extensions import TypedDict

from langchain_core.documents import Document

from langchain_groq import ChatGroq

from langgraph.graph import (
    START,
    END,
    StateGraph,
)


class GraphState(TypedDict):

    question: str

    generation: str

    documents: List[str]

    source: str


def build_graph(
    retriever,
    wiki_tool,
    question_router,
    groq_api_key,
):

    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name="llama-3.1-8b-instant",
    )


    def retrieve(state):

        question = state["question"]

        documents = retriever.invoke(question)

        return {
            "documents": documents,
            "question": question,
            "source": "vectorstore",
        }


    def wiki_search(state):

        question = state["question"]

        docs = wiki_tool.invoke({
            "query": question
        })

        wiki_results = Document(
            page_content=docs
        )

        return {
            "documents": [wiki_results],
            "question": question,
            "source": "wiki_search",
        }


    def route_question(state):

        question = state["question"]

        source = question_router.invoke({
            "question": question
        })

        return source.datasource


    def generate(state):

        question = state["question"]

        documents = state["documents"]

        context = "\n\n".join([
            doc.page_content
            for doc in documents
        ])

        prompt = f"""
        You are a helpful AI assistant.

        Use ONLY the provided context
        to answer the question.

        If the answer is not in context,
        say you don't know.

        Context:
        {context}

        Question:
        {question}
        """

        response = llm.invoke(prompt)

        return {
            "documents": documents,
            "question": question,
            "generation": response.content,
            "source": state["source"],
        }


    workflow = StateGraph(GraphState)

    workflow.add_node(
        "retrieve",
        retrieve,
    )

    workflow.add_node(
        "wiki_search",
        wiki_search,
    )

    workflow.add_node(
        "generate",
        generate,
    )

    workflow.add_conditional_edges(
        START,
        route_question,
        {
            "vectorstore": "retrieve",
            "wiki_search": "wiki_search",
        },
    )

    workflow.add_edge(
        "retrieve",
        "generate",
    )

    workflow.add_edge(
        "wiki_search",
        "generate",
    )

    workflow.add_edge(
        "generate",
        END,
    )

    return workflow.compile()