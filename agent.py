from vectorstore import create_vectorstore

from router import create_router

from wiki_tool import create_wiki_tool

from graph import build_graph


def initialize_agent(
    groq_api_key,
    urls,
):

    vectorstore = create_vectorstore(urls)

    retriever = vectorstore.as_retriever()

    router = create_router(
        groq_api_key,
        urls,
    )

    wiki_tool = create_wiki_tool()

    app = build_graph(
        retriever,
        wiki_tool,
        router,
        groq_api_key,
    )

    return app