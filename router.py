import re

from typing import Literal

from pydantic import BaseModel, Field

from langchain_core.prompts import (
    ChatPromptTemplate,
)

from langchain_groq import ChatGroq


class RouteQuery(BaseModel):

    datasource: Literal[
        "vectorstore",
        "wiki_search"
    ] = Field(
        ...,
        description="Route to vectorstore or wiki_search."
    )


def extract_topic_from_url(url: str):

    slug = url.rstrip("/").split("/")[-1]

    slug = re.sub(
        r"^\d{4}-\d{2}-\d{2}-",
        "",
        slug,
    )

    return slug.replace("-", " ")


def create_router(
    groq_api_key,
    urls,
):

    topics = [
        extract_topic_from_url(u)
        for u in urls
    ]

    topics_str = ", ".join(topics)

    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name="llama-3.1-8b-instant",
    )

    structured_router = llm.with_structured_output(
        RouteQuery
    )

    sys_prompt = f"""
    You are an expert router.

    The vectorstore contains:
    {topics_str}

    Use vectorstore for these topics.

    Otherwise use wiki_search.
    """

    route_prompt = ChatPromptTemplate.from_messages([
        ("system", sys_prompt),
        ("human", "{question}")
    ])

    return route_prompt | structured_router