import re

from typing import Literal
from urllib.parse import urlparse, unquote

from pydantic import BaseModel, Field

from langchain_core.prompts import ChatPromptTemplate

from langchain_groq import ChatGroq


class RouteQuery(BaseModel):
    """Classify a user message to exactly one routing target."""

    datasource: Literal[
        "vectorstore",
        "wiki_search",
        "general_chat",
    ] = Field(
        ...,
        description=(
            "'vectorstore' — question is about a topic stored in the knowledge base; "
            "'wiki_search' — factual/encyclopedic question not covered by vectorstore; "
            "'general_chat' — greeting, pleasantry, small talk, or casual expression."
        ),
    )


def extract_topic_from_url(url: str) -> str:
    parsed = urlparse(url)
    slug = unquote(parsed.path.rstrip("/").split("/")[-1])
    if not slug:
        return parsed.netloc or url
    slug = re.sub(r"^\d{4}-\d{2}-\d{2}-", "", slug)
    return slug.replace("-", " ")


def create_router(groq_api_key: str, urls: list):

    topics = [extract_topic_from_url(u) for u in urls]
    topics_str = ", ".join(topics)

    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name="llama-3.1-8b-instant",
    )

    structured_router = llm.with_structured_output(RouteQuery)

    system = f"""You are an expert query classifier. \
Given a user message, output EXACTLY ONE label:

"vectorstore"  → The question is specifically about one of these knowledge-base topics:
                 {topics_str}
                 Use this for direct questions about prompt engineering, AI agents,
                 agent memory, planning, tool use, or adversarial attacks on LLMs.
                 Example: "What is prompt engineering?", "Explain agent memory"

"wiki_search"  → Factual or encyclopedic question NOT covered by the topics above.
                 Example: "Who is Albert Einstein?", "What is quantum computing?"

"general_chat" → Greeting, pleasantry, small talk, or casual expression that needs
                 NO factual lookup.
                 Example: "Hello", "Hi!", "How are you?", "Thanks!", "Bye", "Good morning"

CRITICAL RULE: ANY greeting or social pleasantry → ALWAYS "general_chat".
Never route greetings or small talk to vectorstore or wiki_search.
Do not route factual world-knowledge questions to general_chat."""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", "{question}"),
    ])

    return prompt | structured_router
