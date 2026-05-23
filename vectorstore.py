"""AstraDB vectorstore lifecycle helpers."""

from __future__ import annotations

import os
from typing import Iterable
from urllib.parse import urlparse

os.environ.setdefault("USER_AGENT", "LangGraph-RAG-Agent/1.0")

import cassio
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores.cassandra import Cassandra
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


TABLE_NAME = "qa_mini_demo"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class VectorstoreInitError(RuntimeError):
    """User-safe vectorstore initialization error."""


def _validate_url(url: str) -> str:
    url = url.strip()
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise VectorstoreInitError(f"Invalid knowledge-base URL: {url}")
    return url


def _is_missing_collection_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(
        needle in message
        for needle in (
            "does not exist",
            "not found",
            "unknown table",
            "unconfigured table",
        )
    )


def _load_documents(urls: Iterable[str]):
    docs = []
    for url in urls:
        url = _validate_url(url)
        try:
            docs.extend(WebBaseLoader(url).load())
        except Exception as exc:
            raise VectorstoreInitError(
                f"Could not load knowledge-base URL: {url}"
            ) from exc
    return docs


def _split_documents(docs):
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=500,
        chunk_overlap=50,
    )
    return splitter.split_documents(docs)


def _connect_astra(token: str, database_id: str) -> None:
    try:
        cassio.init(
            token=token,
            database_id=database_id,
            keyspace=os.getenv("ASTRA_DB_KEYSPACE") or None,
        )
    except Exception as exc:
        raise VectorstoreInitError(
            "Could not connect to AstraDB. Check your token and database ID."
        ) from exc


def _new_vectorstore(embeddings):
    try:
        return Cassandra(
            embedding=embeddings,
            table_name=TABLE_NAME,
            session=None,
            keyspace=None,
        )
    except Exception as exc:
        raise VectorstoreInitError(
            "Could not initialize the AstraDB vector collection."
        ) from exc


def _replace_collection_documents(vectorstore, doc_splits) -> None:
    """Clear the AstraDB collection, then add only the current document chunks."""
    try:
        vectorstore.delete_collection()
    except Exception as exc:
        if not _is_missing_collection_error(exc):
            raise VectorstoreInitError(
                "Could not clear the previous AstraDB vector data."
            ) from exc

    try:
        vectorstore.add_documents(doc_splits)
    except Exception as exc:
        raise VectorstoreInitError(
            "Could not add documents to the AstraDB vector collection."
        ) from exc


def create_vectorstore(urls: list[str], astra_token: str, astra_db_id: str):
    """Create a fresh AstraDB vectorstore for the current indexing session.

    This intentionally clears the shared table before indexing. It is a
    temporary global-reset strategy until authentication and user isolation are
    added.
    """

    if not astra_token or not astra_db_id:
        raise VectorstoreInitError("AstraDB token and database ID are required.")

    docs = _load_documents(urls)
    if not docs:
        raise VectorstoreInitError("No documents were loaded from the provided URLs.")

    doc_splits = _split_documents(docs)
    if not doc_splits:
        raise VectorstoreInitError("No text chunks were created from the provided URLs.")

    _connect_astra(astra_token, astra_db_id)

    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"local_files_only": True},
        )
    except Exception as exc:
        raise VectorstoreInitError(
            "Could not load the cached embedding model. Connect to the internet once to download it, then try again."
        ) from exc

    vectorstore = _new_vectorstore(embeddings)
    _replace_collection_documents(vectorstore, doc_splits)

    try:
        vectorstore.as_retriever().invoke("health check")
    except Exception as exc:
        if _is_missing_collection_error(exc):
            vectorstore = _new_vectorstore(embeddings)
        else:
            raise VectorstoreInitError(
                "AstraDB initialized, but retrieval is not available yet."
            ) from exc

    return vectorstore
