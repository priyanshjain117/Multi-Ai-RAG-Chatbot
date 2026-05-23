import os
import shutil

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
)

from langchain_community.document_loaders import (
    WebBaseLoader,
)

from langchain_huggingface import (
    HuggingFaceEmbeddings,
)

from langchain_community.vectorstores import Chroma

# Local directory to persist the ChromaDB index
CHROMA_DIR = os.path.join(os.path.dirname(__file__), ".chroma_db")


def create_vectorstore(urls):

    # Clear any existing index so we start fresh each init
    if os.path.exists(CHROMA_DIR):
        shutil.rmtree(CHROMA_DIR)

    # Load Documents from URLs
    docs = [
        WebBaseLoader(url).load()
        for url in urls
    ]

    docs_list = [
        item
        for sublist in docs
        for item in sublist
    ]

    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=500,
        chunk_overlap=50,
    )

    doc_splits = splitter.split_documents(docs_list)

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    vectorstore = Chroma.from_documents(
        documents=doc_splits,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
    )

    return vectorstore