import cassio

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
)

from langchain_community.document_loaders import (
    WebBaseLoader,
)

from langchain_huggingface import (
    HuggingFaceEmbeddings,
)

from langchain_community.vectorstores.cassandra import (
    Cassandra,
)


def create_vectorstore(
    urls,
    astra_token,
    astra_db_id,
):

    # AstraDB Init
    cassio.init(
        token=astra_token,
        database_id=astra_db_id,
    )

    # Load Documents
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

    doc_splits = splitter.split_documents(
        docs_list
    )

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    vectorstore = Cassandra(
        embedding=embeddings,
        table_name="qa_mini_demo",
        session=None,
        keyspace=None,
    )

    vectorstore.delete_collection()

    vectorstore.add_documents(doc_splits)

    return vectorstore