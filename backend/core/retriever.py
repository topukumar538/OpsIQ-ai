# Location: backend/core/retriever.py
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from config import EMBED_MODEL

# Single shared embedding model instance
_embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)


def get_embeddings() -> HuggingFaceEmbeddings:
    return _embeddings


def retrieve(store: FAISS, query: str, k: int = 4) -> str:
    # Retrieve top-k chunks and return as formatted string
    docs = store.as_retriever(search_kwargs={"k": k}).invoke(query)
    return "\n\n".join([doc.page_content for doc in docs])