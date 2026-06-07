from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from config import EMBED_MODEL, RAG_CHUNK_SIZE, RAG_CHUNK_OVERLAP, RAG_TOP_K
from router import normalize_path

# Load embedding model once
_embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)


def load_and_chunk(filepath: str) -> list:
    path = normalize_path(filepath)
    ext  = path.suffix.lower()

    if ext == ".pdf":
        docs = PyPDFLoader(str(path)).load()
    elif ext in {".docx", ".doc"}:
        docs = Docx2txtLoader(str(path)).load()
    else:
        docs = TextLoader(str(path), encoding="utf-8").load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=RAG_CHUNK_SIZE,
        chunk_overlap=RAG_CHUNK_OVERLAP,
    )
    return splitter.split_documents(docs)


def build_store(filepath: str) -> FAISS:
    # Create fresh in-memory FAISS store from a document
    chunks = load_and_chunk(filepath)
    return FAISS.from_documents(chunks, _embeddings)


def add_to_store(store: FAISS, filepath: str) -> None:
    # Merge new document into existing store
    chunks = load_and_chunk(filepath)
    store.add_documents(chunks)


def get_retriever(store: FAISS):
    return store.as_retriever(search_kwargs={"k": RAG_TOP_K})