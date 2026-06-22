from pathlib import Path
from langchain_openrouter import ChatOpenRouter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from .config import get_settings


def build_chat_model() -> ChatOpenRouter:
    s = get_settings()
    # ChatOpenRouter reads Openrouter_API_KEY from the environment.
    return ChatOpenRouter(
        model=s.model,
        max_tokens=s.max_tokens,
        temperature=s.temperature,
        max_retries=3,
    )


def build_embeddings() -> HuggingFaceEmbeddings:
    s = get_settings()
    return HuggingFaceEmbeddings(model_name=s.embedding_model)

def build_retriever():
    s = get_settings()
    DB_NAME = str(Path(__file__).resolve().parents[3] / s.persist_dir)
    vectorstore = Chroma(
        persist_directory=DB_NAME,
        collection_name=s.collection_name,
        embedding_function=build_embeddings(),
    )
    return vectorstore.as_retriever(search_kwargs={"k": s.retrival_k})