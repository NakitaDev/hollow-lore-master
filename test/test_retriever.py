from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION = "test_collection"

SAMPLE = """# Quirrel

Quirrel is a traveling pilgrim and one of the few friendly faces the
Knight meets across Hallownest. He carries a nail and is often found
resting at points of interest, reflecting on the kingdom's history.
"""


def test_retriever_returns_relevant_documents(tmp_path):
    # Build a tiny corpus on disk.
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "quirrel.md").write_text(SAMPLE, encoding="utf-8")

    docs = DirectoryLoader(
        str(raw_dir), glob="**/*.md", loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    ).load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    persist_dir = str(tmp_path / "vector_store")

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir,
        collection_name=COLLECTION,
    )

    vectorstore = Chroma(
        persist_directory=persist_dir,
        collection_name=COLLECTION,
        embedding_function=embeddings,
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    results = retriever.invoke("Who is the traveling pilgrim Quirrel?")

    assert len(results) > 0
    combined = " ".join(d.page_content for d in results).lower()
    assert "quirrel" in combined
