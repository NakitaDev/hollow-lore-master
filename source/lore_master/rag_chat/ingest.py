import hashlib
from tqdm import tqdm
from pathlib import Path
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from lore_master.core.components import build_embeddings, ensure_pinecone_index
from lore_master.core.config import get_settings


s = get_settings()
KNOWLEDGE_BASE = str(Path(__file__).resolve().parents[3] / s.knowledge_dir)

def fetch_documents():
    base = Path(KNOWLEDGE_BASE)
    documents = []
    # rglob walks every nested sub-category folder the crawler creates,
    # not just the immediate children of the knowledge-base directory.
    for md_path in tqdm(sorted(base.rglob("*.md")), desc="fetching documents"):
        doc_type = md_path.relative_to(base).parts[0]
        loader = TextLoader(str(md_path), encoding="utf-8")
        for doc in loader.load():
            doc.metadata["doc_type"] = doc_type
            documents.append(doc)
    print("finished fetch doc")
    return documents

def create_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=s.chunk_size, chunk_overlap=s.chunk_overlap
    )
    chunks = text_splitter.split_documents(documents)
    print("finished chunks doc")
    return chunks

def _stable_id(chunk) -> str:
    """Deterministic chunk ID from its source file + exact text.

    Chroma assigns a random UUID to every chunk when no ``ids`` are given,
    so re-running ingest on unchanged content still produces a brand-new
    set of record IDs each time. Hashing (source, text) instead means the
    same chunk gets the same ID across runs, while content that actually
    changed naturally gets a new ID.
    """
    source = chunk.metadata.get("source", "")
    digest = hashlib.sha256(f"{source}::{chunk.page_content}".encode("utf-8"))
    return digest.hexdigest()

def create_embeddings(chunks):
    ids = [_stable_id(chunk) for chunk in chunks]

    pc = ensure_pinecone_index()
    index = pc.Index(s.pinecone_index_name)

    # Clear out whatever's already in the index before re-ingesting, mirroring
    # the old "delete the collection, then rebuild" behavior.
    stats = index.describe_index_stats()
    if stats["total_vector_count"] > 0:
        index.delete(delete_all=True)

    vectorstore = PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=build_embeddings(),
        index_name=s.pinecone_index_name,
        ids=ids,
    )

    stats = index.describe_index_stats()
    print(
        f"There are {stats['total_vector_count']:,} vectors with "
        f"{stats['dimension']:,} dimensions in the vector store"
    )
    return vectorstore
