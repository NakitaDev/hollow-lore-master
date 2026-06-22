import os
import glob
from tqdm import tqdm
from pathlib import Path
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from lore_master.core.components import build_embeddings
from lore_master.core.config import get_settings


s = get_settings()
DB_NAME = str(Path(__file__).resolve().parents[3] / s.persist_dir)
KNOWLEDE_BASE = str(Path(__file__).resolve().parents[3] / s.knowlede_dir)

def fetch_documents():
    folders = glob.glob(str(Path(KNOWLEDE_BASE) / "*"))
    documents =[]
    for folder in folders:
        doc_type = os.path.basename(folder)
        loader = DirectoryLoader(folder, glob="**/*.md", loader_cls=TextLoader, 
        loader_kwargs={"encoding": "utf-8"})
        folder_docs = loader.load()
        for doc in tqdm(folder_docs, desc="fetching documents" ):
            doc.metadata["doc_type"] = doc_type
            documents.append(doc)
    print("finished fetch doc")
    return documents

def create_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    print("finished chunks doc")
    return chunks

def create_embeddings(chunks):
    if os.path.exists(DB_NAME):
        Chroma(persist_directory=DB_NAME, embedding_function=build_embeddings()).delete_collection()

    vectorstore = Chroma.from_documents(
        documents=chunks, 
        embedding=build_embeddings(), 
        persist_directory=DB_NAME,
        collection_name=s.collection_name
    )

    collection = vectorstore._collection
    count = collection.count()

    sample_embedding = collection.get(limit=1, include=["embeddings"])["embeddings"][0]
    dimensions = len(sample_embedding)
    print(f"There are {count:,} vectors with {dimensions:,} dimensions in the vector store")
    return vectorstore
