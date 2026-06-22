from lore_master.rag_chat.fetch_wiki import fetch_wiki
from lore_master.rag_chat.ingest import fetch_documents, create_chunks, create_embeddings

if __name__ == "__main__":
    fetch_wiki()
    documents = fetch_documents()
    chunks = create_chunks(documents)
    create_embeddings(chunks)
    print("Ingestion complete")

