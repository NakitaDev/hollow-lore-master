---
title: Hollow Lore Master
emoji: 🦋
colorFrom: purple
colorTo: blue
sdk: gradio
sdk_version: "6.19.0"
python_version: "3.12"
app_file: app.py
pinned: false
---

# Lore Master (LangChain Edition)

A **Hollow Knight lore Q&A bot** built with **LangChain**. It scrapes lore from
the [Hollow Knight Fandom wiki](https://hollowknight.fandom.com), embeds it into a
local Chroma vector store, and answers questions in English through a Gradio chat
UI — grounded in the retrieved lore and citing its sources.

The retrieval chain is **history-aware**: follow-up questions like *"where can I
find him?"* are rewritten into standalone queries before retrieval, so the bot
keeps context across a conversation.

## Architecture

```
fetch_wiki  →  data/knowledge-base/*.md  →  ingest  →  data/vector_db (Chroma)
                                                            │
 user ──► Gradio ──► rag_chain ──► (rewrite question) ──► retriever ──► prompt ──► LLM ──► answer
```

| Concern        | Component                                                                 |
|----------------|---------------------------------------------------------------------------|
| Chat model     | `ChatOpenRouter` — `anthropic/claude-haiku-4-5` ([`core/components.py`](source/lore_master/core/components.py)) |
| Embeddings     | `HuggingFaceEmbeddings` — `all-MiniLM-L6-v2`, local & free                 |
| Fetch lore     | [`rag_chat/fetch_wiki.py`](source/lore_master/rag_chat/fetch_wiki.py) — MediaWiki API → clean `.md` |
| Ingestion      | [`rag_chat/ingest.py`](source/lore_master/rag_chat/ingest.py) — load → split → embed → Chroma |
| Retrieval      | `build_retriever()` — `vectorstore.as_retriever(k=4)`                      |
| RAG chain      | [`rag_chat/rag_chain.py`](source/lore_master/rag_chat/rag_chain.py) — history-aware LCEL chain |
| UI             | [`scripts/app_rag.py`](scripts/app_rag.py) — `gr.ChatInterface`            |


## Setup

Requires Python 3.12+.

```bash
uv venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
uv pip install -r requirements.txt
uv pip install -e .
```

Copy `.env.example` to `.env` and set your OpenRouter key:

```
OPENROUTER_API_KEY=your-key-here
```

> The first run downloads the embedding model (`all-MiniLM-L6-v2`, ~90 MB).
> After that, embeddings run locally and offline.

## Build the knowledge base

Fetch lore from the wiki and ingest it into the vector store in one step:

```bash
python scripts/run_fetch_ingest.py
```

This:
1. Crawls the wiki recursively starting from `fetch_wiki.ROOT_CATEGORY` (every
   sub-category found along the way is followed too) and saves clean `.md`
   files into `data/knowledge-base/`, mirroring the category hierarchy as
   nested folders. Pages already saved from a previous run are skipped.
2. Splits them into chunks, embeds them, and (re)builds the Chroma store at
   `data/vector_db/`.

> Re-running rebuilds the collection from scratch (the old one is deleted first).

## Run the chatbot

```bash
python scripts/app_rag.py
```

Launches a Gradio web UI that answers Hollow Knight questions grounded in the
ingested lore, cites source files, and declines to guess when the answer isn't in
context.

## Configuration

All settings live in [`source/lore_master/core/config.py`](source/lore_master/core/config.py):
model, `temperature`, number of chunks retrieved (`retrival_k`), wiki categories,
and the knowledge-base / vector-store paths.

## Deploy to Hugging Face Spaces

The YAML header at the top of this file configures a **Gradio Space** whose entry
point is [`app.py`](app.py). To deploy:

1. Create a new **Gradio** Space and push this repo to it.
2. In the Space **Settings → Secrets**, add `OPENROUTER_API_KEY`.
3. Commit the prebuilt `data/vector_db/` (and `data/knowledge-base/`) so the Space
   has a knowledge base without running the fetch step — or add a build step that
   runs `scripts/run_fetch_ingest.py`.

> `app.py` adds `source/` to `sys.path` itself, so the Space works with a plain
> `pip install -r requirements.txt` — no editable install (`pip install -e .`)
> needed.

## What LangChain abstracts

Compared to a from-scratch build, LangChain collapses a lot of hand-written
plumbing into framework calls: the retry/backoff loop becomes
`max_retries=3` on the chat model; manual chunking becomes
`RecursiveCharacterTextSplitter`; opening Chroma and building ids/metadata becomes
`Chroma.from_documents(...)`; and the whole retrieve → rewrite → prompt → answer
flow becomes a single LCEL pipe with `RunnablePassthrough.assign`. Less code to
maintain, but more behavior hidden behind abstractions.
