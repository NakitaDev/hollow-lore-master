"""Hugging Face Spaces entry point.

HF Spaces only runs ``pip install -r requirements.txt`` — it does NOT run
``pip install -e .``. So the ``lore_master`` package under ``source/`` isn't
importable by default. This file fixes that by adding ``source/`` to ``sys.path``
before importing, then launches the same Gradio UI as ``scripts/app_rag.py``.
"""

import sys
import os
from pathlib import Path

# ให้ Python หา package lore_master ใน source/ เจอ โดยไม่ต้องติดตั้งแบบ editable
SOURCE_DIR = Path(__file__).resolve().parent / "source"
sys.path.insert(0, str(SOURCE_DIR))

import gradio as gr

from lore_master.rag_chat.rag_chain import build_rag_chain, ingest_documents

chain = build_rag_chain()
# เช็คว่ามี vector DB ไหม ถ้าไม่มีให้ ingest ใหม่
if not os.path.exists("data/vector_db"):
    print("Building vector DB...")
    ingest_documents()  # สร้าง DB ใหม่จากไฟล์ lore

def chat(message: str, history: list[dict]) -> str:
    return chain.invoke({"question": message, "history": history})


demo = gr.ChatInterface(
    fn=chat, title="Hollow Knight Lore Bot (RAG, LangChain)"
)

if __name__ == "__main__":
    demo.launch()
