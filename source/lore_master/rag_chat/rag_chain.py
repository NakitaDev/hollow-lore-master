from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough

from lore_master.core.components import build_chat_model, build_retriever

RAG_SYSTEM_PROMPT = (
    """You answer questions about Hollow Knight lore using ONLY the provided
    context. Answer in English. If the answer is not in the context, say you
    don't know rather than guessing. Cite the sources you used by their filename."""
)

# ใช้ history + คำถามล่าสุด -> เขียนเป็นคำค้นเดี่ยวๆ (ไม่ตอบ แค่ rewrite)
# ถ้าคำถาม standalone อยู่แล้ว ให้คืนเดิม -> ไม่ต้องมี branch/heuristic
CONTEXTUALIZE_SYSTEM = (
    "Given the chat history and the latest user question, rewrite the question "
    "as a standalone search query in English that makes sense without the history. "
    "If it is already standalone, return it unchanged. Do NOT answer it."
)


def format_docs(docs) -> str:
    return "\n\n".join(
        f"[source: {d.metadata.get('source', 'unknown')}]\n{d.page_content}"
        for d in docs
    )


def build_rag_chain():
    retriever = build_retriever()
    model = build_chat_model()
    parser = StrOutputParser()

    # ---- ขั้นที่ 1: ทำคำถามให้เป็น standalone เสมอ ----
    contextualize_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", CONTEXTUALIZE_SYSTEM),
            MessagesPlaceholder("history"),
            ("human", "{question}"),
        ]
    )
    standalone_question = contextualize_prompt | model | parser

    # ---- ขั้นที่ 2: retrieve ด้วยคำถาม standalone แล้วตอบ ----
    answer_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", RAG_SYSTEM_PROMPT),
            MessagesPlaceholder("history"),
            ("human", "Context:\n{context}\n\nQuestion: {question}"),
        ]
    )

    # assign เพิ่ม key "context" โดยยังเก็บ question/history ไว้ให้ answer_prompt
    return (
        RunnablePassthrough.assign(
            context=standalone_question | retriever | format_docs
        )
        | answer_prompt
        | model
        | parser
    )
