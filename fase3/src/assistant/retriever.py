"""
RAG (Retrieval-Augmented Generation) com ChromaDB.
Indexa protocolos e FAQ do hospital para recuperação semântica.
"""

import json
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

COLLECTION_NAME = "hospital_knowledge_base"
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def _get_client(chroma_path: str = "chroma_db") -> chromadb.PersistentClient:
    return chromadb.PersistentClient(path=chroma_path)


def _get_collection(chroma_path: str = "chroma_db") -> chromadb.Collection:
    client = _get_client(chroma_path)
    ef = SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
    return client.get_or_create_collection(name=COLLECTION_NAME, embedding_function=ef)


def build_index(
    protocols_path: str = "data/synthetic/protocols.json",
    faq_path: str = "data/synthetic/faq.json",
    chroma_path: str = "chroma_db",
) -> None:
    """Cria ou recria o índice vetorial com protocolos e FAQ."""
    collection = _get_collection(chroma_path)

    documents, metadatas, ids = [], [], []

    with open(protocols_path, "r", encoding="utf-8") as f:
        protocols = json.load(f)
    for i, proto in enumerate(protocols):
        documents.append(f"{proto['title']}\n\n{proto['content']}")
        metadatas.append({"type": "protocol", "title": proto["title"]})
        ids.append(f"protocol_{i}")

    with open(faq_path, "r", encoding="utf-8") as f:
        faq = json.load(f)
    for i, qa in enumerate(faq):
        documents.append(f"P: {qa['question']}\nR: {qa['answer']}")
        metadatas.append({"type": "faq", "question": qa["question"]})
        ids.append(f"faq_{i}")

    collection.upsert(documents=documents, metadatas=metadatas, ids=ids)
    print(f"Índice criado com {len(documents)} documentos em '{chroma_path}/'")


def retrieve(
    query: str,
    n_results: int = 3,
    chroma_path: str = "chroma_db",
) -> list[dict]:
    """Recupera os documentos mais relevantes para a query."""
    collection = _get_collection(chroma_path)
    results = collection.query(query_texts=[query], n_results=n_results)

    docs = []
    for i, doc in enumerate(results["documents"][0]):
        docs.append({
            "content": doc,
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i],
        })
    return docs


def format_context(docs: list[dict]) -> tuple[str, list[str]]:
    """Formata os documentos recuperados em contexto para o LLM e retorna as fontes."""
    context_parts = []
    sources = []
    for doc in docs:
        meta = doc["metadata"]
        source = meta.get("title") or meta.get("question", "Documento interno")
        context_parts.append(f"[{meta['type'].upper()}] {source}:\n{doc['content']}")
        sources.append(source)
    return "\n\n---\n\n".join(context_parts), sources
