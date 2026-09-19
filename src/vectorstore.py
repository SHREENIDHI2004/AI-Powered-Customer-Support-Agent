import json
import os
from typing import List, Optional, Tuple

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from src.embeddings import get_embeddings_model
from src.ingest import get_chunked_documents

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
INDEX_DIR = os.path.join(ROOT_DIR, "data", "faiss_index")
INDEX_META_PATH = os.path.join(INDEX_DIR, "index_meta.json")
DEFAULT_K = 4


def build_faiss_index(
    index_dir: str = INDEX_DIR,
    model_type: str = "dense",
    model_name: Optional[str] = None,
) -> FAISS:
    chunks = get_chunked_documents()
    if not chunks:
        raise RuntimeError("No knowledge-base documents found under data/kb/.")

    kwargs = {"model_type": model_type}
    if model_name:
        kwargs["model_name"] = model_name
    embeddings = get_embeddings_model(**kwargs)

    if model_type == "tfidf":
        embeddings.embed_documents([c.page_content for c in chunks])

    store = FAISS.from_documents(chunks, embeddings)
    os.makedirs(index_dir, exist_ok=True)
    store.save_local(index_dir)
    with open(INDEX_META_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_type": model_type,
                "model_name": getattr(embeddings, "model_name", model_type),
                "num_chunks": len(chunks),
            },
            f,
            indent=2,
        )
    return store


def build_in_memory_index(model_type: str = "tfidf", model_name: Optional[str] = None) -> FAISS:
    """Build a non-persistent FAISS index (used for TF-IDF baseline comparison)."""
    chunks = get_chunked_documents()
    kwargs = {"model_type": model_type}
    if model_name:
        kwargs["model_name"] = model_name
    embeddings = get_embeddings_model(**kwargs)
    if model_type == "tfidf":
        embeddings.embed_documents([c.page_content for c in chunks])
    return FAISS.from_documents(chunks, embeddings)


def load_faiss_index(
    index_dir: str = INDEX_DIR,
    model_type: str = "dense",
    model_name: Optional[str] = None,
) -> FAISS:
    if not os.path.exists(os.path.join(index_dir, "index.faiss")):
        return build_faiss_index(index_dir=index_dir, model_type=model_type, model_name=model_name)

    kwargs = {"model_type": model_type}
    if model_name:
        kwargs["model_name"] = model_name
    embeddings = get_embeddings_model(**kwargs)
    return FAISS.load_local(
        index_dir,
        embeddings,
        allow_dangerous_deserialization=True,
    )


def similarity_search_with_scores(
    store: FAISS,
    query: str,
    k: int = DEFAULT_K,
) -> List[Tuple[Document, float]]:
    """Return (document, L2 distance) pairs. Lower distance is more similar."""
    return store.similarity_search_with_score(query, k=k)


if __name__ == "__main__":
    store = build_faiss_index()
    sample = similarity_search_with_scores(store, "What is the return policy for headphones?", k=3)
    print(f"Indexed FAISS store at {INDEX_DIR}")
    print(f"Sample retrieval for return-policy query:")
    for doc, score in sample:
        snippet = doc.page_content.replace("\n", " ")[:120]
        print(f"  distance={score:.4f} source={doc.metadata.get('source')} :: {snippet}...")
