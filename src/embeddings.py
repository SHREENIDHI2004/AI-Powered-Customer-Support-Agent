import numpy as np
from typing import List
from langchain_core.embeddings import Embeddings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

DEFAULT_DENSE_MODEL = "BAAI/bge-small-en-v1.5"
FALLBACK_DENSE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

class DenseHuggingFaceEmbeddings(Embeddings):
    """LangChain Embeddings wrapper using sentence-transformers (BGE / MiniLM)."""
    def __init__(self, model_name: str = DEFAULT_DENSE_MODEL):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        try:
            self._model = SentenceTransformer(model_name, device="cpu")
        except Exception as e:
            print(f"Warning: Failed to load {model_name} ({e}). Falling back to {FALLBACK_DENSE_MODEL}.")
            self.model_name = FALLBACK_DENSE_MODEL
            self._model = SentenceTransformer(FALLBACK_DENSE_MODEL, device="cpu")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        vectors = self._model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False
        )
        return np.asarray(vectors).tolist()

    def embed_query(self, text: str) -> List[float]:
        vector = self._model.encode(
            [text], normalize_embeddings=True, show_progress_bar=False
        )[0]
        return np.asarray(vector).tolist()


class LocalTfidfLsaEmbeddings(Embeddings):
    """Zero-download TF-IDF + LSA Embeddings implementation for offline baseline evaluation."""
    def __init__(self, n_components: int = 64):
        self.n_components = n_components
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        self.is_fitted = False

    def fit(self, texts: List[str]):
        tfidf_mat = self.vectorizer.fit_transform(texts)
        actual_components = min(self.n_components, tfidf_mat.shape[1] - 1, tfidf_mat.shape[0] - 1)
        if actual_components < 2:
            actual_components = max(1, min(self.n_components, tfidf_mat.shape[1]))
        self.svd = TruncatedSVD(n_components=actual_components, random_state=42)
        self.svd.fit(tfidf_mat)
        self.is_fitted = True

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not self.is_fitted:
            self.fit(texts)
        tfidf_mat = self.vectorizer.transform(texts)
        dense_vecs = self.svd.transform(tfidf_mat)
        # Normalize
        norms = np.linalg.norm(dense_vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normalized = dense_vecs / norms
        return normalized.tolist()

    def embed_query(self, text: str) -> List[float]:
        if not self.is_fitted:
            # Fallback if not fitted yet
            return [0.0] * self.n_components
        tfidf_mat = self.vectorizer.transform([text])
        dense_vec = self.svd.transform(tfidf_mat)[0]
        norm = np.linalg.norm(dense_vec)
        if norm > 0:
            dense_vec = dense_vec / norm
        return dense_vec.tolist()


def get_embeddings_model(model_type: str = "dense", model_name: str = DEFAULT_DENSE_MODEL) -> Embeddings:
    """Factory function for embedding models."""
    if model_type == "tfidf":
        return LocalTfidfLsaEmbeddings()
    return DenseHuggingFaceEmbeddings(model_name=model_name)

if __name__ == "__main__":
    emb = get_embeddings_model("dense")
    vec = emb.embed_query("What is the return policy?")
    print(f"Dense vector length: {len(vec)}")
