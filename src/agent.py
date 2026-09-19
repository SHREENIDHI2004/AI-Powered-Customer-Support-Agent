from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from langchain_core.documents import Document

from src.guardrails import apply_guardrails, detect_ood_query, make_refusal
from src.llm_backends import LLMBackend, get_backend
from src.schemas import SupportResponse
from src.vectorstore import DEFAULT_K, load_faiss_index, similarity_search_with_scores


@dataclass
class AgentResult:
    response: SupportResponse
    retrieved: List[Document] = field(default_factory=list)
    distances: List[float] = field(default_factory=list)
    grounding_ratio: float = 0.0
    guardrail_status: str = "pass"
    repair_attempts: int = 0
    retrieved_sources: List[str] = field(default_factory=list)


class SupportAgent:
    def __init__(
        self,
        backend: Optional[LLMBackend] = None,
        model_type: str = "dense",
        k: int = DEFAULT_K,
        store=None,
    ):
        self.backend = backend or get_backend("ollama")
        self.k = k
        self.store = store if store is not None else load_faiss_index(model_type=model_type)

    def retrieve(self, query: str) -> Tuple[List[Document], List[float]]:
        hits = similarity_search_with_scores(self.store, query, k=self.k)
        docs = [doc for doc, _ in hits]
        distances = [float(score) for _, score in hits]
        return docs, distances

    def ask(self, query: str) -> AgentResult:
        ood = detect_ood_query(query)
        if ood:
            refused = make_refusal(ood)
            return AgentResult(
                response=refused,
                grounding_ratio=0.0,
                guardrail_status="refuse",
                repair_attempts=0,
            )

        docs, distances = self.retrieve(query)
        sources = [d.metadata.get("source", "unknown") for d in docs]
        context = "\n\n".join(d.page_content for d in docs)

        raw_response, repair_attempts = self.backend.generate(query, docs)
        guarded = apply_guardrails(raw_response, context, query, distances)
        final = guarded.response or raw_response
        status = guarded.status
        if repair_attempts and status == "pass":
            status = "repair"

        return AgentResult(
            response=final,
            retrieved=docs,
            distances=distances,
            grounding_ratio=guarded.grounding_ratio,
            guardrail_status=status,
            repair_attempts=repair_attempts,
            retrieved_sources=sources,
        )


def get_agent(backend_name: str = "ollama", model_type: str = "dense") -> SupportAgent:
    return SupportAgent(backend=get_backend(backend_name), model_type=model_type)
