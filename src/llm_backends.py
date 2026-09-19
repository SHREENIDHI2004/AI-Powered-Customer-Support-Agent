import json
import re
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

from langchain_core.documents import Document
from pydantic import ValidationError

from src.schemas import SupportResponse

JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)

SUPPORT_JSON_INSTRUCTIONS = """You are a customer support agent for an e-commerce store.
Use ONLY the provided CONTEXT. Do not invent policies, prices, or products.

Return a single JSON object with exactly these keys:
- answer (string): the customer-facing reply
- can_answer (boolean): true whenever CONTEXT contains the facts needed to reply
- sources (array of strings): knowledge-base filenames you used
- confidence (number 0 to 1)
- refusal_reason (string or null): set only when can_answer is false

can_answer means "the knowledge base covers this question", not "the customer's request is approved".
A policy "no" is still an answer: set can_answer=true (for example, cannot change an address after fulfillment).
If CONTEXT lists a supported payment method (cards, wallets, Klarna, etc.) and its limits, answer from those facts
with can_answer=true. Cryptocurrency and other methods listed as not accepted should be answered from CONTEXT
when they appear there, also with can_answer=true.
If CONTEXT includes product specifications (switches, battery, IP rating, price, …), answer from them with can_answer=true.

Set can_answer=false only when CONTEXT does not cover the question at all. Then leave answer brief and set refusal_reason.
A true coverage refusal is a valid, complete response. Do not refuse a covered question.
"""


def _strip_json_payload(raw: str) -> str:
    text = (raw or "").strip()
    fenced = JSON_FENCE_RE.search(text)
    if fenced:
        text = fenced.group(1).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start : end + 1]
    return text


def parse_support_response(raw: str) -> SupportResponse:
    payload = json.loads(_strip_json_payload(raw))
    if not isinstance(payload, dict):
        raise ValueError("JSON root must be an object")
    return SupportResponse.model_validate(payload)


class LLMBackend(ABC):
    name: str = "base"

    @abstractmethod
    def generate(
        self,
        query: str,
        documents: List[Document],
        *,
        max_repairs: int = 2,
    ) -> Tuple[SupportResponse, int]:
        """Return (response, repair_attempts). Repair attempts increment only on JSON/schema failure."""


class OllamaBackend(LLMBackend):
    """Local Ollama JSON backend. Repair loop fires only on malformed JSON/schema — never on refusals."""

    name = "ollama"

    def __init__(self, model: str = "llama3.2:latest", temperature: float = 0.0):
        self.model = model
        self.temperature = temperature

    def _chat(self, messages: list) -> str:
        try:
            import ollama
        except ImportError as exc:
            raise RuntimeError("The ollama package is required for OllamaBackend.") from exc

        result = ollama.chat(
            model=self.model,
            messages=messages,
            format="json",
            options={"temperature": self.temperature},
        )
        message = result.get("message") or {}
        return message.get("content") or ""

    @staticmethod
    def _format_context(documents: List[Document]) -> str:
        blocks = []
        for doc in documents:
            source = doc.metadata.get("source", "unknown")
            blocks.append(f"[SOURCE: {source}]\n{doc.page_content}")
        return "\n\n".join(blocks) if blocks else "(no retrieved context)"

    def generate(
        self,
        query: str,
        documents: List[Document],
        *,
        max_repairs: int = 2,
    ) -> Tuple[SupportResponse, int]:
        context = self._format_context(documents)
        user_prompt = (
            f"{SUPPORT_JSON_INSTRUCTIONS}\n\nQUESTION:\n{query}\n\nCONTEXT:\n{context}"
        )
        messages = [{"role": "user", "content": user_prompt}]
        last_error: Optional[str] = None
        last_raw = ""
        repair_attempts = 0

        for attempt in range(max_repairs + 1):
            if attempt > 0:
                repair_attempts += 1
                messages.append({"role": "assistant", "content": last_raw or ""})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Your previous message was not valid JSON matching the SupportResponse schema. "
                            f"Parser/schema error: {last_error}\n\n"
                            "Output ONLY a JSON object with keys answer, can_answer, sources, "
                            "confidence, and refusal_reason.\n"
                            "If you already determined the question is out of scope, keep "
                            "can_answer=false. Do not change a valid refusal into an answer. "
                            "Repair JSON/schema only; do not try harder to answer."
                        ),
                    }
                )
            last_raw = self._chat(messages)
            try:
                parsed = parse_support_response(last_raw)
                return parsed, repair_attempts
            except (json.JSONDecodeError, ValidationError, ValueError) as exc:
                last_error = str(exc)
                continue

        raise ValueError(
            f"Ollama output did not match SupportResponse after {max_repairs} repair(s): {last_error}"
        )


class ExtractiveLocalBackend(LLMBackend):
    """Baseline backend: extract overlapping sentences from retrieved chunks (no generative LLM)."""

    name = "extractive"

    def generate(
        self,
        query: str,
        documents: List[Document],
        *,
        max_repairs: int = 2,
    ) -> Tuple[SupportResponse, int]:
        del max_repairs  # extractive path never repairs JSON
        if not documents:
            return (
                SupportResponse(
                    answer="I don't have enough information in the knowledge base to answer that.",
                    can_answer=False,
                    sources=[],
                    confidence=0.2,
                    refusal_reason="No documents retrieved.",
                ),
                0,
            )

        query_tokens = set(re.findall(r"[a-z0-9]+", query.lower()))
        scored_sentences: List[Tuple[int, str, str]] = []
        sources = []
        for doc in documents:
            source = doc.metadata.get("source", "unknown")
            sources.append(source)
            for sentence in re.split(r"(?<=[.!?])\s+", doc.page_content):
                sentence = sentence.strip()
                if len(sentence) < 20:
                    continue
                sent_tokens = set(re.findall(r"[a-z0-9]+", sentence.lower()))
                overlap = len(query_tokens & sent_tokens)
                scored_sentences.append((overlap, sentence, source))

        scored_sentences.sort(key=lambda item: item[0], reverse=True)
        selected = [s for s in scored_sentences if s[0] > 0][:4]
        if not selected:
            return (
                SupportResponse(
                    answer="I don't have enough information in the knowledge base to answer that.",
                    can_answer=False,
                    sources=list(dict.fromkeys(sources)),
                    confidence=0.25,
                    refusal_reason="Retrieved context did not overlap the question.",
                ),
                0,
            )

        answer = " ".join(item[1] for item in selected)
        used_sources = list(dict.fromkeys(item[2] for item in selected))
        best_overlap = selected[0][0]
        confidence = min(1.0, 0.35 + 0.1 * best_overlap)
        return (
            SupportResponse(
                answer=answer,
                can_answer=True,
                sources=used_sources,
                confidence=confidence,
                refusal_reason=None,
            ),
            0,
        )


def get_backend(name: str = "ollama", **kwargs) -> LLMBackend:
    if name in ("extractive", "tfidf", "local"):
        return ExtractiveLocalBackend()
    return OllamaBackend(**kwargs)
