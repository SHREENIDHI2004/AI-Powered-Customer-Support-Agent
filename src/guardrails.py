import re
from typing import Iterable, List, Optional, Sequence, Tuple

from src.schemas import GuardrailResult, SupportResponse

STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "then", "than", "that", "this",
    "these", "those", "to", "of", "in", "on", "for", "with", "as", "at", "by",
    "from", "is", "are", "was", "were", "be", "been", "being", "it", "its",
    "you", "your", "we", "our", "they", "them", "their", "can", "will", "do",
    "does", "did", "not", "no", "yes", "please", "what", "when", "where", "how",
    "which", "who", "whom", "into", "about", "over", "under", "also", "any",
}

# Query-level OOD gate (pre-retrieval). Supported in-store payment providers
# (Klarna, PayPal, cards, wallets, "payment method") are intentionally NOT listed.
# Payment blocking is crypto-only, not a generic payment-method blocklist.
OOD_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (
        re.compile(r"\b(price\s*match|match(?:ing)?\s+(?:the\s+)?price).{0,40}\bamazon\b|\bamazon\b.{0,40}\b(price\s*match|match(?:ing)?\s+(?:the\s+)?price)\b", re.I),
        "We do not handle Amazon price-match requests in this support channel.",
    ),
    (
        re.compile(r"\b(ebay|walmart\s+marketplace|craigslist)\b", re.I),
        "Third-party marketplace price matching is out of scope for this agent.",
    ),
    (
        re.compile(
            r"\b(bitcoin|ethereum|usdt|crypto(?:currency)?s?)\b",
            re.I,
        ),
        "Cryptocurrency payment requests are out of scope for this agent.",
    ),
    (
        re.compile(r"\b(electric\s+scooters?|e-?bikes?|motorized\s+(?:scooter|bike))\b", re.I),
        "Electric scooters and e-bikes are not in our catalog.",
    ),
    (
        re.compile(
            r"\b(iphone\s*\d+|samsung\s+galaxy|smartphones?\b|mobile\s+handsets?|sell(?:ing)?\s+(?:an?\s+)?iphone)\b",
            re.I,
        ),
        "We do not sell smartphones; that product line is out of scope.",
    ),
    (
        re.compile(r"\b(grocer(?:y|ies)|perishable\s+food|fresh\s+food)\b", re.I),
        "Grocery and perishable food questions are out of scope.",
    ),
    (
        re.compile(r"\b(car\s+batter(?:y|ies)|automotive|tires?|auto\s+parts?)\b", re.I),
        "Automotive parts are out of scope.",
    ),
    (
        re.compile(r"\b(tesla|stock\s+market|medical\s+advice|diagnos(?:e|is)|lawsuit|legal\s+advice)\b", re.I),
        "This question is outside store policy and product support.",
    ),
    # Live / real-time data a static markdown KB cannot provide.
    (
        re.compile(
            r"\b(order\s*#\s*\d+|status\s+of\s+(?:my\s+)?order|track(?:ing)?\s+(?:my\s+)?order|"
            r"where\s+is\s+(?:my\s+)?order|units?\b.{0,40}\bright\s+now\b|"
            r"\bin\s+stock\s+right\s+now\b|warehouse\s+(?:right\s+now|inventory)|"
            r"how\s+many\s+units\b|live\s+inventory|current\s+stock)\b",
            re.I,
        ),
        "Live order status and real-time inventory are outside this knowledge-base agent.",
    ),
]

MIN_GROUNDING_RATIO = 0.35
# Hard floor: answers below this are refused even if the model set can_answer=true.
HARD_GROUNDING_FLOOR = 0.5
MAX_L2_DISTANCE = 1.15


def tokenize(text: str) -> List[str]:
    tokens = re.findall(r"[a-z0-9]+", (text or "").lower())
    return [t for t in tokens if t not in STOPWORDS and len(t) > 2]


def lexical_grounding_ratio(answer: str, context: str) -> float:
    """Fraction of content tokens in the answer that also appear in retrieved context."""
    answer_tokens = set(tokenize(answer))
    if not answer_tokens:
        return 0.0
    context_tokens = set(tokenize(context))
    if not context_tokens:
        return 0.0
    return len(answer_tokens & context_tokens) / len(answer_tokens)


def detect_ood_query(query: str) -> Optional[str]:
    for pattern, reason in OOD_PATTERNS:
        if pattern.search(query or ""):
            return reason
    return None


def retrieval_too_far(distances: Sequence[float], max_l2: float = MAX_L2_DISTANCE) -> bool:
    if not distances:
        return True
    return min(distances) > max_l2


def make_refusal(reason: str, confidence: float = 0.9) -> SupportResponse:
    return SupportResponse(
        answer="I don't have enough grounded store-policy or catalog information to answer that.",
        can_answer=False,
        sources=[],
        confidence=confidence,
        refusal_reason=reason,
    )


def apply_guardrails(
    response: SupportResponse,
    context: str,
    query: str,
    distances: Iterable[float],
) -> GuardrailResult:
    reasons: List[str] = []
    distance_list = list(distances)

    # OOD is applied on the query in SupportAgent.ask *before* retrieval/generation.
    # Do not re-run it here: a well-grounded generated answer must not be replaced
    # by a keyword gate after the fact.

    if retrieval_too_far(distance_list):
        reason = "Retrieved context is too dissimilar to the question (distance threshold)."
        refused = make_refusal(reason)
        return GuardrailResult(
            passed=False,
            status="refuse",
            grounding_ratio=0.0,
            reasons=[reason],
            response=refused,
        )

    grounding = lexical_grounding_ratio(response.answer, context)

    # Hard floor applies to every generated answer, regardless of can_answer.
    if grounding < HARD_GROUNDING_FLOOR:
        reason = (
            f"Answer failed hard grounding floor "
            f"(ratio={grounding:.2f} < {HARD_GROUNDING_FLOOR})."
        )
        refused = make_refusal(reason, confidence=min(response.confidence, 0.4))
        return GuardrailResult(
            passed=False,
            status="refuse",
            grounding_ratio=grounding,
            reasons=[reason],
            response=refused,
        )

    if response.can_answer and grounding < MIN_GROUNDING_RATIO:
        reason = (
            f"Answer failed lexical grounding (ratio={grounding:.2f} < {MIN_GROUNDING_RATIO})."
        )
        refused = make_refusal(reason, confidence=min(response.confidence, 0.4))
        return GuardrailResult(
            passed=False,
            status="refuse",
            grounding_ratio=grounding,
            reasons=[reason],
            response=refused,
        )

    if not response.can_answer:
        answer_text = (response.answer or "").strip()
        combined = f"{answer_text} {response.refusal_reason or ''}".lower()
        coverage_refusal = any(
            marker in combined
            for marker in (
                "out of scope",
                "not covered",
                "don't have enough",
                "do not have enough",
                "unable to",
                "cannot access",
                "no information",
            )
        )
        # can_answer=false is often a policy "no" (limits, eligibility), not OOD.
        # Only promote when the model still produced a grounded factual reply.
        if (
            answer_text
            and grounding >= HARD_GROUNDING_FLOOR
            and not coverage_refusal
        ):
            promoted = response.model_copy(update={"can_answer": True, "refusal_reason": None})
            return GuardrailResult(
                passed=True,
                status="pass",
                grounding_ratio=grounding,
                reasons=["Kept grounded answer; model can_answer=false was not treated as OOD."],
                response=promoted,
            )
        return GuardrailResult(
            passed=True,
            status="refuse",
            grounding_ratio=grounding,
            reasons=reasons or ["Model flagged the query as unanswerable."],
            response=response,
        )

    return GuardrailResult(
        passed=True,
        status="pass",
        grounding_ratio=grounding,
        reasons=reasons,
        response=response,
    )
