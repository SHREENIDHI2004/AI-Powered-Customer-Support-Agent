import json
import os
import sys
from typing import Any, Dict, List

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.agent import SupportAgent
from src.llm_backends import ExtractiveLocalBackend, OllamaBackend
from src.vectorstore import build_in_memory_index, load_faiss_index

EVAL_SET_PATH = os.path.join(os.path.dirname(__file__), "eval_set.json")
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "last_results.json")
README_BASELINE_NOTE = (
    "No prior README.md existed in this repository; TF-IDF/LSA + extractive numbers "
    "below are the captured baseline from the same eval_set.json."
)

# Targets from the implementation plan — report honestly; do not retune tests to hit them.
TARGETS = {
    "refusal_accuracy": 0.90,
    "false_answer_rate": 0.05,
    "avg_grounding_ratio": 0.75,
}


def load_eval_cases() -> List[Dict[str, Any]]:
    with open(EVAL_SET_PATH, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload["cases"]


def source_hit(retrieved_sources: List[str], expected_sources: List[str]) -> bool:
    if not expected_sources:
        return True
    retrieved = set(retrieved_sources)
    return any(src in retrieved for src in expected_sources)


def evaluate_agent(agent: SupportAgent, cases: List[Dict[str, Any]], label: str) -> Dict[str, Any]:
    rows = []
    source_hits = 0
    answerable = 0
    unanswerable = 0
    correct_refusals = 0
    false_answers = 0
    grounding_sum = 0.0
    grounding_n = 0

    print(f"\n=== Running eval: {label} ({len(cases)} cases) ===")
    for i, case in enumerate(cases, start=1):
        try:
            result = agent.ask(case["question"])
        except Exception as exc:
            print(f"  [{i:02d}/{len(cases)}] {case['id']} ERROR {exc}")
            rows.append(
                {
                    "id": case["id"],
                    "question": case["question"],
                    "should_answer": bool(case["should_answer"]),
                    "can_answer": False,
                    "guardrail_status": "error",
                    "repair_attempts": 0,
                    "grounding_ratio": 0.0,
                    "retrieved_sources": [],
                    "source_hit": False if case["should_answer"] else None,
                    "answer_preview": "",
                    "refusal_reason": str(exc),
                    "error": str(exc),
                }
            )
            if case["should_answer"]:
                answerable += 1
            else:
                unanswerable += 1
            continue
        should_answer = bool(case["should_answer"])
        can_answer = bool(result.response.can_answer)
        expected = case.get("expected_sources") or []
        hit = source_hit(result.retrieved_sources, expected) if should_answer else None

        if should_answer:
            answerable += 1
            if hit:
                source_hits += 1
        else:
            unanswerable += 1
            if not can_answer:
                correct_refusals += 1
            else:
                false_answers += 1

        if can_answer:
            grounding_sum += result.grounding_ratio
            grounding_n += 1

        row = {
            "id": case["id"],
            "question": case["question"],
            "should_answer": should_answer,
            "can_answer": can_answer,
            "guardrail_status": result.guardrail_status,
            "repair_attempts": result.repair_attempts,
            "grounding_ratio": round(result.grounding_ratio, 4),
            "retrieved_sources": result.retrieved_sources,
            "source_hit": hit,
            "answer_preview": (result.response.answer or "")[:180],
            "refusal_reason": result.response.refusal_reason,
        }
        rows.append(row)
        flag = "OK" if (should_answer and can_answer) or ((not should_answer) and (not can_answer)) else "MISS"
        print(f"  [{i:02d}/{len(cases)}] {case['id']} {flag} status={result.guardrail_status} can_answer={can_answer}")

    source_accuracy = (source_hits / answerable) if answerable else 0.0
    refusal_accuracy = (correct_refusals / unanswerable) if unanswerable else 0.0
    false_answer_rate = (false_answers / unanswerable) if unanswerable else 0.0
    avg_grounding = (grounding_sum / grounding_n) if grounding_n else 0.0

    metrics = {
        "label": label,
        "n_cases": len(cases),
        "n_answerable": answerable,
        "n_unanswerable": unanswerable,
        "source_accuracy": source_accuracy,
        "refusal_accuracy": refusal_accuracy,
        "false_answer_rate": false_answer_rate,
        "avg_grounding_ratio": avg_grounding,
        "n_answered_for_grounding": grounding_n,
    }
    return {"metrics": metrics, "rows": rows}


def pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def print_metrics(metrics: Dict[str, Any]) -> None:
    print(f"\n--- {metrics['label']} ---")
    print(f"Source accuracy:     {pct(metrics['source_accuracy'])}")
    print(f"Refusal accuracy:    {pct(metrics['refusal_accuracy'])}  (target >= {pct(TARGETS['refusal_accuracy'])})")
    print(f"False-answer rate:   {pct(metrics['false_answer_rate'])}  (target <= {pct(TARGETS['false_answer_rate'])})")
    print(f"Avg grounding ratio: {metrics['avg_grounding_ratio']:.3f}  (target >= {TARGETS['avg_grounding_ratio']:.2f})")


def print_comparison(baseline: Dict[str, Any], upgrade: Dict[str, Any]) -> None:
    keys = [
        ("source_accuracy", "Source accuracy", True),
        ("refusal_accuracy", "Refusal accuracy", True),
        ("false_answer_rate", "False-answer rate", False),
        ("avg_grounding_ratio", "Avg grounding ratio", True),
    ]
    print("\n=== Baseline vs. dense-embedding upgrade ===")
    print(f"{'Metric':<22} {'TF-IDF extractive':>20} {'Dense + Ollama':>18} {'Delta':>10}")
    for key, title, higher_better in keys:
        b = baseline[key]
        u = upgrade[key]
        delta = u - b
        if key == "avg_grounding_ratio":
            print(f"{title:<22} {b:>20.3f} {u:>18.3f} {delta:>+10.3f}")
        else:
            print(f"{title:<22} {pct(b):>20} {pct(u):>18} {delta * 100:>+9.1f} pp")
        del higher_better


def run_suite() -> Dict[str, Any]:
    cases = load_eval_cases()

    print("Loading dense FAISS index...")
    dense_store = load_faiss_index(model_type="dense")
    dense_agent = SupportAgent(backend=OllamaBackend(), store=dense_store)

    print("Building in-memory TF-IDF/LSA FAISS index (baseline)...")
    tfidf_store = build_in_memory_index(model_type="tfidf")
    baseline_agent = SupportAgent(backend=ExtractiveLocalBackend(), store=tfidf_store)

    baseline = evaluate_agent(baseline_agent, cases, "TF-IDF/LSA + extractive (baseline)")
    upgrade = evaluate_agent(dense_agent, cases, "Dense BGE + Ollama llama3.2")

    print_metrics(baseline["metrics"])
    print_metrics(upgrade["metrics"])
    print_comparison(baseline["metrics"], upgrade["metrics"])
    print(f"\nNote: {README_BASELINE_NOTE}")

    payload = {
        "baseline_note": README_BASELINE_NOTE,
        "targets": TARGETS,
        "baseline": baseline,
        "upgrade": upgrade,
    }
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote {RESULTS_PATH}")
    return payload


if __name__ == "__main__":
    run_suite()
