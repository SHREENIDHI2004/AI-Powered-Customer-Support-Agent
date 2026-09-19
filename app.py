import json
import os

import streamlit as st

from src.agent import SupportAgent
from src.llm_backends import OllamaBackend

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_PATH = os.path.join(ROOT_DIR, "eval", "last_results.json")


@st.cache_resource(show_spinner="Loading FAISS index and embeddings...")
def get_cached_agent() -> SupportAgent:
    return SupportAgent(backend=OllamaBackend())


def render_metrics(result) -> None:
    cols = st.columns(3)
    cols[0].metric("Grounding ratio", f"{result.grounding_ratio:.2f}")
    cols[1].metric("Guardrail", result.guardrail_status.upper())
    cols[2].metric("JSON repairs", result.repair_attempts)
    if result.retrieved_sources:
        st.caption("Retrieved sources: " + ", ".join(result.retrieved_sources))
    if result.response.sources:
        st.caption("Cited sources: " + ", ".join(result.response.sources))
    if result.distances:
        st.caption("L2 distances: " + ", ".join(f"{d:.3f}" for d in result.distances))
    if result.response.refusal_reason:
        st.warning(result.response.refusal_reason)


def main() -> None:
    st.set_page_config(page_title="Store Support Agent", page_icon="🛍️", layout="wide")
    st.title("AI Customer Support Agent")
    st.caption("Dense retrieval (BGE) + FAISS + Ollama llama3.2 with JSON schema and grounding guardrails.")

    agent = get_cached_agent()
    tab_chat, tab_eval = st.tabs(["Live support chat", "Held-out evaluation"])

    with tab_chat:
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg.get("meta"):
                    st.caption(msg["meta"])

        prompt = st.chat_input("Ask about returns, shipping, products, warranty...")
        if prompt:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Retrieving policy context and generating a grounded reply..."):
                    result = agent.ask(prompt)
                st.markdown(result.response.answer)
                render_metrics(result)
                meta = (
                    f"status={result.guardrail_status} | "
                    f"grounding={result.grounding_ratio:.2f} | "
                    f"can_answer={result.response.can_answer}"
                )
            st.session_state.messages.append(
                {"role": "assistant", "content": result.response.answer, "meta": meta}
            )

    with tab_eval:
        st.subheader("Benchmark vs TF-IDF baseline")
        st.write(
            "Runs `eval/eval_set.json` against TF-IDF/LSA + extractive retrieval and "
            "dense BGE + Ollama. Targets are not forced: refusal ≥ 90%, false-answer ≤ 5%, grounding ≥ 0.75."
        )
        if os.path.exists(RESULTS_PATH):
            with open(RESULTS_PATH, "r", encoding="utf-8") as f:
                payload = json.load(f)
            b = payload["baseline"]["metrics"]
            u = payload["upgrade"]["metrics"]
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**TF-IDF/LSA + extractive**")
                st.metric("Source accuracy", f"{b['source_accuracy']*100:.1f}%")
                st.metric("Refusal accuracy", f"{b['refusal_accuracy']*100:.1f}%")
                st.metric("False-answer rate", f"{b['false_answer_rate']*100:.1f}%")
                st.metric("Avg grounding", f"{b['avg_grounding_ratio']:.3f}")
            with c2:
                st.markdown("**Dense BGE + Ollama**")
                st.metric("Source accuracy", f"{u['source_accuracy']*100:.1f}%")
                st.metric("Refusal accuracy", f"{u['refusal_accuracy']*100:.1f}%")
                st.metric("False-answer rate", f"{u['false_answer_rate']*100:.1f}%")
                st.metric("Avg grounding", f"{u['avg_grounding_ratio']:.3f}")
        else:
            st.info("No eval results yet. Run `python eval/run_eval.py` or use the button below.")

        if st.button("Run evaluation suite", type="primary"):
            with st.spinner("Evaluating both backends. This can take several minutes..."):
                from eval.run_eval import run_suite

                run_suite()
            st.rerun()


if __name__ == "__main__":
    main()
