"""AfriMed Tutor — Streamlit entry point.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import streamlit as st
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env", override=True)

# ── Page config must come first ──────────────────────────────────────────────
st.set_page_config(
    page_title="AfriMed Tutor",
    page_icon="🩺",
    layout="wide",
)

# ── Lazy imports so Streamlit shows a nice error if deps are missing ─────────
@st.cache_resource(show_spinner="Loading retrieval index…")
def _load_retriever():
    from tutor.retriever import build_retriever
    return build_retriever()


@st.cache_resource(show_spinner="Loading LLM client…")
def _load_llm_client():
    from tutor.llm_client import build_llm_client
    return build_llm_client()


@st.cache_resource(show_spinner="Loading MCQ bank…")
def _load_quiz_loader():
    from tutor.quiz import QuizLoader
    return QuizLoader()


@st.cache_resource(show_spinner="Loading SAQ cases…")
def _load_saq_loader():
    from tutor.explain import SAQLoader
    try:
        return SAQLoader()
    except FileNotFoundError:
        return None


# ── Disclaimer (always visible) ──────────────────────────────────────────────
st.info(
    "⚠️ **Study aid only.** AfriMed Tutor is designed for learning, not for clinical \
decision-making. Never use this tool to guide patient care.",
    icon="🩺",
)

# ── Sidebar: mode selector and settings ──────────────────────────────────────
st.sidebar.title("AfriMed Tutor")
st.sidebar.caption("Guideline-grounded study assistant")

MODE = st.sidebar.radio(
    "Mode",
    ["Ask", "Quiz", "Explain (SAQ)"],
    index=0,
    help="Ask: free-text question  |  Quiz: MCQ practice  |  Explain: clinical scenario",
)

with st.sidebar.expander("⚙ Settings"):
    top_k = st.slider("Top-k chunks", 1, 10, int(os.getenv("RETRIEVER_TOP_K", "5")))
    backend = st.selectbox(
        "Retriever",
        ["dense", "sparse"],
        index=0 if os.getenv("RETRIEVER_BACKEND", "dense") == "dense" else 1,
    )
    show_chunks = st.checkbox("Show retrieved guideline excerpts", value=False)
    show_related = st.checkbox("Show related practice questions", value=True)

os.environ["RETRIEVER_TOP_K"] = str(top_k)
os.environ["RETRIEVER_BACKEND"] = backend


def _render_chunks_expander(chunks) -> None:
    if not chunks:
        st.caption("No relevant guideline excerpts retrieved.")
        return
    for i, c in enumerate(chunks, 1):
        with st.expander(f"Excerpt {i}: {c.source_doc} — {c.section_title}", expanded=False):
            st.write(c.text)
            st.caption(f"Score: {c.score:.3f}" + (f" | Page: {c.page_number}" if c.page_number else ""))


def _render_related(related) -> None:
    if not related:
        return
    st.markdown("---")
    st.markdown("**Related practice questions:**")
    for q in related:
        st.markdown(f"- [{q.specialty}] {q.question}")


def _latency_tokens_caption(resp) -> None:
    st.caption(
        f"Model: {resp.model_name}  |  Tokens: {resp.input_tokens} in / "
        f"{resp.output_tokens} out  |  Latency: {resp.latency_ms:.0f} ms"
    )


# ── Mode: Ask ────────────────────────────────────────────────────────────────
if MODE == "Ask":
    st.header("Ask a clinical question")
    st.caption("Ask anything covered by the African clinical guidelines in our corpus.")

    question = st.text_area(
        "Your question",
        placeholder="e.g. What is the first-line treatment for malaria in a child under 5 in Ghana?",
        height=100,
    )

    if st.button("Get answer", type="primary", disabled=not question.strip()):
        from tutor.orchestrator import handle_ask

        retriever = _load_retriever()
        llm_client = _load_llm_client()

        with st.spinner("Retrieving guidelines and generating answer…"):
            resp = handle_ask(
                question=question,
                llm_client=llm_client,
                retriever=retriever,
                k=top_k,
            )

        if resp.answer.strip():
            st.markdown(resp.answer)
        else:
            st.warning("The model returned an empty response. Please try again.")
        _latency_tokens_caption(resp)

        if show_chunks:
            _render_chunks_expander(resp.retrieved_chunks)
        if show_related:
            _render_related(resp.related_questions)


# ── Mode: Quiz ────────────────────────────────────────────────────────────────
elif MODE == "Quiz":
    st.header("MCQ Practice")
    st.caption("Drawn from the AfriMed-QA question bank.")

    quiz_loader = _load_quiz_loader()

    # Specialty filter
    all_specialties = ["Any"] + quiz_loader.specialties
    specialty_choice = st.selectbox("Specialty filter", all_specialties)

    if "quiz_item" not in st.session_state:
        st.session_state.quiz_item = None
        st.session_state.quiz_submitted = False
        st.session_state.quiz_response = None

    col1, col2 = st.columns([1, 5])
    with col1:
        if st.button("New question"):
            spec = None if specialty_choice == "Any" else specialty_choice
            try:
                st.session_state.quiz_item = quiz_loader.sample(specialty=spec)
            except ValueError as e:
                st.error(str(e))
                st.stop()
            st.session_state.quiz_submitted = False
            st.session_state.quiz_response = None

    item = st.session_state.quiz_item
    if item is None:
        st.info("Click **New question** to start.")
    else:
        st.markdown(f"**Specialty:** {item.specialty}")
        st.markdown(f"**Question:** {item.question}")

        student_choice = st.radio(
            "Your answer",
            options=[f"{o.key}. {o.text}" for o in item.options],
            index=None,
            key=f"quiz_radio_{item.question_id}",
            disabled=st.session_state.quiz_submitted,
        )

        if not st.session_state.quiz_submitted:
            if st.button(
                "Submit answer",
                type="primary",
                disabled=student_choice is None,
            ):
                from tutor.orchestrator import handle_quiz_submit

                retriever = _load_retriever()
                llm_client = _load_llm_client()
                choice_key = student_choice.split(".")[0].strip() if student_choice else ""

                with st.spinner("Generating explanation…"):
                    resp = handle_quiz_submit(
                        question_id=item.question_id,
                        student_choice=choice_key,
                        llm_client=llm_client,
                        retriever=retriever,
                        quiz_loader=quiz_loader,
                    )
                st.session_state.quiz_submitted = True
                st.session_state.quiz_response = resp
                st.rerun()

        if st.session_state.quiz_submitted and st.session_state.quiz_response:
            resp = st.session_state.quiz_response
            if resp.is_correct:
                st.success("✅ Correct!")
            else:
                st.error(f"❌ Incorrect. Correct answer: **{item.gold_answer}**")
            st.markdown(resp.explanation)
            _latency_tokens_caption(resp)
            if show_chunks:
                _render_chunks_expander(resp.retrieved_chunks)
            if show_related:
                _render_related(resp.related_questions)


# ── Mode: Explain (SAQ) ───────────────────────────────────────────────────────
elif MODE == "Explain (SAQ)":
    st.header("Clinical Scenario — Explain your reasoning")
    st.caption(
        "Work through a short-answer clinical scenario. Submit your reasoning "
        "and receive feedback grounded in African clinical guidelines."
    )

    saq_loader = _load_saq_loader()
    if saq_loader is None:
        st.warning(
            "SAQ data file not found. Run `python data/load_afrimedqa.py` first."
        )
        st.stop()

    if "saq_item" not in st.session_state:
        st.session_state.saq_item = None
        st.session_state.saq_submitted = False
        st.session_state.saq_response = None

    if st.button("New scenario"):
        try:
            st.session_state.saq_item = saq_loader.sample()
        except ValueError as e:
            st.error(str(e))
            st.stop()
        st.session_state.saq_submitted = False
        st.session_state.saq_response = None

    case = st.session_state.saq_item
    if case is None:
        st.info("Click **New scenario** to start.")
    else:
        st.markdown(f"**Specialty:** {case.specialty}")
        st.markdown(f"**Scenario:**\n\n{case.scenario}")

        reasoning = st.text_area(
            "Your reasoning and management plan",
            height=150,
            disabled=st.session_state.saq_submitted,
            key=f"saq_text_{case.case_id}",
        )

        if not st.session_state.saq_submitted:
            if st.button(
                "Submit reasoning",
                type="primary",
                disabled=not reasoning.strip(),
            ):
                from tutor.orchestrator import handle_explain

                retriever = _load_retriever()
                llm_client = _load_llm_client()

                with st.spinner("Comparing reasoning…"):
                    resp = handle_explain(
                        case_id=case.case_id,
                        student_reasoning=reasoning,
                        llm_client=llm_client,
                        retriever=retriever,
                        saq_loader=saq_loader,
                    )
                st.session_state.saq_submitted = True
                st.session_state.saq_response = resp
                st.rerun()

        if st.session_state.saq_submitted and st.session_state.saq_response:
            resp = st.session_state.saq_response
            st.markdown(resp.comparison)
            _latency_tokens_caption(resp)
            if show_chunks:
                _render_chunks_expander(resp.retrieved_chunks)
