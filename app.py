# app.py
# Agentic AI for Reliable Academic Literature Review
# MSc Data Science — University of Hertfordshire
# Run: streamlit run app.py

import os
import sys
import time
import tempfile
from pathlib import Path
from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import html

from app_styles import CSS
from src.document_reader import load_input, extract_topic_from_document

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

st.set_page_config(
    page_title="LitReview Agent",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
def init_session():
    defaults = {
        "page": "🏠 Home",
        "pipeline_state": None,
        "baseline_state": None,
        "pipeline_ran": False,
        "baseline_ran": False,
        "run_history": [],
        "topic": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_session()


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------
def render_loader(text: str):
    st.markdown(
        f"""
        <div class="floating-loader">
            <div class="loader-ring"></div>
            <div class="loader-text">{text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_card(value, label, colour="blue"):
    st.markdown(
        f"""
        <div class="metric-card metric-card-{colour}">
            <div class="metric-value">{value}</div>
            <div class="metric-label">{label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_open(title: str):
    st.markdown(
        f"""
        <div class="section-card">
            <div class="section-title">{title}</div>
        """,
        unsafe_allow_html=True,
    )


def section_close():
    st.markdown("</div>", unsafe_allow_html=True)


def render_header():
    st.markdown(
        """
        <div class="hero-shell">
            <div class="hero-badge">Agentic AI • Hallucination Mitigation • Verified Review Pipeline</div>
            <h1 class="hero-title">
                Reliable Academic Literature Review with <span>Self-Correcting AI Agents</span>
            </h1>
            <p class="hero-subtitle">
                A premium research workflow for planning, retrieving, summarising, verifying,
                and assembling literature reviews with stronger citation reliability and cleaner outputs.
            </p>
            <div class="hero-stat-row">
                <div class="hero-stat-pill">Planner → Searcher → Summariser → Verifier → Assembler</div>
                <div class="hero-stat-pill">OpenAlex + Semantic Retrieval</div>
                <div class="hero-stat-pill">MSc Data Science • University of Hertfordshire</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def render_sidebar():
    with st.sidebar:
        st.markdown(
            f"""
            <div class="sidebar-brand">
                <div style="font-size:2.1rem;">📚</div>
                <div class="sidebar-brand-title">LitReview Agent</div>
                <div class="sidebar-brand-subtitle">
                    Premium multi-agent frontend for literature review generation,
                    verification, and hallucination analysis.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="sidebar-section-label">Navigation</div>', unsafe_allow_html=True)
        page = st.radio(
            "Navigation",
            options=["🏠 Home", "🧪 Run Experiment", "📊 Evaluation", "🕘 History", "ℹ️ About"],
            label_visibility="collapsed",
        )

        st.markdown('<div class="sidebar-section-label">Model Settings</div>', unsafe_allow_html=True)
        model = st.selectbox(
            "LLM Model",
            [
                "llama-3.3-70b-versatile",
                "meta-llama/llama-4-maverick-17b-128e-instruct",
                "meta-llama/llama-4-scout-17b-16e-instruct",
                "llama-3.1-8b-instant",
            ],
            index=0,
        )
        max_papers = st.slider("Papers per query", 3, 10, 5)

        st.markdown(
            f"""
            <div class="sidebar-mini-card">
                <div style="font-size:0.72rem; color:#7f90af; text-transform:uppercase; letter-spacing:.14em; font-weight:700;">
                    Project
                </div>
                <div style="margin-top:8px; font-size:0.9rem; font-weight:700; color:#eef4ff;">
                    Hallucination-aware literature review system
                </div>
                <div style="margin-top:8px; font-size:0.78rem; line-height:1.6; color:#9db0cf;">
                    MSc Data Science • University of Hertfordshire<br/>
                    UI version • {datetime.now().strftime("%B %Y")}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    return page, model, max_papers


# ---------------------------------------------------------------------------
# Home
# ---------------------------------------------------------------------------
def page_home():
    render_header()

    st.markdown("### Pipeline Architecture")
    cols = st.columns(5)
    agents = [
        ("🧭", "Planner", "Decomposes the research topic into focused sub-queries."),
        ("🔎", "Searcher", "Fetches relevant scholarly papers from trusted metadata sources."),
        ("📝", "Summariser", "Produces review-ready synthesis across selected papers."),
        ("🛡️", "Verifier", "Checks claims and citations against metadata and retrieval evidence."),
        ("🧩", "Assembler", "Removes weak references and finalises the review output."),
    ]

    for col, (icon, name, desc) in zip(cols, agents):
        with col:
            st.markdown(
                f"""
                <div class="pipeline-node">
                    <div class="pipeline-node-icon">{icon}</div>
                    <div class="pipeline-node-title">{name}</div>
                    <div class="pipeline-node-desc">{desc}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<div style='height:18px;'></div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        section_open("Research Question")
        st.markdown(
            """
            <div class="result-highlight">
                <div class="soft-text" style="font-style:italic; line-height:1.8;">
                    How effectively can an agentic AI system perform autonomous academic literature review
                    while using self-correcting mechanisms to detect and reduce LLM hallucinations in
                    generated summaries and citations?
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        section_close()

        section_open("Hypotheses")
        st.markdown(
            """
            <div style="display:flex; gap:10px; margin-bottom:12px; align-items:flex-start;">
                <span class="badge-hallucinated">H0</span>
                <span class="soft-text">Self-correction adds no significant improvement over a single LLM baseline.</span>
            </div>
            <div style="display:flex; gap:10px; align-items:flex-start;">
                <span class="badge-valid">H1</span>
                <span class="soft-text">The multi-agent verifier system significantly reduces hallucination rate and improves citation accuracy.</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        section_close()

    with col2:
        section_open("Technology Stack")
        tech_items = [
            ("🐍", "Python 3.13", "Core implementation language"),
            ("🧠", "LangGraph", "Multi-agent orchestration"),
            ("⚡", "Groq LLM", "Fast inference"),
            ("📚", "OpenAlex API", "Academic metadata source"),
            ("🧲", "FAISS + SBERT", "Semantic retrieval"),
            ("🎨", "Streamlit", "Frontend interface"),
        ]
        for icon, tech, desc in tech_items:
            st.markdown(
                f"""
                <div style="display:flex; align-items:center; gap:12px; padding:9px 0; border-bottom:1px solid rgba(138,180,248,.08);">
                    <span style="font-size:1.1rem; width:24px;">{icon}</span>
                    <span style="font-weight:700; color:#eef4ff; width:150px;">{tech}</span>
                    <span class="soft-text" style="font-size:.84rem;">{desc}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        section_close()

        section_open("Key Metrics")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            metric_card("5", "AI Agents", "blue")
        with c2:
            metric_card("OpenAlex", "Data Source", "teal")
        with c3:
            metric_card("H-Rate", "Primary Metric", "orange")
        with c4:
            metric_card("Fuzzy", "Citation Match", "green")
        section_close()

    colbtn = st.columns([3, 2, 3])[1]
    with colbtn:
        if st.button("Start Experiment", use_container_width=True):
            st.session_state.page = "🧪 Run Experiment"
            st.rerun()


# ---------------------------------------------------------------------------
# Run Experiment
# ---------------------------------------------------------------------------
def page_run_experiment(model: str, max_papers: int):
    render_header()
    st.markdown("### Run Literature Review Pipeline")

    section_open("Research Topic")
    topic = st.text_area(
        "Enter your research topic",
        value=st.session_state.get("topic", ""),
        height=95,
        placeholder="e.g. Agentic AI for reliable academic literature review and hallucination mitigation in large language models",
        label_visibility="collapsed",
        key="topic_input",
    )
    st.session_state.topic = topic
    section_close()

    section_open("Attach Research Document")
    st.markdown(
        """
        <div class="soft-text" style="margin-bottom: 10px; line-height:1.7;">
            Upload a PDF, DOCX, or TXT research brief and the system will try to extract
            a clean research topic automatically.
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Upload PDF, DOCX, or TXT file",
        type=["pdf", "docx", "txt"],
        help="Upload a research brief, paper, or topic document.",
        label_visibility="collapsed",
    )

    if uploaded_file is not None:
        suffix = Path(uploaded_file.name).suffix.lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        try:

            with st.spinner(f"Reading {uploaded_file.name}..."):
                raw_text = load_input(tmp_path, mode="auto")

            if raw_text and len(raw_text.strip()) > 500:
                with st.spinner("Extracting research topic from document..."):
                    extracted_topic = extract_topic_from_document(raw_text)

                if extracted_topic and extracted_topic.strip():
                    st.session_state.topic = extracted_topic.strip()
                    st.success(f"Topic extracted: {extracted_topic.strip()}")
                else:
                    st.warning("Document was read, but no clean topic could be extracted automatically.")
            else:
                if raw_text and raw_text.strip():
                    st.session_state.topic = raw_text.strip()
                    st.success("Document loaded and used as the topic text.")
                else:
                    st.warning("The uploaded document appears empty or unreadable.")

        except Exception as e:
            st.error(f"Could not read uploaded file: {e}")
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
    section_close()

    st.markdown("#### Or choose a preset topic")
    preset_cols = st.columns(3)
    presets = [
        "Hallucination detection and mitigation in large language models",
        "Retrieval augmented generation for reducing LLM hallucinations",
        "Multi-agent systems for automated academic literature review",
    ]
    for i, (col, preset) in enumerate(zip(preset_cols, presets)):
        with col:
            if st.button(preset[:45] + "...", key=f"preset_{i}", use_container_width=True):
                st.session_state.topic = preset
                st.rerun()

    st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        run_experimental = st.button("Run Experimental Multi-Agent Pipeline", use_container_width=True)
    with col2:
        run_baseline = st.button("Run Baseline Single LLM", use_container_width=True)
    with col3:
        run_both = st.button("Run Both Full Comparison", use_container_width=True)

    active_topic = st.session_state.get("topic", "").strip()

    if (run_experimental or run_baseline or run_both) and not active_topic:
        st.error("Please enter a research topic before running.")
        return

    if run_experimental or run_both:
        st.session_state.topic = active_topic
        run_experimental_pipeline(active_topic)

    if run_baseline or run_both:
        st.session_state.topic = active_topic
        run_baseline_pipeline(active_topic)

    if st.session_state.pipeline_state or st.session_state.baseline_state:
        render_results()


# ---------------------------------------------------------------------------
# Pipelines
# ---------------------------------------------------------------------------
def run_experimental_pipeline(topic: str):
    st.markdown("---")
    st.markdown("### Experimental Pipeline")
    render_loader("Launching planner, retrieval, summarisation, verification, and assembly workflow...")

    steps = [
        ("Planner Agent", "Decomposing topic into sub-queries..."),
        ("Search Agent", "Fetching papers from OpenAlex..."),
        ("Summariser Agent", "Writing literature review..."),
        ("Verifier Agent", "Verifying citations..."),
        ("Assembler Agent", "Assembling final review..."),
    ]

    placeholder = st.empty()
    progress_bar = st.progress(0)
    status_text = st.empty()

    def render_steps(statuses):
        with placeholder.container():
            for i, (name, desc) in enumerate(steps):
                css_class = (
                    "step-done" if statuses[i] == "done"
                    else "step-running" if statuses[i] == "running"
                    else "step-error" if statuses[i] == "error"
                    else "step-pending"
                )
                icon = "✅" if statuses[i] == "done" else "⏳" if statuses[i] == "running" else "❌" if statuses[i] == "error" else "•"
                st.markdown(
                    f"""
                    <div class="agent-step {css_class}">
                        <span>{icon}</span>
                        <span><b>{name}</b> &nbsp; <span style="font-weight:500; opacity:.82;">{desc}</span></span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    statuses = ["pending"] * 5

    try:
        from graph.workflow_graph import run_workflow

        statuses[0] = "running"
        render_steps(statuses)
        progress_bar.progress(10)
        status_text.info("Planner: generating sub-queries...")

        final_state = run_workflow(topic)

        for i in range(5):
            statuses[i] = "done"
            render_steps(statuses)
            progress_bar.progress(min(20 + i * 16, 100))
            time.sleep(0.25)

        progress_bar.progress(100)
        status_text.success("Experimental pipeline complete!")

        st.session_state.pipeline_state = final_state
        st.session_state.pipeline_ran = True

        # Save topic to dynamic ablation history
        import json, os
        _hist_path = os.path.join(
            r"C:\Users\BEST LAPTOP\Desktop\FYP-LLM",
            "evaluation_results", "topic_history.json"
        )
        os.makedirs(os.path.dirname(_hist_path), exist_ok=True)
        try:
            with open(_hist_path, "r", encoding="utf-8") as _f:
                _hist = json.load(_f)
        except (FileNotFoundError, json.JSONDecodeError):
            _hist = []

        # Add this topic if not already in history
        _topic_entry = {
            "topic": topic,
            "timestamp": __import__('datetime').datetime.now().isoformat(),
            "hall_pct": final_state.get("hallucination_rate", 0) * 100
                        if isinstance(final_state.get("hallucination_rate"), float)
                        else final_state.get("hallucination_rate", 0),
            "valid": final_state.get("valid_citations", 0),
            "hallucinated": final_state.get("hallucinated_citations", 0),
            "total": final_state.get("total_citations", 0),
            "latency": final_state.get("latency_seconds", 0),
            "model": final_state.get("selected_model", "unknown"),
            "papers": final_state.get("papers_retrieved", 0),
        }
        # Avoid duplicates
        if not any(h.get("topic") == topic for h in _hist):
            _hist.append(_topic_entry)
            with open(_hist_path, "w", encoding="utf-8") as _f:
                json.dump(_hist, _f, indent=2)
        st.session_state.run_history.append(
            {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "topic": topic[:60],
                "system": "Experimental",
                "papers": len(final_state.get("papers", [])),
                "citations": final_state.get("total_citations", 0),
                "hallucinated": final_state.get("hallucinated_citations", 0),
                "hallucination_rate": f"{final_state.get('hallucination_rate', 0):.1%}",
            }
        )
    except Exception as e:
        if "running" in statuses:
            statuses[statuses.index("running")] = "error"
        render_steps(statuses)
        status_text.error(f"Pipeline error: {str(e)}")
        st.exception(e)


def run_baseline_pipeline(topic: str):
    st.markdown("---")
    st.markdown("### Baseline Pipeline")
    render_loader("Running single-LLM baseline without verifier safeguards...")

    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        from graph.baseline_graph import run_baseline

        status_text.info("Searching papers using raw topic...")
        progress_bar.progress(30)

        baseline_state = run_baseline(topic)

        progress_bar.progress(100)
        status_text.success("Baseline pipeline complete!")

        st.session_state.baseline_state = baseline_state
        st.session_state.baseline_ran = True
        st.session_state.run_history.append(
            {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "topic": topic[:60],
                "system": "Baseline",
                "papers": len(baseline_state.get("papers", [])),
                "citations": baseline_state.get("total_citations", 0),
                "hallucinated": baseline_state.get("hallucinated_citations", 0),
                "hallucination_rate": f"{baseline_state.get('hallucination_rate', 0):.1%}",
            }
        )
    except Exception as e:
        status_text.error(f"Baseline error: {str(e)}")
        st.exception(e)


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
def render_system_results(state: dict, system: str):
    is_exp = system == "experimental"

    if is_exp:
        review_text = state.get("final_review") or state.get("draft_review", "")
    else:
        review_text = state.get("review_text", "")

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        metric_card(len(state.get("papers", [])), "Papers Retrieved", "blue")
    with c2:
        metric_card(state.get("total_citations", 0), "Total Citations", "blue")
    with c3:
        metric_card(state.get("valid_citations", 0), "Valid", "green")
    with c4:
        metric_card(state.get("hallucinated_citations", 0), "Hallucinated", "red")
    with c5:
        rate = state.get("hallucination_rate", 0)
        colour = "green" if rate == 0 else "orange" if rate <= 0.2 else "red"
        metric_card(f"{rate:.1%}", "Hallucination Rate", colour)

    if is_exp:
        ca, cb = st.columns(2)
        with ca:
            metric_card(len(state.get("subqueries", [])), "Sub-queries Used", "teal")
        with cb:
            changes = state.get("changes", {})
            metric_card(changes.get("words_removed", 0), "Words Removed", "orange")

        latency = state.get("latency_seconds", 0)
        tokens = state.get("token_estimate", 0)
        if latency or tokens:
            cc1, cc2, cc3 = st.columns(3)
            with cc1:
                metric_card(f"{latency}s", "Pipeline Latency", "teal")
            with cc2:
                metric_card(f"{tokens:,}", "Token Estimate", "blue")
            with cc3:
                est_cost = (tokens / 1_000_000) * 0.79 if tokens else 0
                metric_card(f"${est_cost:.4f}", "Est. Cost USD", "orange")
        # Show MAB selected model if available
        selected_model = state.get("selected_model", "")
        topic_type     = state.get("topic_type",     "")
        if selected_model:
            st.markdown(
                f'<div style="display:flex;gap:16px;padding:8px 12px;'
                f'background:rgba(30,58,138,0.3);border-radius:8px;'
                f'margin-bottom:12px;font-size:0.85rem;">'
                f'<span style="color:#94a3b8;">MAB Selected Model:</span>'
                f'<span style="color:#eef4ff;font-weight:600;">'
                f'{selected_model}</span>'
                f'&nbsp;|&nbsp;'
                f'<span style="color:#94a3b8;">Topic Type:</span>'
                f'<span style="color:#eef4ff;font-weight:600;">'
                f'{topic_type}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
            

    st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)

    col_review, col_citations = st.columns([3, 2])

    with col_review:
        section_open("Generated Review")
        st.markdown(f'<div class="review-box">{review_text}</div>', unsafe_allow_html=True)
        section_close()

        st.download_button(
            label="Download Review",
            data=review_text,
            file_name=f"review_{system}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
        )

    with col_citations:
        section_open("Citation Verification")
        citations = state.get("citation_details", [])

        if citations:
            for cit in citations:
                raw = getattr(cit, "raw_reference", str(cit))
                valid = getattr(cit, "valid", None)
                reason = getattr(cit, "error_reason", None)

                safe_raw = html.escape(str(raw)) if raw else ""
                safe_reason = html.escape(str(reason)) if reason else ""

                if valid is True and not safe_reason:
                    badge = '<span class="badge-valid">Valid</span>'
                elif valid is True and safe_reason:
                    badge = '<span class="badge-partial">Partial</span>'
                else:
                    badge = '<span class="badge-hallucinated">Hallucinated</span>'

                reason_html = ""
                if safe_reason:
                    reason_html = f"<span class='soft-text' style='font-size:.74rem; text-align:right;'>{safe_reason}</span>"

                st.markdown(
                    f"""
                    <div class="citation-row">
                        <div style="color:#dce7fb; line-height:1.6;">{safe_raw}</div>
                        <div style="display:flex; flex-direction:column; align-items:flex-end; gap:6px;">
                            {badge}
                            {reason_html}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.info("No citation details available.")
        section_close()

        if is_exp and state.get("subqueries"):
            section_open("Sub-queries Generated")
            for i, q in enumerate(state.get("subqueries", []), 1):
                st.markdown(f'<div class="subquery-pill"><b>{i}.</b> {q}</div>', unsafe_allow_html=True)
            section_close()

def render_comparison(base: dict, exp: dict):
    st.markdown("### Baseline vs Experimental Comparison")

    base_rate = base.get("hallucination_rate", 0)
    exp_rate = exp.get("hallucination_rate", 0)
    diff = base_rate - exp_rate

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card(f"{base_rate:.1%}", "Baseline Hall. Rate", "red" if base_rate > 0 else "green")
    with c2:
        metric_card(f"{exp_rate:.1%}", "Experimental Hall. Rate", "red" if exp_rate > 0 else "green")
    with c3:
        metric_card(f"{diff:.1%}", "Improvement", "green" if diff > 0 else "red" if diff < 0 else "blue")
    with c4:
        metric_card(len(exp.get("subqueries", [])), "Sub-queries Used", "teal")

    section_open("Detailed Comparison Table")
    df = pd.DataFrame(
        {
            "Metric": [
                "Papers Retrieved",
                "Sub-queries Used",
                "Review Length (chars)",
                "Total Citations",
                "Valid Citations",
                "Partial Citations",
                "Hallucinated Citations",
                "Hallucination Rate",
            ],
            "Baseline": [
                len(base.get("papers", [])),
                1,
                len(base.get("review_text", "")),
                base.get("total_citations", 0),
                base.get("valid_citations", 0),
                base.get("partial_citations", 0),
                base.get("hallucinated_citations", 0),
                f"{base_rate:.1%}",
            ],
            "Experimental": [
                len(exp.get("papers", [])),
                len(exp.get("subqueries", [])),
                len(exp.get("final_review") or exp.get("draft_review", "")),
                exp.get("total_citations", 0),
                exp.get("valid_citations", 0),
                exp.get("partial_citations", 0),
                exp.get("hallucinated_citations", 0),
                f"{exp_rate:.1%}",
            ],
        }
    )
    st.dataframe(df, use_container_width=True, hide_index=True)
    section_close()

    cc1, cc2 = st.columns(2)

    with cc1:
        section_open("Citation Status Comparison")
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                name="Baseline",
                x=["Valid", "Partial", "Hallucinated"],
                y=[
                    base.get("valid_citations", 0),
                    base.get("partial_citations", 0),
                    base.get("hallucinated_citations", 0),
                ],
                marker_color="#fb7185",
                opacity=0.85,
            )
        )
        fig.add_trace(
            go.Bar(
                name="Experimental",
                x=["Valid", "Partial", "Hallucinated"],
                y=[
                    exp.get("valid_citations", 0),
                    exp.get("partial_citations", 0),
                    exp.get("hallucinated_citations", 0),
                ],
                marker_color="#5ee7df",
                opacity=0.85,
            )
        )
        fig.update_layout(
            barmode="group",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#eaf2ff"),
            height=300,
            margin=dict(l=20, r=20, t=30, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig, use_container_width=True)
        section_close()

    with cc2:
        section_open("Hallucination Rate Gauge")
        fig2 = go.Figure(
            go.Indicator(
                mode="gauge+number+delta",
                value=exp_rate * 100,
                number={"suffix": "%", "valueformat": ".1f"},
                delta={"reference": base_rate * 100, "valueformat": ".1f"},
                title={"text": "Experimental Hall. Rate (%)"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#1a2b5e"},
                    "steps": [
                        {"range": [0, 20], "color": "#d1fae5"},
                        {"range": [20, 50], "color": "#fef3c7"},
                        {"range": [50, 100], "color": "#fee2e2"},
                    ],
                    "threshold": {
                        "line": {"color": "#c62828", "width": 3},
                        "thickness": 0.75,
                        "value": base_rate * 100,
                    },
                },
            )
        )
        fig2.update_layout(
            height=300,
            margin=dict(l=20, r=20, t=40, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#eaf2ff"),
        )
        st.plotly_chart(fig2, use_container_width=True)
        section_close()

    section_open("Statistical Interpretation")
    if diff > 0:
        st.success(
            f"The experimental system reduced the hallucination rate by {diff:.1%} "
            f"compared to the baseline ({base_rate:.1%} → {exp_rate:.1%}). This supports H1."
        )
    elif diff == 0:
        st.warning("Both systems achieved the same hallucination rate. Cannot reject H0 from this run alone.")
    else:
        st.error(
            f"The baseline had a lower hallucination rate by {abs(diff):.1%}. "
            f"This run supports H0."
        )
    section_close()

    st.download_button(
        label="Download Comparison CSV",
        data=df.to_csv(index=False),
        file_name=f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
    )


def render_results():
    st.markdown("---")
    st.markdown("## Results")

    exp = st.session_state.pipeline_state
    base = st.session_state.baseline_state

    labels = []
    if exp:
        labels.append("Experimental")
    if base:
        labels.append("Baseline")
    if exp and base:
        labels.append("Comparison")

    tabs = st.tabs(labels)
    idx = 0

    if exp:
        with tabs[idx]:
            render_system_results(exp, "experimental")
        idx += 1

    if base:
        with tabs[idx]:
            render_system_results(base, "baseline")
        idx += 1

    if exp and base:
        with tabs[idx]:
            render_comparison(base, exp)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def page_evaluation() -> None:
    """Evaluation page — shows eval results, MAB results, and ablation study."""
    # Hero header
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #1a2b5e 0%, #2d3748 100%);
                padding: 1.5rem 2rem; border-radius: 12px; margin-bottom: 1.5rem;
                box-shadow: 0 4px 12px rgba(0,0,0,0.3);">
        <h1 style="color:#eef4ff; margin:0 0 0.5rem 0; font-size:1.8rem;">
            📊 Evaluation Dashboard
        </h1>
        <p style="color:#94a3b8; margin:0; font-size:0.9rem; line-height:1.5;">
            Real-time results from the agentic pipeline evaluation.
            Compare <strong style="color:#5ee7df;">Experimental</strong> vs
            <strong style="color:#fb7185;">Baseline</strong> performance.
        </p>
    </div>
    """, unsafe_allow_html=True)

    ROOT     = Path(__file__).resolve().parent
    EVAL_DIR = ROOT / "data" / "eval"
    MAB_DIR  = ROOT / "data" / "eval" / "mab_results"
    ABL_DIR  = ROOT / "data" / "eval" / "ablation"

    for d in (EVAL_DIR, MAB_DIR, ABL_DIR):
        d.mkdir(parents=True, exist_ok=True)

    # ── Load real evaluation data ──────────────────
    import glob as _glob
    from collections import Counter as _Counter
    _ROOT = r"C:\Users\BEST LAPTOP\Desktop\FYP-LLM"

    def _load_json(rel):
        import json, os
        try:
            with open(os.path.join(_ROOT, rel), encoding="utf-8") as f:
                return json.load(f)
        except:
            return []

    def _load_csv(rel):
        import csv, os
        try:
            with open(os.path.join(_ROOT, rel), encoding="utf-8") as f:
                return list(csv.DictReader(f))
        except:
            return []

    _tax  = _load_json("evaluation_results/error_taxonomy_log.json")
    _wc   = _load_json("evaluation_results/wilson_ci_log.json")
    _cp   = _load_json("evaluation_results/correction_passes_log.json")
    _ab   = _load_csv("evaluation_results/ablation_raw.csv")
    _hist = _load_json("evaluation_results/topic_history.json")
    _vfiles = sorted(_glob.glob(_ROOT + "/data/eval/verifier_logs/*.json"))
    _runs = [_load_json(f.replace(_ROOT + "/", "")) for f in _vfiles]

    # Summary metrics with visual cards
    _tc = sum(r.get("total",0) for r in _runs)
    _tv = sum(r.get("valid",0) for r in _runs)
    _th = sum(r.get("hallucinated",0) for r in _runs)
    _tr = round(_th / _tc * 100, 1) if _tc else 0

    st.markdown("### 📊 Pipeline Performance Summary")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card(len(_runs), "Pipeline Runs", "blue")
    with c2:
        metric_card(_tc, "Citations Checked", "teal")
    with c3:
        metric_card(_th, "Hallucinated", "red" if _th > 0 else "green")
    with c4:
        metric_card(f"{_tr}%", "Overall Hall. Rate", "red" if _tr > 10 else "orange" if _tr > 0 else "green")

    st.divider()

    # Wilson CI
    if _wc:
        st.markdown("### 📈 Wilson Score Confidence Intervals")
        st.markdown(
            "<div style='color:#94a3b8; font-size:0.85rem; margin-bottom:1rem;'>"
            "95% Wilson score CIs on hallucination rates per pipeline run.</div>",
            unsafe_allow_html=True
        )
        real = [e for e in _wc if e.get("n_total",0) not in [25,20,10,30,100]]
        for e in (real[-5:] if real else _wc[-5:]):
            rate = e.get('hallucination_rate_pct',0)
            color = "green" if rate < 10 else "orange" if rate < 20 else "red"
            st.markdown(
                f"<div style='padding:10px; background:rgba(30,58,138,0.3); "
                f"border-left:4px solid {color}; border-radius:6px; margin-bottom:8px;'>"
                f"<span style='color:#eef4ff; font-weight:600;'>{e.get('timestamp','')[:16]}</span>  |  "
                f"<span style='color:{color}; font-weight:700;'>{e.get('n_hallucinated','?')}/{e.get('n_total','?')} hallucinated</span>  |  "
                f"Rate: <span style='color:{color}; font-weight:700;'>{rate}%</span>  |  "
                f"CI: [{e.get('wilson_ci_lower_pct','?')}%, {e.get('wilson_ci_upper_pct','?')}%]"
                f"{'<span style=\"color:#fbbf24;\">  | ⚠ ' + e.get('warning','') + '</span>' if e.get('warning') else ''}"
                f"</div>",
                unsafe_allow_html=True
            )

    # Error taxonomy
    if _tax:
        st.markdown("### 🏷 Error Taxonomy Breakdown")
        st.markdown(
            "<div style='color:#94a3b8; font-size:0.85rem; margin-bottom:1rem;'>"
            "Distribution of citation error types across all verifier runs.</div>",
            unsafe_allow_html=True
        )
        cnts = _Counter(e.get("error_type","?") for e in _tax)
        for etype, count in cnts.most_common():
            pct = round(count / len(_tax) * 100, 1)
            color = ("#22c55e" if etype == "VALID" else
                     "#ef4444" if etype == "FABRICATED_PAPER" else
                     "#f97316" if etype == "WRONG_AUTHOR" else
                     "#3b82f6")
            icon = ("✅" if etype == "VALID" else
                    "❌" if etype == "FABRICATED_PAPER" else
                    "⚠️" if etype == "WRONG_AUTHOR" else "ℹ️")
            st.markdown(
                f"<div style='display:flex; justify-content:space-between; "
                f"align-items:center; margin:8px 0; padding:8px 12px; "
                f"background:rgba(30,41,59,0.5); border-radius:8px;'>"
                f"<span>{icon} <strong style='color:#eef4ff;'>{etype}</strong></span>"
                f"<span style='color:{color}; font-weight:700;'>{count} ({pct}%)</span>"
                f"</div>",
                unsafe_allow_html=True
            )
            st.progress(count / len(_tax), color)

    # Correction passes
    if _cp:
        st.markdown("### 🔄 Self-Correction Passes")
        st.markdown(
            "<div style='color:#94a3b8; font-size:0.85rem; margin-bottom:1rem;'>"
            "Shows how hallucination rates decrease after each correction pass.</div>",
            unsafe_allow_html=True
        )
        for e in _cp[-5:]:
            bef = e.get("h_rate_before")
            aft = e.get("h_rate_after")
            bs = f"{float(bef)*100:.1f}%" if bef not in [None,"?"] else "N/A"
            af = f"{float(aft)*100:.1f}%" if aft not in [None,"?"] else "N/A"
            bef_val = float(bef) if bef not in [None,"?"] else 0
            color = "green" if bef_val < 0.1 else "orange" if bef_val < 0.2 else "red"
            st.markdown(
                f"<div style='padding:10px; margin:6px 0; background:rgba(30,41,59,0.6); "
                f"border-left:4px solid {color}; border-radius:6px;'>"
                f"<strong style='color:#eef4ff;'>{str(e.get('run_id','?'))[:14]}</strong> | "
                f"Pass {e.get('pass_number','?')} | "
                f"Before: <span style='color:{color}; font-weight:700;'>{bs}</span> → "
                f"After: <span style='color:green; font-weight:700;'>{af}</span> | "
                f"Removed: <span style='color:#fbbf24;'>{e.get('claims_removed',0)} claims</span>"
                f"</div>",
                unsafe_allow_html=True
            )

    # Ablation study
    if _ab:
        st.markdown("### 🧪 Ablation Study Results")
        st.markdown(
            "<div style='color:#94a3b8; font-size:0.85rem; margin-bottom:1rem;'>"
            "Live runs use actual UI topics. Dry-run shows placeholder data.</div>",
            unsafe_allow_html=True
        )
        real_ab = [r for r in _ab if r.get("latency_sec") not in ["45.2","12.1","14.8"]]
        show_ab = real_ab if real_ab else _ab
        st.caption(f"{'📊 Live data' if real_ab else '🎮 Dry-run'} — {len(show_ab)} rows")

        for row in show_ab:
            try:
                hall = float(row.get('hall_pct',0))
                color = "green" if hall == 0 else "orange" if hall < 15 else "red"
                icon = "✅" if hall == 0 else "⚠️" if hall < 15 else "❌"
            except:
                color = "grey"
                icon = "❓"
            st.markdown(
                f"<div style='padding:8px 12px; margin:4px 0; background:rgba(30,41,59,0.6); "
                f"border-left:4px solid {color}; border-radius:6px; display:flex; "
                f"justify-content:space-between; align-items:center;'>"
                f"<span>{icon} <code style='color:#eef4ff;'>{row.get('model','?')[:28]}</code></span>"
                f"<span style='color:#94a3b8;'>{row.get('topic_type','?')}</span>"
                f"<span style='color:{color}; font-weight:700;'>Hall: {row.get('hall_pct','-')}%</span>"
                f"<span style='color:#5ee7df;'>Valid: {row.get('valid','-')}</span>"
                f"<span style='color:#94a3b8;'>{row.get('latency_sec','-')}s</span>"
                f"</div>",
                unsafe_allow_html=True
            )

    # Topic history
    if _hist:
        st.markdown("### 📚 Your Topic History")
        st.markdown(
            "<div style='color:#94a3b8; font-size:0.85rem; margin-bottom:1rem;'>"
            f"{len(_hist)} topics run so far. Use these for dynamic ablation.</div>",
            unsafe_allow_html=True
        )
        for h in reversed(_hist[-8:]):
            hall = h.get('hall_pct', 0)
            try:
                color = "green" if float(hall) == 0 else "orange" if float(hall) < 15 else "red"
                icon = "✅" if float(hall) == 0 else "⚠️" if float(hall) < 15 else "❌"
            except:
                color = "grey"
                icon = "❓"
            ts = h.get('timestamp','')[:16]
            topic_str = h.get('topic', '?')[:55] if h.get('topic') else '?'
            st.markdown(
                f"<div style='padding:10px 14px; margin:6px 0; background:rgba(30,41,59,0.6); "
                f"border-left:4px solid {color}; border-radius:6px;'>"
                f"<div style='display:flex; justify-content:space-between; align-items:center;'>"
                f"<span style='color:#94a3b8; font-size:0.8rem;'>{ts}</span>"
                f"<span style='color:{color}; font-weight:700;'>{icon} Hall: {hall}%</span>"
                f"</div>"
                f"<div style='color:#eef4ff; margin-top:6px; font-size:0.9rem;'>"
                f"`{topic_str}`</div>"
                f"<div style='display:flex; gap:12px; margin-top:8px; font-size:0.8rem;'>"
                f"<span style='color:#5ee7df;'>✅ Valid: {h.get('valid','?')}</span>"
                f"<span style='color:#fb7185;'>📉 Model: {h.get('model','?')[:20]}</span>"
                f"</div></div>",
                unsafe_allow_html=True
            )
        st.markdown(
            "<div style='margin-top:1rem; padding:12px; background:rgba(34,197,94,0.1); "
            "border:1px solid rgba(34,197,94,0.3); border-radius:8px;'>"
            "💡 <strong style='color:#22c55e;'>Tip:</strong> "
            "<span style='color:#94a3b8;'>Run ablation on YOUR topics with:</span><br>"
            "<code style='background:rgba(0,0,0,0.3); padding:4px 8px; border-radius:4px; color:#eef4ff;'>"
            "python tools/run_ablation.py --dynamic</code></div>",
            unsafe_allow_html=True
        )

    if not any([_runs, _tax, _wc, _ab]):
        st.info("No evaluation data yet. Go to Run Experiment and run a topic first.")
    # ── End evaluation data block ──────────────────


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------
def page_history():
    render_header()
    st.markdown("### Run History")

    history = st.session_state.get("run_history", [])
    if not history:
        st.info("No runs recorded in this session yet. Go to Run Experiment to start.")
        return

    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card(len(history), "Total Runs", "blue")
    with c2:
        metric_card(sum(1 for r in history if r["system"] == "Experimental"), "Experimental Runs", "teal")
    with c3:
        metric_card(sum(1 for r in history if r["system"] == "Baseline"), "Baseline Runs", "orange")

    section_open("All Runs This Session")
    st.dataframe(pd.DataFrame(history), use_container_width=True, hide_index=True)
    section_close()

    st.markdown("### Run Details")
    for i, run in enumerate(reversed(history), 1):
        with st.expander(f"Run {len(history) - i + 1} • {run['system']} • {run['timestamp']}"):
            ca, cb, cc, cd = st.columns(4)
            with ca:
                metric_card(run["papers"], "Papers", "blue")
            with cb:
                metric_card(run["citations"], "Citations", "blue")
            with cc:
                metric_card(run["hallucinated"], "Hallucinated", "red")
            with cd:
                metric_card(
                    run["hallucination_rate"],
                    "Hall. Rate",
                    "green" if run["hallucination_rate"] == "0.0%" else "red"
                )
            st.markdown(f"**Topic:** {run['topic']}")

    if st.button("Clear History"):
        st.session_state.run_history = []
        st.rerun()


# ---------------------------------------------------------------------------
# About
# ---------------------------------------------------------------------------
def page_about():
    render_header()
    st.markdown("### About This Project")

    col1, col2 = st.columns([3, 2])

    with col1:
        section_open("Project Overview")
        st.markdown(
            """
            <div class="soft-text" style="line-height:1.8;">
                This system implements an agentic AI pipeline for autonomous academic literature review
                with integrated hallucination detection and mitigation.
                <br><br>
                The experimental system uses five cooperating AI agents to plan, search, summarise,
                verify, and assemble a high-quality literature review, while the verifier agent catches
                and removes fabricated or incorrect citations before the final output is produced.
                <br><br>
                This is compared against a single-LLM baseline that writes a review in one shot without verification.
            </div>""",
            unsafe_allow_html=True,
        )
        section_close()

        section_open("Evaluation Metrics")
        eval_items = [
            ("Citation Precision", "Valid citations / total generated citations"),
            ("Citation Recall", "Valid citations / total citations in gold standard"),
            ("Hallucination Rate", "Hallucinated / total citations"),
            ("Verifier Sensitivity", "Hallucinations correctly flagged"),
            ("Verifier Specificity", "Valid citations correctly accepted"),
            ("Cohen’s Kappa", "Inter/intra-annotator agreement"),
            ("Two-proportion z-test", "Statistical significance of hallucination difference"),
        ]
        for metric, desc in eval_items:
            st.markdown(
                f"""
                <div style="display:flex; gap:12px; padding:9px 0; border-bottom:1px solid rgba(138,180,248,.08);">
                    <span style="font-weight:700; color:#eef4ff; min-width:200px;">{metric}</span>
                    <span class="soft-text">{desc}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        section_close()

    with col2:
        section_open("Project Details")
        details = [
            ("Student", "Sami Ullah"),
            ("Degree", "MSc Data Science and Analytics"),
            ("University", "University of Hertfordshire"),
            ("Module", "Final Year Project"),
            ("Year", "2025–2026"),
        ]
        for label, value in details:
            st.markdown(
                f"""
                <div style="padding:8px 0; border-bottom:1px solid rgba(138,180,248,.08);">
                    <span style="font-size:.74rem; color:#7f90af; text-transform:uppercase; letter-spacing:.1em;">{label}</span><br>
                    <span style="font-size:.94rem; font-weight:700; color:#eef4ff;">{value}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        section_close()

        section_open("Project File Structure")
        files = [
            ("src", "Core utilities"),
            ("agents", "5 AI agents"),
            ("graph", "LangGraph workflows"),
            ("configs", "Frozen prompts"),
            ("data/corpus", "Gold standard"),
            ("evaluation", "Metrics and annotation"),
            ("main.py", "CLI entry point"),
            ("app.py", "Streamlit frontend"),
        ]
        for fname, desc in files:
            st.markdown(
                f"""
                <div style="display:flex; justify-content:space-between; gap:12px; padding:7px 0; border-bottom:1px solid rgba(138,180,248,.08);">
                    <code style="color:#9dd9ff;">{fname}</code>
                    <span class="soft-text">{desc}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        section_close()

        section_open("How to Run")
        st.code(
            "pip install -r requirements.txt\nstreamlit run app.py\nuv run main.py\nuv run test_evaluation.py",
            language="bash",
        )
        section_close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    page, model, max_papers = render_sidebar()

    if page != st.session_state.get("page"):
        st.session_state.page = page

    current_page = st.session_state.page

    if "Home" in current_page:
        page_home()
    elif "Run Experiment" in current_page:
        page_run_experiment(model, max_papers)
    elif "Evaluation" in current_page:
        page_evaluation()
    elif "History" in current_page:
        page_history()
    elif "About" in current_page:
        page_about()


if __name__ == "__main__":
    main()