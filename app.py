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

                if valid is True and not reason:
                    badge = '<span class="badge-valid">Valid</span>'
                elif valid is True and reason:
                    badge = '<span class="badge-partial">Partial</span>'
                else:
                    badge = '<span class="badge-hallucinated">Hallucinated</span>'

                reason_html = ""
                if reason:
                    reason_html = f"<span class='soft-text' style='font-size:.74rem; text-align:right;'>{reason}</span>"

                st.markdown(
                    f"""
                    <div class="citation-row">
                        <div style="color:#dce7fb; line-height:1.6;">{raw}</div>
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
    st.markdown("## Evaluation Results")
    st.markdown("Compare the **Experimental** multi-agent pipeline against the **Baseline** single-LLM pipeline.")

    ROOT     = Path(__file__).resolve().parent
    EVAL_DIR = ROOT / "data" / "eval"
    MAB_DIR  = ROOT / "data" / "eval" / "mab_results"
    ABL_DIR  = ROOT / "data" / "eval" / "ablation"

    for d in (EVAL_DIR, MAB_DIR, ABL_DIR):
        d.mkdir(parents=True, exist_ok=True)

    # ── How to generate results ───────────────────────────────────────────
    with st.expander("How to generate evaluation data", expanded=False):
        st.markdown("""
        **Option A — Instant sample data (no API calls):**
        
        Run the evaluation script to generate sample results without API calls.
        """)


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