# app.py
# Agentic AI for Reliable Academic Literature Review
# MSc Data Science — University of Hertfordshire
# Run: streamlit run app.py

import sys
import time
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

st.set_page_config(
    page_title="LitReview Agent",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inline CSS styles
st.markdown("""
<style>
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background-color: #f4f6fb; }
[data-testid="stSidebar"] { background: linear-gradient(180deg, #1a2b5e 0%, #0d1b3e 100%); }
[data-testid="stSidebar"] * { color: white !important; }
.header-banner { background: linear-gradient(135deg, #1a2b5e 0%, #0d47a1 50%, #1565c0 100%); padding: 28px 32px; border-radius: 16px; margin-bottom: 24px; display: flex; align-items: center; gap: 20px; box-shadow: 0 4px 20px rgba(26,43,94,0.3); }
.header-title { color: white; font-size: 1.8rem; font-weight: 700; margin: 0; line-height: 1.2; }
.header-subtitle { color: #a8d4f5; font-size: 0.9rem; margin: 6px 0 0 0; }
.metric-card { background: white; border-radius: 12px; padding: 20px 24px; box-shadow: 0 2px 12px rgba(0,0,0,0.08); border-left: 5px solid #1a2b5e; margin-bottom: 16px; text-align: center; }
.metric-card-green { border-left-color: #2e7d32; }
.metric-card-red { border-left-color: #c62828; }
.metric-card-orange { border-left-color: #ef6c00; }
.metric-card-blue { border-left-color: #1565c0; }
.metric-card-teal { border-left-color: #00695c; }
.metric-value { font-size: 2.2rem; font-weight: 700; color: #1a2b5e; line-height: 1; }
.metric-label { font-size: 0.8rem; color: #6b7280; font-weight: 500; text-transform: uppercase; letter-spacing: 0.05em; margin-top: 6px; }
.section-card { background: white; border-radius: 12px; padding: 24px; box-shadow: 0 2px 12px rgba(0,0,0,0.06); margin-bottom: 20px; }
.section-title { font-size: 1.05rem; font-weight: 600; color: #1a2b5e; margin-bottom: 14px; padding-bottom: 8px; border-bottom: 2px solid #e5e7eb; }
.agent-step { display: flex; align-items: center; gap: 12px; padding: 12px 16px; border-radius: 8px; margin-bottom: 8px; font-size: 0.9rem; font-weight: 500; }
.step-pending { background:#f3f4f6; color:#9ca3af; }
.step-running { background:#dbeafe; color:#1d4ed8; border:1px solid #93c5fd; }
.step-done { background:#d1fae5; color:#065f46; }
.step-error { background:#fee2e2; color:#991b1b; }
.badge-valid { background:#d1fae5; color:#065f46; padding:3px 10px; border-radius:12px; font-size:0.78rem; font-weight:600; }
.badge-partial { background:#fef3c7; color:#92400e; padding:3px 10px; border-radius:12px; font-size:0.78rem; font-weight:600; }
.badge-hallucinated { background:#fee2e2; color:#991b1b; padding:3px 10px; border-radius:12px; font-size:0.78rem; font-weight:600; }
.review-box { background: #fafafa; border: 1px solid #e5e7eb; border-radius: 10px; padding: 20px; font-size: 0.92rem; line-height: 1.8; color: #374151; max-height: 420px; overflow-y: auto; white-space: pre-wrap; }
.citation-row { display: flex; align-items: center; justify-content: space-between; padding: 10px 14px; border-radius: 8px; background: #f9fafb; border: 1px solid #e5e7eb; margin-bottom: 8px; font-size: 0.88rem; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
def init_session():
    defaults = {
        "page":           "🏠 Home",
        "pipeline_state": None,
        "baseline_state": None,
        "run_history":    [],
        "topic":          "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def render_sidebar():
    with st.sidebar:
        st.markdown(
            '<div style="text-align:center;padding:16px 0 20px 0;">'
            '<div style="font-size:2.6rem;">📚</div>'
            '<div style="font-size:1.05rem;font-weight:700;color:white;'
            'line-height:1.3;margin-top:8px;">LitReview Agent</div>'
            '<div style="font-size:0.72rem;color:#a8c4e0;margin-top:4px;">'
            'Agentic AI · Hallucination Detection</div></div>',
            unsafe_allow_html=True,
        )
        st.divider()
        st.markdown("**Navigation**")
        page = st.radio(
            "nav",
            ["🏠 Home", "🔬 Run Experiment", "📊 Evaluation", "📋 History", "ℹ️ About"],
            label_visibility="collapsed",
        )
        st.divider()
        st.markdown("**Model Settings**")
        model = st.selectbox(
            "LLM Model",
            ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"],
        )
        max_papers = st.slider("Papers per query", 3, 10, 5)
        st.divider()
        st.markdown(
            f'<div style="font-size:0.72rem;color:#a8c4e0;text-align:center;">'
            f'MSc Data Science<br/>University of Hertfordshire<br/>'
            f'v1.0 · {datetime.now().strftime("%B %Y")}</div>',
            unsafe_allow_html=True,
        )
    return page, model, max_papers


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def mc(value, label, colour="blue"):
    st.markdown(
        f'<div class="metric-card metric-card-{colour}">'
        f'<div class="metric-value">{value}</div>'
        f'<div class="metric-label">{label}</div></div>',
        unsafe_allow_html=True,
    )

def header():
    st.markdown(
        '<div class="header-banner">'
        '<div style="font-size:3rem;flex-shrink:0;">📚</div>'
        '<div><p class="header-title">Agentic AI for Reliable Academic Literature Review</p>'
        '<p class="header-subtitle">Multi-agent pipeline · Citation verification · '
        'Hallucination detection &nbsp;·&nbsp; MSc Data Science &nbsp;·&nbsp; '
        'University of Hertfordshire</p></div></div>',
        unsafe_allow_html=True,
    )

def sc_open(title):
    st.markdown(
        f'<div class="section-card"><div class="section-title">{title}</div>',
        unsafe_allow_html=True,
    )

def sc_close():
    st.markdown('</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# PAGE: Home
# ---------------------------------------------------------------------------
def page_home():
    header()
    st.markdown("### Pipeline Architecture")
    cols = st.columns(5)
    for col, (icon, name, desc) in zip(cols, [
        ("🧠","Planner","Decomposes topic into 3–5 sub-queries"),
        ("🔍","Searcher","Fetches papers from OpenAlex API"),
        ("✍️","Summariser","Writes 300–400 word review"),
        ("✅","Verifier","Checks citations against metadata"),
        ("📝","Assembler","Removes hallucinated refs"),
    ]):
        with col:
            st.markdown(
                f'<div class="section-card" style="text-align:center;min-height:150px;">'
                f'<div style="font-size:1.9rem;">{icon}</div>'
                f'<div style="font-weight:700;color:#1a2b5e;font-size:0.98rem;'
                f'margin:8px 0 4px;">{name}</div>'
                f'<div style="font-size:0.78rem;color:#6b7280;">{desc}</div></div>',
                unsafe_allow_html=True,
            )

    st.markdown(
        "<p style='text-align:center;color:#9ca3af;margin:-8px 0 16px;'>"
        "Planner → Searcher → Summariser → Verifier → Assembler</p>",
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            '<div class="section-card">'
            '<div class="section-title">Research Question</div>'
            '<p style="font-size:0.92rem;font-style:italic;color:#374151;'
            'border-left:4px solid #1565c0;padding-left:14px;margin:0;">'
            '"How effectively can an agentic AI system perform autonomous academic '
            'literature review while using self-correcting mechanisms to detect and '
            'reduce LLM hallucinations in generated summaries and citations?"'
            '</p></div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="section-card">'
            '<div class="section-title">Hypotheses</div>'
            '<div style="display:flex;align-items:flex-start;gap:10px;margin-bottom:12px;">'
            '<span style="background:#fee2e2;color:#991b1b;padding:2px 8px;'
            'border-radius:6px;font-size:0.78rem;font-weight:700;flex-shrink:0;">H₀</span>'
            '<span style="color:#6b7280;font-size:0.88rem;">Self-correction adds no '
            'significant improvement over a single LLM baseline</span></div>'
            '<div style="display:flex;align-items:flex-start;gap:10px;">'
            '<span style="background:#d1fae5;color:#065f46;padding:2px 8px;'
            'border-radius:6px;font-size:0.78rem;font-weight:700;flex-shrink:0;">H₁</span>'
            '<span style="color:#6b7280;font-size:0.88rem;">Multi-agent verifier '
            'significantly reduces hallucination rate vs single-LLM baseline'
            '</span></div></div>',
            unsafe_allow_html=True,
        )

    with col2:
        sc_open("Technology Stack")
        for icon, tech, desc in [
            ("🐍","Python 3.13","Core language"),
            ("🔗","LangGraph","Agent orchestration"),
            ("🤖","Groq LLM","Fast inference"),
            ("📖","OpenAlex API","Academic metadata"),
            ("🗄️","FAISS+SBERT","Semantic retrieval"),
            ("📊","Streamlit","Frontend"),
        ]:
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:12px;padding:8px 0;'
                f'border-bottom:1px solid #f3f4f6;">'
                f'<span style="font-size:1.2rem;width:24px;">{icon}</span>'
                f'<span style="font-weight:600;color:#1a2b5e;font-size:0.88rem;'
                f'width:150px;">{tech}</span>'
                f'<span style="color:#6b7280;font-size:0.83rem;">{desc}</span></div>',
                unsafe_allow_html=True,
            )
        sc_close()

    _, btn_col, _ = st.columns([3, 2, 3])
    with btn_col:
        if st.button("🚀 Start Experiment", use_container_width=True):
            st.session_state["page"] = "🔬 Run Experiment"
            st.rerun()


# ---------------------------------------------------------------------------
# PAGE: Run Experiment
# ---------------------------------------------------------------------------
def page_run_experiment(model, max_papers):
    header()
    st.markdown("### Run Literature Review Pipeline")

    sc_open("Research Topic")
    topic = st.text_area(
        "topic",
        value=st.session_state.get("topic", ""),
        height=80,
        placeholder=(
            "e.g. Agentic AI for reliable academic literature review "
            "and hallucination mitigation"
        ),
        label_visibility="collapsed",
        key="topic_input",          # ADD THIS KEY
    )
    st.session_state["topic"] = topic
    sc_close()
    if topic.strip():
        st.session_state["topic"] = topic.strip()

    # Add file upload option
    st.markdown("**Or upload a research document:**")

    uploaded_file = st.file_uploader(
        "Upload PDF, DOCX, or TXT file",
        type=["pdf", "docx", "txt"],
        help=(
            "Upload a research brief, paper, or document. "
            "The system will extract the topic automatically."
        ),
    )

    if uploaded_file is not None:
        import tempfile
        import os

        suffix = Path(uploaded_file.name).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        try:
            from src.document_reader import load_input, extract_topic_from_document

            with st.spinner(f"Reading {uploaded_file.name}..."):
                raw_text = load_input(tmp_path, mode="auto")
                if len(raw_text) > 500:
                    with st.spinner("Extracting research topic from document..."):
                        extracted_topic = extract_topic_from_document(raw_text)
                    st.success(f"Topic extracted: **{extracted_topic}**")
                    st.session_state["topic"] = extracted_topic
                else:
                    st.session_state["topic"] = raw_text
                    st.success("Document loaded as topic.")
        except Exception as e:
            st.error(f"Could not read file: {e}")
        finally:
            os.unlink(tmp_path)


    st.markdown("**Or choose a preset:**")
    presets = [
        "Hallucination detection and mitigation in large language models",
        "Retrieval augmented generation for reducing LLM hallucinations",
        "Multi-agent systems for automated academic literature review",
    ]
    preset_cols = st.columns(3)   # ← ADD THIS LINE
    for i, (col, preset) in enumerate(zip(preset_cols, presets)):
        with col:
            if st.button(f"📌 {preset[:45]}...", key=f"preset_{i}", use_container_width=True):
                st.session_state["topic"] = preset
                st.rerun()

    st.markdown("<br/>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        run_exp  = st.button("🔬 Run Experimental\n(Multi-Agent Pipeline)", use_container_width=True)
    with c2:
        run_base = st.button("📄 Run Baseline\n(Single LLM)", use_container_width=True)
    with c3:
        run_both = st.button("⚖️ Run Both\n(Full Comparison)", use_container_width=True)

    # Get topic from text area OR session state
    # (Streamlit sometimes loses text area value on button click)
    active_topic = topic.strip() if topic.strip() else st.session_state.get("topic", "").strip()

    if (run_exp or run_base or run_both) and not active_topic:
        st.error("Please enter a research topic before running.")
        return

    if run_exp or run_both:
        st.session_state["topic"] = active_topic
        run_experimental(active_topic)

    if run_base or run_both:
        st.session_state["topic"] = active_topic
        run_baseline(active_topic)

    if st.session_state["pipeline_state"] or st.session_state["baseline_state"]:
        show_results()


def run_experimental(topic):
    st.markdown("---")
    st.markdown("### 🔬 Experimental Pipeline Running...")
    steps = [
        ("🧠", "Planner Agent",    "Decomposing topic into sub-queries..."),
        ("🔍", "Search Agent",     "Fetching papers from OpenAlex..."),
        ("✍️",  "Summariser Agent", "Writing literature review..."),
        ("✅", "Verifier Agent",   "Verifying citations..."),
        ("📝", "Assembler Agent",  "Assembling final review..."),
    ]
    ph = st.empty()
    pb = st.progress(0)
    st_ = st.empty()

    def show(statuses):
        with ph.container():
            for i, (icon, name, desc) in enumerate(steps):
                css = (
                    "step-done"    if statuses[i] == "done"    else
                    "step-running" if statuses[i] == "running" else
                    "step-error"   if statuses[i] == "error"   else
                    "step-pending"
                )
                ind = (
                    "✅" if statuses[i] == "done"    else
                    "⏳" if statuses[i] == "running" else
                    "❌" if statuses[i] == "error"   else
                    "⬜"
                )
                st.markdown(
                    f'<div class="agent-step {css}">{ind} &nbsp;'
                    f'<b>{name}</b> &nbsp;'
                    f'<span style="font-weight:400;opacity:0.8;">— {desc}</span></div>',
                    unsafe_allow_html=True,
                )

    statuses = ["pending"] * 5
    try:
        from graph.workflow_graph import run_workflow
        statuses[0] = "running"
        show(statuses)
        pb.progress(10)
        st_.info("🧠 Planner: Generating sub-queries...")
        fs = run_workflow(topic)
        for i in range(5):
            statuses[i] = "done"
            show(statuses)
            pb.progress(20 + i * 16)
            time.sleep(0.2)
        pb.progress(100)
        st_.success("✅ Experimental pipeline complete!")
        st.session_state["pipeline_state"] = fs
        st.session_state["run_history"].append({
            "timestamp":          datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "topic":              topic[:60],
            "system":             "Experimental",
            "papers":             len(fs.get("papers", [])),
            "citations":          fs.get("total_citations", 0),
            "hallucinated":       fs.get("hallucinated_citations", 0),
            "hallucination_rate": f"{fs.get('hallucination_rate', 0):.1%}",
        })
    except Exception as e:
        if "running" in statuses:
            statuses[statuses.index("running")] = "error"
        show(statuses)
        st_.error(f"Error: {e}")
        st.exception(e)


def run_baseline(topic):
    st.markdown("---")
    st.markdown("### 📄 Baseline Pipeline Running...")
    pb = st.progress(0)
    st_ = st.empty()
    try:
        from graph.baseline_graph import run_baseline as rb
        st_.info("🔍 Searching papers...")
        pb.progress(30)
        bs = rb(topic)
        pb.progress(100)
        st_.success("✅ Baseline complete!")
        st.session_state["baseline_state"] = bs
        st.session_state["run_history"].append({
            "timestamp":          datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "topic":              topic[:60],
            "system":             "Baseline",
            "papers":             len(bs.get("papers", [])),
            "citations":          bs.get("total_citations", 0),
            "hallucinated":       bs.get("hallucinated_citations", 0),
            "hallucination_rate": f"{bs.get('hallucination_rate', 0):.1%}",
        })
    except Exception as e:
        st_.error(f"Error: {e}")
        st.exception(e)


def show_results():
    st.markdown("---")
    st.markdown("### Results")
    exp  = st.session_state["pipeline_state"]
    base = st.session_state["baseline_state"]
    labels = []
    if exp:  labels.append("🔬 Experimental")
    if base: labels.append("📄 Baseline")
    if exp and base: labels.append("⚖️ Comparison")
    tabs = st.tabs(labels)
    idx  = 0
    if exp:
        with tabs[idx]:
            show_system(exp, "experimental")
        idx += 1
    if base:
        with tabs[idx]:
            show_system(base, "baseline")
        idx += 1
    if exp and base:
        with tabs[idx]:
            show_comparison(base, exp)


def show_system(state, system):
    is_exp  = system == "experimental"
    review  = (
        state.get("final_review", "") or state.get("draft_review", "")
        if is_exp else
        state.get("review_text", "")
    )
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: mc(len(state.get("papers", [])), "Papers Retrieved", "blue")
    with c2: mc(state.get("total_citations", 0), "Total Citations", "blue")
    with c3: mc(state.get("valid_citations", 0), "Valid", "green")
    with c4: mc(state.get("hallucinated_citations", 0), "Hallucinated", "red")
    with c5:
        r = state.get("hallucination_rate", 0)
        mc(f"{r:.1%}", "Hallucination Rate", "green" if r == 0 else "orange" if r < 0.2 else "red")

    if is_exp:
        ca, cb = st.columns(2)
        with ca: mc(len(state.get("sub_queries", [])), "Sub-queries Used", "teal")
        with cb: mc(state.get("changes", {}).get("words_removed", 0), "Words Removed", "orange")

    st.markdown("<br/>", unsafe_allow_html=True)
    cr, cc = st.columns([3, 2])

    with cr:
        sc_open("Generated Review")
        st.markdown(f'<div class="review-box">{review}</div>', unsafe_allow_html=True)
        sc_close()
        st.download_button(
            "⬇️ Download Review", review,
            f"review_{system}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            "text/plain",
        )

    with cc:
        sc_open("Citation Verification")
        citations = state.get("citation_details", [])
        # REPLACE the citations loop in show_system() with this:
        if citations:
            for cit in citations:
                raw    = getattr(cit, "raw_reference", str(cit))
                valid  = getattr(cit, "valid",         None)
                reason = getattr(cit, "error_reason",  None)

                # Status badge
                if valid is True and not reason:
                    badge  = '<span class="badge-valid">✓ VALID</span>'
                elif valid is True and reason:
                    badge  = f'<span class="badge-partial">⚠ PARTIAL</span>'
                else:
                    error_label = reason or "No match found"
                    badge  = f'<span class="badge-hallucinated">✗ HALLUCINATED</span>'

                # Error type pill
                error_pill = ""
                if reason:
                    pill_colour = "#fef3c7" if valid else "#fee2e2"
                    text_colour = "#92400e" if valid else "#991b1b"
                    error_pill = (
                        f'<span style="font-size:0.7rem;background:{pill_colour};'
                        f'color:{text_colour};padding:2px 6px;border-radius:8px;'
                        f'margin-left:6px;">{reason}</span>'
                    )

                st.markdown(
                    f'<div class="citation-row">'
                    f'<span style="font-size:0.83rem;color:#374151;'
                    f'max-width:55%;overflow:hidden;">{raw}</span>'
                    f'<span>{badge}{error_pill}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.info("No citation details available.")
        sc_close()

        if is_exp and state.get("sub_queries"):
            sc_open("Sub-queries Generated")
            for i, q in enumerate(state["sub_queries"], 1):
                st.markdown(
                    f'<div style="padding:6px 10px;margin-bottom:6px;background:#f0f4ff;'
                    f'border-radius:6px;font-size:0.83rem;color:#1a2b5e;">'
                    f'<b>{i}.</b> {q}</div>',
                    unsafe_allow_html=True,
                )
            sc_close()
    # Add latency and cost display for experimental system
    if is_exp:
        latency = state.get("latency_seconds", 0)
        tokens  = state.get("token_estimate",  0)

        # Cost estimate (Groq llama-3.3-70b pricing)
        cost_usd = (tokens / 1_000_000) * 0.79

        ca, cb, cc = st.columns(3)
        with ca:
            mc(
                f"{latency}s",
                "Pipeline Latency",
                "teal",
            )
        with cb:
            mc(
                f"~{tokens:,}",
                "Token Estimate",
                "blue",
            )
        with cc:
            mc(
                f"${cost_usd:.4f}",
                "Est. Cost (USD)",
                "orange",
            )


def show_comparison(base, exp):
    st.markdown("### Baseline vs Experimental Comparison")
    br   = base.get("hallucination_rate", 0)
    er   = exp.get("hallucination_rate", 0)
    diff = br - er
    c1, c2, c3, c4 = st.columns(4)
    with c1: mc(f"{br:.1%}", "Baseline Hall. Rate",      "red"  if br > 0 else "green")
    with c2: mc(f"{er:.1%}", "Experimental Hall. Rate",  "red"  if er > 0 else "green")
    with c3: mc(f"{diff:+.1%}", "Improvement",           "green" if diff > 0 else "red" if diff < 0 else "blue")
    with c4: mc(len(exp.get("sub_queries", [])), "Sub-queries Used", "teal")

    st.markdown("<br/>", unsafe_allow_html=True)
    sc_open("Comparison Table")
    df = pd.DataFrame({
        "Metric": ["Papers Retrieved","Sub-queries","Review Length (chars)","Total Citations","Valid","Partial","Hallucinated","Hallucination Rate"],
        "Baseline": [
            len(base.get("papers", [])), 1,
            len(base.get("review_text", "")),
            base.get("total_citations", 0), base.get("valid_citations", 0),
            base.get("partial_citations", 0), base.get("hallucinated_citations", 0),
            f"{br:.1%}",
        ],
        "Experimental": [
            len(exp.get("papers", [])), len(exp.get("sub_queries", [])),
            len(exp.get("final_review", "") or exp.get("draft_review", "")),
            exp.get("total_citations", 0), exp.get("valid_citations", 0),
            exp.get("partial_citations", 0), exp.get("hallucinated_citations", 0),
            f"{er:.1%}",
        ],
    })
    st.dataframe(df, use_container_width=True, hide_index=True)
    sc_close()

    ca, cb = st.columns(2)
    with ca:
        fig = go.Figure()
        fig.add_trace(go.Bar(name="Baseline",      x=["Valid","Partial","Hallucinated"], y=[base.get("valid_citations",0),base.get("partial_citations",0),base.get("hallucinated_citations",0)], marker_color="#c62828"))
        fig.add_trace(go.Bar(name="Experimental",  x=["Valid","Partial","Hallucinated"], y=[exp.get("valid_citations",0), exp.get("partial_citations",0), exp.get("hallucinated_citations",0)],  marker_color="#1a2b5e"))
        fig.update_layout(barmode="group", plot_bgcolor="white", paper_bgcolor="white", height=280, margin=dict(l=20,r=20,t=40,b=20), legend=dict(orientation="h",yanchor="bottom",y=1.02))
        st.plotly_chart(fig, use_container_width=True)
    with cb:
        fig2 = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=er*100,
            delta={"reference": br*100, "valueformat": ".1f"},
            title={"text": "Experimental Hall. Rate (%)"},
            gauge={"axis":{"range":[0,100]},"bar":{"color":"#1a2b5e"},"steps":[{"range":[0,20],"color":"#d1fae5"},{"range":[20,50],"color":"#fef3c7"},{"range":[50,100],"color":"#fee2e2"}],"threshold":{"line":{"color":"#c62828","width":3},"thickness":0.75,"value":br*100}},
            number={"suffix":"%","valueformat":".1f"},
        ))
        fig2.update_layout(height=280, margin=dict(l=20,r=20,t=40,b=20), paper_bgcolor="white")
        st.plotly_chart(fig2, use_container_width=True)

    sc_open("Statistical Interpretation")
    if diff > 0:
                st.success(f"✅ Experimental reduced hallucination rate by **{diff:.1%}** ({br:.1%} → {er:.1%}). Supports H₁.")
    elif diff == 0:
        st.warning("⚠️ Both systems achieved the same hallucination rate. Cannot reject H₀.")
    else:
        st.error(f"❌ Baseline had lower hallucination rate by {abs(diff):.1%}. Supports H₀.")
    sc_close()

    st.download_button(
        "⬇️ Download Comparison CSV",
        df.to_csv(index=False),
        f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        "text/csv",
    )


# ---------------------------------------------------------------------------
# PAGE: Evaluation
# ---------------------------------------------------------------------------
def page_evaluation():
    header()
    st.markdown("### Evaluation Dashboard")

    eval_dir  = ROOT / "data" / "eval"
    csv_files = list(eval_dir.glob("results_*.csv")) if eval_dir.exists() else []

    if not csv_files:
        st.info("No evaluation results found. Run: uv run test_evaluation.py")
        return

    selected = st.selectbox(
        "Select evaluation run:",
        [f.name for f in sorted(csv_files, reverse=True)],
    )
    df  = pd.read_csv(eval_dir / selected)
    bdf = df[df["system"] == "baseline"]
    edf = df[df["system"] == "experimental"]

    st.markdown(f"Loaded `{selected}` — {len(df)} rows")
    st.markdown("### Summary")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        mc(len(df["topic"].unique()), "Topics Evaluated", "blue")
    with c2:
        ab = bdf["hallucination_rate"].mean() if len(bdf) > 0 else 0
        mc(f"{ab:.1%}", "Avg Baseline Hall. Rate", "red")
    with c3:
        ae = edf["hallucination_rate"].mean() if len(edf) > 0 else 0
        mc(f"{ae:.1%}", "Avg Experimental Hall. Rate", "green" if ae <= ab else "red")
    with c4:
        imp = ab - ae
        mc(f"{imp:+.1%}", "Overall Improvement", "green" if imp > 0 else "red")

    st.markdown("<br/>", unsafe_allow_html=True)

    sc_open("Full Results Table")
    st.dataframe(df, use_container_width=True, hide_index=True)
    sc_close()

    cc1, cc2 = st.columns(2)
    with cc1:
        sc_open("Hallucination Rate per Topic")
        fig = px.bar(
            df, x="topic", y="hallucination_rate", color="system",
            barmode="group",
            color_discrete_map={"baseline": "#c62828", "experimental": "#1a2b5e"},
        )
        fig.update_layout(
            plot_bgcolor="white", paper_bgcolor="white", height=300,
            margin=dict(l=20, r=20, t=20, b=20), xaxis_tickangle=-30,
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig, use_container_width=True)
        sc_close()

    with cc2:
        sc_open("Papers Retrieved per Run")
        fig2 = px.bar(
            df, x="topic", y="papers_retrieved", color="system",
            barmode="group",
            color_discrete_map={"baseline": "#ef6c00", "experimental": "#00695c"},
        )
        fig2.update_layout(
            plot_bgcolor="white", paper_bgcolor="white", height=300,
            margin=dict(l=20, r=20, t=20, b=20), xaxis_tickangle=-30,
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig2, use_container_width=True)
        sc_close()

    st.download_button(
        "⬇️ Download Results CSV",
        df.to_csv(index=False),
        selected,
        "text/csv",
    )


# ---------------------------------------------------------------------------
# PAGE: History
# ---------------------------------------------------------------------------
def page_history():
    header()
    st.markdown("### Run History")

    history = st.session_state.get("run_history", [])

    if not history:
        st.info("No runs yet. Go to Run Experiment to start.")
        return

    c1, c2, c3 = st.columns(3)
    with c1: mc(len(history), "Total Runs", "blue")
    with c2: mc(sum(1 for r in history if r["system"] == "Experimental"), "Experimental Runs", "teal")
    with c3: mc(sum(1 for r in history if r["system"] == "Baseline"), "Baseline Runs", "orange")

    st.markdown("<br/>", unsafe_allow_html=True)

    sc_open("All Runs This Session")
    st.dataframe(pd.DataFrame(history), use_container_width=True, hide_index=True)
    sc_close()

    st.markdown("### Run Details")
    for i, run in enumerate(reversed(history), 1):
        with st.expander(
            f"Run {len(history)-i+1} — {run['system']} — {run['timestamp']}"
        ):
            ca, cb, cc, cd = st.columns(4)
            with ca: mc(run["papers"],      "Papers",      "blue")
            with cb: mc(run["citations"],   "Citations",   "blue")
            with cc: mc(run["hallucinated"],"Hallucinated","red")
            with cd: mc(
                run["hallucination_rate"], "Hall. Rate",
                "green" if run["hallucination_rate"] == "0.0%" else "red",
            )
            st.markdown(f"**Topic:** {run['topic']}")

    if st.button("🗑️ Clear History"):
        st.session_state["run_history"] = []
        st.rerun()


# ---------------------------------------------------------------------------
# PAGE: About
# ---------------------------------------------------------------------------
def page_about():
    header()
    st.markdown("### About This Project")

    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown(
            '<div class="section-card">'
            '<div class="section-title">Project Overview</div>'
            '<p style="color:#374151;font-size:0.92rem;line-height:1.75;">'
            'This system implements an agentic AI pipeline for autonomous academic '
            'literature review with integrated hallucination detection and mitigation.</p>'
            '<p style="color:#374151;font-size:0.92rem;line-height:1.75;">'
            'The experimental system uses five cooperating AI agents to plan, search, '
            'summarise, verify, and assemble a high-quality literature review — while '
            'the verifier agent catches and removes fabricated or incorrect citations '
            'before the final output is produced.</p>'
            '<p style="color:#374151;font-size:0.92rem;line-height:1.75;">'
            'This is compared against a single-LLM baseline that writes a review '
            'in one shot without verification.</p></div>',
            unsafe_allow_html=True,
        )

        sc_open("Evaluation Metrics")
        for metric, desc in [
            ("Citation Precision",    "Valid citations / total generated citations"),
            ("Citation Recall",       "Valid citations / total in gold standard"),
            ("Hallucination Rate",    "Hallucinated / total citations"),
            ("Verifier Sensitivity",  "Hallucinations correctly flagged"),
            ("Verifier Specificity",  "Valid citations correctly accepted"),
            ("Cohen's Kappa",         "Inter/intra-annotator agreement"),
            ("Two-proportion z-test", "Statistical significance test"),
        ]:
            st.markdown(
                f'<div style="display:flex;gap:12px;padding:8px 0;'
                f'border-bottom:1px solid #f3f4f6;font-size:0.88rem;">'
                f'<span style="font-weight:600;color:#1a2b5e;min-width:200px;">{metric}</span>'
                f'<span style="color:#6b7280;">{desc}</span></div>',
                unsafe_allow_html=True,
            )
        sc_close()

    with col2:
        sc_open("Project Details")
        for label, value in [
            ("Student", "Sami Ullah"),
            ("Degree", "MSc Data Science and Analytics"),
            ("University", "University of Hertfordshire"),
            ("Module", "Final Year Project"),
            ("Year", "2025–2026"),
        ]:
            st.markdown(
                f'<div style="padding:8px 0;border-bottom:1px solid #f3f4f6;">'
                f'<span style="font-size:0.75rem;color:#9ca3af;text-transform:uppercase;">'
                f'{label}</span><br/>'
                f'<span style="font-size:0.92rem;font-weight:600;color:#1a2b5e;">'
                f'{value}</span></div>',
                unsafe_allow_html=True,
            )
        sc_close()

        sc_open("File Structure")
        for fname, desc in [
            ("src/",          "Core utilities"),
            ("agents/",       "5 AI agents"),
            ("graph/",        "LangGraph workflows"),
            ("configs/",      "Frozen prompts"),
            ("data/corpus/",  "Gold standard"),
            ("evaluation/",   "Metrics + annotation"),
            ("main.py",       "CLI entry point"),
            ("app.py",        "Streamlit frontend"),
        ]:
            st.markdown(
                f'<div style="display:flex;'
                f'justify-content:space-between;'
                f'padding:5px 0;border-bottom:1px solid #f9fafb;'
                f'font-size:0.83rem;">'
                f'<code style="color:#1565c0;">{fname}</code>'
                f'<span style="color:#9ca3af;">{desc}</span></div>',
                unsafe_allow_html=True,
            )
        sc_close()

        sc_open("How to Run")
        st.code(
            "pip install -r requirements.txt\n"
            "streamlit run app.py\n"
            "uv run main.py\n"
            "uv run test_evaluation.py\n"
            "uv run test_corpus.py",
            language="bash",
        )
        sc_close()


# ---------------------------------------------------------------------------
# MAIN ROUTER
# ---------------------------------------------------------------------------
def main():
    page, model, max_papers = render_sidebar()

    if page != st.session_state.get("page"):
        st.session_state["page"] = page

    p = st.session_state["page"]

    if   "Home"           in p: page_home()
    elif "Run Experiment" in p: page_run_experiment(model, max_papers)
    elif "Evaluation"     in p: page_evaluation()
    elif "History"        in p: page_history()
    elif "About"          in p: page_about()


if __name__ == "__main__":
    main()
