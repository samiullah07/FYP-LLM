"""
Agentic AI for Reliable Academic Literature Review
Streamlit Demo Application (Standalone with Mock Data)

University of Hertfordshire — MSc Data Science
Final Year Project 2025

This is a standalone demo application that uses mock data for demonstration
purposes. It can run without the actual backend being configured.

Run with: streamlit run demo_app.py
"""

import streamlit as st
import pandas as pd
import time
from datetime import datetime

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="Agentic Literature Review System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': "https://github.com/your-repo/fyp-llm",
        'Report a bug': "https://github.com/your-repo/fyp-llm/issues",
        'About': """
        # Agentic AI for Reliable Academic Literature Review

        **University of Hertfordshire** — MSc Data Science
        **Final Year Project 2025**

        This system uses multiple AI agents to generate literature reviews
        with verified citations, reducing hallucination rates compared to
        standard LLM approaches.
        """
    }
)

# =============================================================================
# CUSTOM CSS STYLING
# =============================================================================

st.markdown("""
<style>
    /* Main header styling */
    .main-header {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d5a87 100%);
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        color: white;
    }

    .main-header h1 {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
    }

    .main-header p {
        font-size: 1.1rem;
        opacity: 0.9;
        margin-top: 0.5rem;
    }

    /* Status badge styling */
    .status-valid {
        background-color: #d4edda;
        color: #155724;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }

    .status-partial {
        background-color: #fff3cd;
        color: #856404;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }

    .status-hallucinated {
        background-color: #f8d7da;
        color: #721c24;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }

    /* Footer styling */
    .footer {
        margin-top: 3rem;
        padding-top: 2rem;
        border-top: 1px solid #e0e0e0;
        text-align: center;
        color: #666;
        font-size: 0.9rem;
    }

    /* Review text box */
    .review-box {
        background: #fafafa;
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        padding: 20px;
        font-size: 0.95rem;
        line-height: 1.8;
        color: #374151;
        max-height: 500px;
        overflow-y: auto;
        white-space: pre-wrap;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# MOCK DATA GENERATORS
# =============================================================================

def get_mock_review(topic: str, system_type: str = "agentic") -> str:
    """Generate a mock literature review based on the topic."""

    if system_type == "agentic":
        return f"""## Literature Review: {topic.title()}

The intersection of artificial intelligence and academic research has garnered significant attention in recent years. This review synthesizes current findings on reliable literature generation using multi-agent AI systems.

### Multi-Agent Systems in Academic Contexts

The application of multi-agent systems for academic tasks has shown promising results. According to Zhang et al. (2024) [1], agent-based approaches can significantly reduce error rates in information retrieval tasks. Their study demonstrated a 40% reduction in factual inconsistencies when using specialized agents for different subtasks.

Chen and Williams (2023) [2] extended this work to the domain of literature reviews, finding that decomposition of complex queries into focused sub-queries improved retrieval precision by 35%. This finding has been corroborated by Kumar et al. (2024) [3], who reported similar improvements across multiple academic domains.

### Hallucination Mitigation Strategies

A critical challenge in AI-generated academic content is the phenomenon of hallucination. Thompson et al. (2024) [4] identified three primary categories of hallucination in LLM-generated text: fabrication of sources, misattribution of claims, and numerical distortions. Their verification framework achieved 89% accuracy in detecting hallucinated citations.

The work of Martinez and Lee (2023) [5] introduced a post-generation verification pipeline that cross-references all citations against academic databases. Their approach reduced hallucination rates from 12% to 2.3% in controlled experiments.

### Retrieval-Augmented Generation

Recent advances in retrieval-augmented generation (RAG) have provided additional mechanisms for improving accuracy. Brown et al. (2024) [6] demonstrated that grounding generation in retrieved documents significantly reduces fabrication. However, they note that retrieval quality remains a limiting factor.

### Current Limitations and Future Directions

Despite these advances, several challenges remain. The work of Anderson (2024) [7] highlights the computational overhead of multi-agent systems as a barrier to widespread adoption. Additionally, Patel et al. (2023) [8] note that verification systems may introduce false negatives, potentially discarding valid but hard-to-match citations.

Future research should focus on optimizing the balance between verification thoroughness and computational efficiency, as well as developing more robust matching algorithms for citation verification.
"""
    else:  # baseline
        return f"""## Literature Review: {topic.title()}

Artificial intelligence has become increasingly important in academic research and literature review generation. This section provides an overview of current approaches and their effectiveness.

Recent work by Smith et al. (2023) [1] explored the use of large language models for academic writing assistance. They found that LLMs can generate coherent text but often struggle with accurate citation. The study reported a hallucination rate of approximately 15% in generated references.

Johnson and Brown (2024) [2] investigated retrieval-augmented approaches to improve citation accuracy. Their system showed modest improvements but still produced hallucinated citations in 8% of cases.

The application of multi-agent systems has been proposed by Davis (2023) [3] as a potential solution. Early results suggest that decomposing tasks among specialized agents may improve overall accuracy, though comprehensive evaluation is still needed.

Wilson et al. (2024) [4] examined the role of verification in reducing hallucinations. Their post-hoc verification approach identified 70% of hallucinated citations but could not prevent their initial generation.

Comparative studies by Garcia (2023) [5] and Miller et al. (2024) [6] have begun to establish benchmarks for evaluating AI-generated literature reviews. However, standardized evaluation protocols remain an open challenge in the field.

The current state of research indicates that while significant progress has been made, there remains substantial room for improvement in the reliability of AI-generated academic content.
"""


def get_mock_verification_data(system_type: str = "agentic") -> list:
    """Generate mock verification results for citations."""

    if system_type == "agentic":
        return [
            {"reference": "Zhang et al. (2024) - Agent-based approaches in information retrieval", "status": "Valid", "doi_match": True, "author_match": True, "year_match": True, "openalex_url": "https://openalex.org/W123456789", "confidence": 0.98},
            {"reference": "Chen and Williams (2023) - Query decomposition for academic search", "status": "Valid", "doi_match": True, "author_match": True, "year_match": True, "openalex_url": "https://openalex.org/W234567890", "confidence": 0.95},
            {"reference": "Kumar et al. (2024) - Multi-domain evaluation of agent systems", "status": "Valid", "doi_match": True, "author_match": True, "year_match": True, "openalex_url": "https://openalex.org/W345678901", "confidence": 0.92},
            {"reference": "Thompson et al. (2024) - Hallucination categories in LLM text", "status": "Valid", "doi_match": True, "author_match": True, "year_match": True, "openalex_url": "https://openalex.org/W456789012", "confidence": 0.96},
            {"reference": "Martinez and Lee (2023) - Post-generation verification pipeline", "status": "Partial", "doi_match": True, "author_match": True, "year_match": False, "openalex_url": "https://openalex.org/W567890123", "confidence": 0.78, "note": "Year mismatch: found 2022"},
            {"reference": "Brown et al. (2024) - RAG for academic generation", "status": "Valid", "doi_match": True, "author_match": True, "year_match": True, "openalex_url": "https://openalex.org/W678901234", "confidence": 0.94},
            {"reference": "Anderson (2024) - Computational overhead in multi-agent systems", "status": "Valid", "doi_match": True, "author_match": True, "year_match": True, "openalex_url": "https://openalex.org/W789012345", "confidence": 0.91},
            {"reference": "Patel et al. (2023) - False negatives in verification", "status": "Partial", "doi_match": True, "author_match": True, "year_match": True, "openalex_url": "https://openalex.org/W890123456", "confidence": 0.82, "note": "Author order differs"},
        ]
    else:  # baseline
        return [
            {"reference": "Smith et al. (2023) - LLMs for academic writing", "status": "Valid", "doi_match": True, "author_match": True, "year_match": True, "openalex_url": "https://openalex.org/W111222333", "confidence": 0.89},
            {"reference": "Johnson and Brown (2024) - Retrieval-augmented approaches", "status": "Hallucinated", "doi_match": False, "author_match": False, "year_match": False, "openalex_url": None, "confidence": 0.0, "note": "No matching paper found"},
            {"reference": "Davis (2023) - Multi-agent systems proposal", "status": "Valid", "doi_match": True, "author_match": True, "year_match": True, "openalex_url": "https://openalex.org/W222333444", "confidence": 0.87},
            {"reference": "Wilson et al. (2024) - Verification in hallucination reduction", "status": "Partial", "doi_match": True, "author_match": True, "year_match": False, "openalex_url": "https://openalex.org/W333444555", "confidence": 0.71, "note": "Year mismatch: found 2023"},
            {"reference": "Garcia (2023) - Benchmarking AI literature reviews", "status": "Hallucinated", "doi_match": False, "author_match": False, "year_match": False, "openalex_url": None, "confidence": 0.0, "note": "No matching paper found"},
            {"reference": "Miller et al. (2024) - Evaluation protocols", "status": "Valid", "doi_match": True, "author_match": True, "year_match": True, "openalex_url": "https://openalex.org/W444555666", "confidence": 0.93},
        ]


def get_mock_metrics(system_type: str = "agentic", elapsed_time: float = 2.5) -> dict:
    """Generate mock system metrics."""

    if system_type == "agentic":
        return {
            "papers_retrieved": 20,
            "sub_queries_used": 5,
            "total_citations": 8,
            "valid_citations": 6,
            "partial_citations": 2,
            "hallucinated_citations": 0,
            "hallucination_rate": 0.0,
            "citation_precision": 1.0,
            "api_calls_made": 12,
            "processing_time": elapsed_time,
            "tokens_used": 4250,
        }
    else:
        return {
            "papers_retrieved": 15,
            "sub_queries_used": 1,
            "total_citations": 6,
            "valid_citations": 3,
            "partial_citations": 1,
            "hallucinated_citations": 2,
            "hallucination_rate": 0.333,
            "citation_precision": 0.667,
            "api_calls_made": 3,
            "processing_time": elapsed_time,
            "tokens_used": 2100,
        }


# =============================================================================
# BACKEND INTEGRATION HOOKS (PLACEHOLDERS WITH MOCK DATA)
# =============================================================================

def run_agentic_review(topic: str, num_papers: int = 20, strictness: str = "strict") -> tuple:
    """
    Run the agentic literature review pipeline.

    This is a placeholder function that returns mock data for demonstration.
    Replace with actual backend integration when ready.
    """
    review = get_mock_review(topic, system_type="agentic")
    verification = get_mock_verification_data(system_type="agentic")
    metrics = get_mock_metrics(system_type="agentic")

    # Adjust based on strictness
    if strictness == "lenient":
        for v in verification:
            if v["status"] == "Partial":
                v["status"] = "Valid"
                v["confidence"] += 0.1
        metrics["hallucination_rate"] = 0.0
        metrics["valid_citations"] = len(verification)
        metrics["partial_citations"] = 0

    return review, verification, metrics


def run_baseline_review(topic: str, num_papers: int = 15) -> tuple:
    """
    Run the baseline (single-LLM) literature review pipeline.

    This is a placeholder function for comparison purposes.
    """
    review = get_mock_review(topic, system_type="baseline")
    verification = get_mock_verification_data(system_type="baseline")
    metrics = get_mock_metrics(system_type="baseline")

    return review, verification, metrics


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def status_badge(status: str) -> str:
    """Generate HTML for a status badge."""
    status_lower = status.lower()
    if status_lower == "valid":
        return f'<span class="status-valid">✓ {status}</span>'
    elif status_lower == "partial":
        return f'<span class="status-partial">⚠ {status}</span>'
    else:
        return f'<span class="status-hallucinated">✗ {status}</span>'


def format_metrics_table(metrics: dict) -> pd.DataFrame:
    """Format metrics as a DataFrame."""
    data = {
        "Metric": [
            "Papers Retrieved",
            "Sub-queries Used",
            "Total Citations",
            "Valid Citations",
            "Partial Citations",
            "Hallucinated Citations",
            "Hallucination Rate",
            "Citation Precision",
            "API Calls Made",
            "Processing Time",
            "Tokens Used",
        ],
        "Value": [
            metrics.get("papers_retrieved", 0),
            metrics.get("sub_queries_used", 0),
            metrics.get("total_citations", 0),
            metrics.get("valid_citations", 0),
            metrics.get("partial_citations", 0),
            metrics.get("hallucinated_citations", 0),
            f"{metrics.get('hallucination_rate', 0):.1%}",
            f"{metrics.get('citation_precision', 0):.1%}",
            metrics.get("api_calls_made", 0),
            f"{metrics.get('processing_time', 0):.2f}s",
            metrics.get("tokens_used", 0),
        ],
    }
    return pd.DataFrame(data)


# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    """Main Streamlit application."""

    # Header
    st.markdown("""
    <div class="main-header">
        <h1>📚 Agentic Literature Review System</h1>
        <p>Multi-Agent AI for Reliable Academic Literature Review with Verified Citations</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar configuration
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")

        # Sample topics
        sample_topics = [
            "The impact of large language models on academic writing quality",
            "Retrieval-augmented generation for reducing hallucinations in LLMs",
            "Multi-agent systems for automated fact-checking",
            "Neural network approaches to citation verification",
            "Transformer models in scientific literature analysis",
        ]

        if st.button("📋 Use Sample Topic", use_container_width=True):
            if "sample_index" not in st.session_state:
                st.session_state.sample_index = 0
            st.session_state.topic = sample_topics[st.session_state.sample_index]
            st.session_state.sample_index = (st.session_state.sample_index + 1) % len(sample_topics)

        # Topic input
        topic = st.text_area(
            "Research Topic",
            value=st.session_state.get("topic", ""),
            placeholder="Enter your research topic...",
            height=100,
            help="Enter the research topic for your literature review"
        )

        # Number of papers
        num_papers = st.slider(
            "Number of Papers to Retrieve",
            min_value=5,
            max_value=50,
            value=20,
            step=5,
            help="Select how many academic papers to retrieve for the review"
        )

        # Verification strictness
        strictness = st.radio(
            "Verification Strictness",
            options=["strict", "lenient"],
            index=0,
            help="Strict mode: flags any discrepancies. Lenient mode: allows minor variations."
        )

        st.markdown("---")

        # System comparison toggle
        compare_mode = st.checkbox(
            "🔄 Compare with Baseline",
            value=False,
            help="Enable side-by-side comparison with single-LLM baseline"
        )

        st.markdown("---")

        # Generate button
        generate_clicked = st.button(
            "🚀 Generate Review",
            type="primary",
            use_container_width=True,
            disabled=not topic.strip()
        )

    # Initialize session state
    if "generated" not in st.session_state:
        st.session_state.generated = False
        st.session_state.agentic_results = None
        st.session_state.baseline_results = None

    # Handle generation
    if generate_clicked:
        st.session_state.generating = True

        # Create loading state
        with st.spinner(""):
            loading_container = st.empty()

            with loading_container:
                st.markdown("""
                <div style="text-align: center; padding: 2rem;">
                    <h3>🔄 Generating Literature Review...</h3>
                    <p>The multi-agent system is processing your request:</p>
                    <ul style="display: inline-block; text-align: left; color: #666;">
                        <li>✓ Planning sub-queries</li>
                        <li>✓ Searching academic databases</li>
                        <li>✓ Synthesizing literature review</li>
                        <li>✓ Verifying all citations</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)

                progress_bar = st.progress(0)

                # Simulate processing stages
                stages = [
                    "Planning queries...",
                    "Retrieving papers...",
                    "Writing review...",
                    "Verifying citations...",
                ]
                for i, stage in enumerate(stages):
                    time.sleep(0.5)
                    progress_bar.progress((i + 1) * 25, text=stage)

            loading_container.empty()

        # Generate results
        agentic_review, agentic_verification, agentic_metrics = run_agentic_review(
            topic, num_papers, strictness
        )
        st.session_state.agentic_results = {
            "review": agentic_review,
            "verification": agentic_verification,
            "metrics": agentic_metrics,
        }

        if compare_mode:
            baseline_review, baseline_verification, baseline_metrics = run_baseline_review(
                topic, num_papers
            )
            st.session_state.baseline_results = {
                "review": baseline_review,
                "verification": baseline_verification,
                "metrics": baseline_metrics,
            }

        st.session_state.generated = True
        st.session_state.generating = False
        st.rerun()

    # Display results
    if st.session_state.generated and st.session_state.agentic_results:
        results = st.session_state.agentic_results

        if compare_mode and st.session_state.baseline_results:
            # Comparison mode
            baseline = st.session_state.baseline_results

            tab1, tab2, tab3 = st.tabs([
                "📊 Comparison",
                "📄 Agentic Review",
                "📄 Baseline Review"
            ])

            with tab1:
                st.subheader("System Comparison")

                # Metrics comparison
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("#### Agentic Metrics")
                    st.dataframe(
                        format_metrics_table(results["metrics"]),
                        use_container_width=True,
                        hide_index=True
                    )

                with col2:
                    st.markdown("#### Baseline Metrics")
                    st.dataframe(
                        format_metrics_table(baseline["metrics"]),
                        use_container_width=True,
                        hide_index=True
                    )

                # Hallucination comparison chart
                st.markdown("---")
                st.subheader("Hallucination Rate Comparison")

                comparison_data = pd.DataFrame({
                    "System": ["Agentic", "Baseline"],
                    "Hallucination Rate": [
                        results["metrics"]["hallucination_rate"],
                        baseline["metrics"]["hallucination_rate"]
                    ],
                })

                st.bar_chart(
                    comparison_data.set_index("System"),
                    color="#2d5a87"
                )

                improvement = (
                    baseline["metrics"]["hallucination_rate"] -
                    results["metrics"]["hallucination_rate"]
                )

                if improvement > 0:
                    st.success(
                        f"🎉 The agentic system reduced hallucination rate by "
                        f"{improvement:.1%} compared to baseline!"
                    )
                elif improvement == 0:
                    st.info("Both systems achieved the same hallucination rate.")
                else:
                    st.warning(
                        f"The baseline had a lower hallucination rate by "
                        f"{abs(improvement):.1%} in this run."
                    )

            with tab2:
                st.markdown(results["review"])
                st.markdown("---")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Papers", results["metrics"]["papers_retrieved"])
                with col2:
                    st.metric("Citations", results["metrics"]["total_citations"])
                with col3:
                    st.metric("Hallucination Rate", f"{results['metrics']['hallucination_rate']:.1%}")

            with tab3:
                st.markdown(baseline["review"])
                st.markdown("---")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Papers", baseline["metrics"]["papers_retrieved"])
                with col2:
                    st.metric("Citations", baseline["metrics"]["total_citations"])
                with col3:
                    st.metric("Hallucination Rate", f"{baseline['metrics']['hallucination_rate']:.1%}")

        else:
            # Single system view
            tab1, tab2, tab3 = st.tabs([
                "📄 Literature Review",
                "🔍 Verification Report",
                "📈 System Metrics"
            ])

            with tab1:
                st.markdown(results["review"])

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Papers", results["metrics"]["papers_retrieved"])
                with col2:
                    st.metric("Citations", results["metrics"]["total_citations"])
                with col3:
                    st.metric("Valid", results["metrics"]["valid_citations"])
                with col4:
                    st.metric("Hallucination Rate", f"{results['metrics']['hallucination_rate']:.1%}")

            with tab2:
                st.subheader("Citation Verification Results")

                # Create verification table
                verification_df = pd.DataFrame(results["verification"])

                # Add styled status column
                verification_df["status_badge"] = verification_df["status"].apply(status_badge)

                st.dataframe(
                    verification_df[["reference", "status_badge", "doi_match", "author_match", "year_match", "confidence"]].rename(columns={
                        "reference": "Reference",
                        "status_badge": "Status",
                        "doi_match": "DOI Match",
                        "author_match": "Author Match",
                        "year_match": "Year Match",
                        "confidence": "Confidence",
                    }),
                    use_container_width=True,
                    hide_index=True,
                )

                # Summary statistics
                st.markdown("---")
                st.subheader("Verification Summary")

                status_counts = verification_df["status"].value_counts()

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Valid", status_counts.get("Valid", 0))
                with col2:
                    st.metric("Partial", status_counts.get("Partial", 0))
                with col3:
                    st.metric("Hallucinated", status_counts.get("Hallucinated", 0))

            with tab3:
                st.subheader("System Performance Metrics")

                metrics_df = format_metrics_table(results["metrics"])

                st.dataframe(
                    metrics_df,
                    use_container_width=True,
                    hide_index=True,
                )

    # Initial state - show welcome message
    else:
        st.markdown("""
        <div style="text-align: center; padding: 3rem; background: #f8f9fa; border-radius: 12px; margin-top: 2rem;">
            <h2 style="color: #1e3a5f;">Welcome to the Agentic Literature Review System</h2>
            <p style="color: #666; max-width: 600px; margin: 1rem auto; line-height: 1.6;">
                This demo showcases a multi-agent AI system designed to generate reliable
                academic literature reviews with verified citations. The system uses
                specialized agents for planning, searching, summarizing, and verification
                to reduce hallucination rates compared to standard LLM approaches.
            </p>
            <div style="margin-top: 2rem;">
                <span style="display: inline-block; margin: 0.5rem; padding: 0.5rem 1rem; background: #e3f2fd; border-radius: 20px; color: #1565c0;">
                    🎯 Multi-Agent Planning
                </span>
                <span style="display: inline-block; margin: 0.5rem; padding: 0.5rem 1rem; background: #e3f2fd; border-radius: 20px; color: #1565c0;">
                    🔍 Academic Database Search
                </span>
                <span style="display: inline-block; margin: 0.5rem; padding: 0.5rem 1rem; background: #e3f2fd; border-radius: 20px; color: #1565c0;">
                    ✍️ Automated Synthesis
                </span>
                <span style="display: inline-block; margin: 0.5rem; padding: 0.5rem 1rem; background: #e3f2fd; border-radius: 20px; color: #1565c0;">
                    ✅ Citation Verification
                </span>
            </div>
            <p style="margin-top: 2rem; color: #999;">
                ← Enter a research topic in the sidebar and click <strong>"Generate Review"</strong> to begin
            </p>
        </div>
        """, unsafe_allow_html=True)

    # Footer
    st.markdown("""
    <div class="footer">
        <p>
            <strong>Agentic AI for Reliable Academic Literature Review</strong><br>
            University of Hertfordshire — MSc Data Science Final Year Project 2025<br>
            Demo Mode (Mock Data) · Built with Streamlit
        </p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
