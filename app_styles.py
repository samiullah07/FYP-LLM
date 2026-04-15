CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

:root {
    --bg-main: #07111f;
    --bg-secondary: #0b1628;
    --panel: rgba(14, 24, 42, 0.78);
    --panel-2: rgba(18, 32, 56, 0.88);
    --panel-3: rgba(10, 20, 36, 0.96);
    --border: rgba(138, 180, 248, 0.14);
    --text: #eaf2ff;
    --muted: #9db0cf;
    --soft: #7082a5;
    --accent: #5ee7df;
    --accent-2: #7c83ff;
    --accent-3: #9d4edd;
    --success: #34d399;
    --warning: #fbbf24;
    --danger: #fb7185;
    --shadow: 0 14px 40px rgba(0, 0, 0, 0.28);
    --radius-xl: 24px;
    --radius-lg: 18px;
    --radius-md: 14px;
    --radius-sm: 10px;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    color: var(--text);
    background:
        radial-gradient(circle at 15% 20%, rgba(94, 231, 223, 0.10), transparent 24%),
        radial-gradient(circle at 85% 12%, rgba(124, 131, 255, 0.14), transparent 24%),
        radial-gradient(circle at 75% 78%, rgba(157, 78, 221, 0.10), transparent 24%),
        linear-gradient(135deg, #07111f 0%, #091423 42%, #0a1220 100%);
}

.block-container {
    padding-top: 1.35rem;
    padding-bottom: 2rem;
    max-width: 1380px;
}

[data-testid="stSidebar"] {
    background:
        radial-gradient(circle at top left, rgba(94, 231, 223, 0.10), transparent 28%),
        linear-gradient(180deg, #09111f 0%, #060b15 100%) !important;
    border-right: 1px solid rgba(138, 180, 248, 0.10);
}

[data-testid="stSidebar"] * {
    color: var(--text) !important;
}

.sidebar-brand {
    background: linear-gradient(135deg, rgba(94,231,223,0.16), rgba(124,131,255,0.14));
    border: 1px solid rgba(138, 180, 248, 0.16);
    border-radius: 22px;
    padding: 18px 16px;
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.05);
    margin-bottom: 1rem;
}

.sidebar-brand-title {
    font-size: 1.08rem;
    font-weight: 800;
    color: #ffffff;
    margin-top: 0.25rem;
    letter-spacing: -0.02em;
}

.sidebar-brand-subtitle {
    font-size: 0.78rem;
    color: var(--muted);
    margin-top: 0.25rem;
    line-height: 1.55;
}

.sidebar-section-label {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.16em;
    color: var(--soft);
    margin: 1rem 0 0.45rem 0;
    font-weight: 700;
}

.sidebar-mini-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(138, 180, 248, 0.10);
    border-radius: 16px;
    padding: 12px 14px;
    margin-top: 12px;
}

.hero-shell {
    position: relative;
    overflow: hidden;
    background:
        radial-gradient(circle at top left, rgba(94, 231, 223, 0.18), transparent 28%),
        radial-gradient(circle at right, rgba(124, 131, 255, 0.18), transparent 30%),
        linear-gradient(135deg, rgba(16, 26, 48, 0.94), rgba(10, 18, 34, 0.98));
    border: 1px solid rgba(138, 180, 248, 0.16);
    border-radius: 28px;
    padding: 34px 34px 30px 34px;
    margin-bottom: 24px;
    box-shadow: var(--shadow);
}

.hero-shell::before {
    content: "";
    position: absolute;
    width: 260px;
    height: 260px;
    top: -120px;
    right: -80px;
    background: radial-gradient(circle, rgba(94, 231, 223, 0.18), transparent 62%);
    filter: blur(10px);
}

.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 7px 14px;
    border-radius: 999px;
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(138, 180, 248, 0.14);
    color: var(--muted);
    font-size: 0.74rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 0.9rem;
    animation: floatBadge 4.5s ease-in-out infinite;
}

.hero-title {
    color: #ffffff;
    font-size: clamp(1.9rem, 2.3vw, 3rem);
    line-height: 1.08;
    letter-spacing: -0.035em;
    font-weight: 800;
    margin: 0;
    max-width: 780px;
}

.hero-title span {
    background: linear-gradient(135deg, #ffffff 10%, #9dd9ff 45%, #72f0da 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-subtitle {
    color: var(--muted);
    font-size: 0.98rem;
    line-height: 1.7;
    margin: 14px 0 0 0;
    max-width: 860px;
}

.hero-stat-row {
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
    margin-top: 22px;
}

.hero-stat-pill {
    background: rgba(255,255,255,0.045);
    border: 1px solid rgba(138, 180, 248, 0.12);
    color: #dbeafe;
    padding: 10px 14px;
    border-radius: 999px;
    font-size: 0.82rem;
    font-weight: 600;
    backdrop-filter: blur(8px);
}

.metric-card {
    background: linear-gradient(180deg, rgba(16, 28, 49, 0.96), rgba(11, 20, 37, 0.98));
    border: 1px solid rgba(138, 180, 248, 0.10);
    border-radius: 18px;
    padding: 20px 18px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.18);
    margin-bottom: 16px;
    text-align: center;
    transition: transform 0.24s ease, box-shadow 0.24s ease, border-color 0.24s ease;
}

.metric-card:hover {
    transform: translateY(-4px);
    border-color: rgba(138, 180, 248, 0.22);
    box-shadow: 0 18px 38px rgba(0,0,0,0.24);
}

.metric-card-blue { box-shadow: inset 0 0 0 1px rgba(124,131,255,0.10); }
.metric-card-green { box-shadow: inset 0 0 0 1px rgba(52,211,153,0.12); }
.metric-card-red { box-shadow: inset 0 0 0 1px rgba(251,113,133,0.12); }
.metric-card-orange { box-shadow: inset 0 0 0 1px rgba(251,191,36,0.12); }
.metric-card-teal { box-shadow: inset 0 0 0 1px rgba(94,231,223,0.12); }

.metric-value {
    font-size: 2rem;
    font-weight: 800;
    color: #f8fbff;
    line-height: 1;
    letter-spacing: -0.03em;
}

.metric-label {
    font-size: 0.75rem;
    color: var(--muted);
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-top: 8px;
}

.section-card {
    background: linear-gradient(180deg, rgba(13, 24, 43, 0.94), rgba(10, 18, 34, 0.98));
    border-radius: 22px;
    padding: 24px;
    border: 1px solid rgba(138, 180, 248, 0.10);
    box-shadow: 0 12px 36px rgba(0,0,0,0.16);
    margin-bottom: 22px;
    transition: all 0.24s ease;
}

.section-card:hover {
    border-color: rgba(138, 180, 248, 0.18);
    transform: translateY(-2px);
}

.section-title {
    color: #f9fbff;
    font-size: 1.02rem;
    font-weight: 700;
    margin-bottom: 16px;
    padding-bottom: 10px;
    border-bottom: 1px solid rgba(138, 180, 248, 0.10);
    letter-spacing: -0.01em;
}

.soft-text {
    color: var(--muted);
}

.pipeline-node {
    background: linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.015));
    border: 1px solid rgba(138, 180, 248, 0.10);
    border-radius: 18px;
    padding: 18px 16px;
    min-height: 165px;
    text-align: center;
    transition: all 0.25s ease;
    box-shadow: 0 8px 24px rgba(0,0,0,0.12);
}

.pipeline-node:hover {
    transform: translateY(-4px);
    border-color: rgba(138, 180, 248, 0.20);
}

.pipeline-node-icon {
    font-size: 1.9rem;
    margin-bottom: 8px;
}

.pipeline-node-title {
    font-size: 0.98rem;
    font-weight: 700;
    color: #eef4ff;
    margin-bottom: 6px;
}

.pipeline-node-desc {
    font-size: 0.82rem;
    color: var(--muted);
    line-height: 1.55;
}

.agent-step {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 14px 18px;
    border-radius: 14px;
    margin-bottom: 10px;
    font-size: 0.92rem;
    font-weight: 600;
    border: 1px solid transparent;
    transition: all 0.2s ease;
}

.step-pending {
    background: rgba(255,255,255,0.03);
    color: #7f90af;
    border-color: rgba(138, 180, 248, 0.08);
}

.step-running {
    background: linear-gradient(135deg, rgba(94,231,223,0.10), rgba(124,131,255,0.15));
    color: #c9f5ef;
    border-color: rgba(94,231,223,0.22);
    box-shadow: 0 8px 20px rgba(94,231,223,0.08);
    animation: pulseGlow 1.6s ease-in-out infinite;
}

.step-done {
    background: linear-gradient(135deg, rgba(52,211,153,0.10), rgba(16,185,129,0.12));
    color: #a7f3d0;
    border-color: rgba(52,211,153,0.18);
}

.step-error {
    background: linear-gradient(135deg, rgba(251,113,133,0.12), rgba(127,29,29,0.18));
    color: #fecdd3;
    border-color: rgba(251,113,133,0.22);
}

.badge-valid,
.badge-partial,
.badge-hallucinated {
    padding: 5px 12px;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    white-space: nowrap;
}

.badge-valid {
    background: rgba(52,211,153,0.12);
    color: #86efac;
    border: 1px solid rgba(52,211,153,0.20);
}

.badge-partial {
    background: rgba(251,191,36,0.12);
    color: #fde68a;
    border: 1px solid rgba(251,191,36,0.20);
}

.badge-hallucinated {
    background: rgba(251,113,133,0.12);
    color: #fecdd3;
    border: 1px solid rgba(251,113,133,0.20);
}

.review-box {
    background: rgba(8, 15, 28, 0.70);
    border: 1px solid rgba(138, 180, 248, 0.10);
    border-radius: 16px;
    padding: 22px;
    font-size: 0.95rem;
    line-height: 1.85;
    color: #dce7fb;
    max-height: 460px;
    overflow-y: auto;
    white-space: pre-wrap;
}

.citation-row {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: 10px;
    padding: 12px 14px;
    border-radius: 14px;
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(138, 180, 248, 0.08);
    margin-bottom: 10px;
    font-size: 0.88rem;
}

.subquery-pill {
    padding: 9px 12px;
    margin-bottom: 8px;
    background: rgba(124,131,255,0.10);
    border: 1px solid rgba(124,131,255,0.16);
    border-radius: 12px;
    font-size: 0.84rem;
    color: #dfe6ff;
}

.floating-loader {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 16px 18px;
    background: linear-gradient(180deg, rgba(17, 29, 52, 0.96), rgba(11, 20, 36, 0.98));
    border: 1px solid rgba(138,180,248,0.14);
    border-radius: 18px;
    box-shadow: 0 16px 38px rgba(0,0,0,0.22);
    margin: 10px 0 16px 0;
}

.loader-ring {
    width: 24px;
    height: 24px;
    border-radius: 50%;
    border: 3px solid rgba(255,255,255,0.10);
    border-top-color: var(--accent);
    border-right-color: var(--accent-2);
    animation: spin 1s linear infinite;
}

.loader-text {
    color: #ebf4ff;
    font-size: 0.92rem;
    font-weight: 600;
}

.result-highlight {
    background: linear-gradient(135deg, rgba(94,231,223,0.10), rgba(124,131,255,0.10));
    border: 1px solid rgba(138,180,248,0.14);
    border-radius: 18px;
    padding: 14px 16px;
    margin-bottom: 14px;
}

.stButton > button {
    background: linear-gradient(135deg, #5b7cfa 0%, #3dd9d6 100%) !important;
    color: #051120 !important;
    border: none !important;
    border-radius: 14px !important;
    padding: 0.8rem 1.2rem !important;
    font-weight: 800 !important;
    font-size: 0.94rem !important;
    box-shadow: 0 12px 24px rgba(61, 217, 214, 0.18) !important;
    transition: all 0.24s ease !important;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 18px 32px rgba(61, 217, 214, 0.22) !important;
    filter: brightness(1.05);
}

.stButton > button:focus {
    outline: none !important;
    box-shadow: 0 0 0 3px rgba(94,231,223,0.16), 0 12px 24px rgba(61,217,214,0.18) !important;
}

.stTextArea textarea,
.stTextInput input,
.stSelectbox [data-baseweb="select"] > div,
.stMultiSelect [data-baseweb="select"] > div {
    background: rgba(9, 18, 33, 0.90) !important;
    color: #eef4ff !important;
    border: 1px solid rgba(138, 180, 248, 0.16) !important;
    border-radius: 14px !important;
}

.stTextArea textarea:focus,
.stTextInput input:focus {
    border-color: rgba(94,231,223,0.35) !important;
    box-shadow: 0 0 0 3px rgba(94,231,223,0.10) !important;
}

.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.03);
    border-radius: 14px;
    padding: 6px;
    gap: 6px;
    border: 1px solid rgba(138,180,248,0.10);
}

.stTabs [data-baseweb="tab"] {
    border-radius: 12px !important;
    font-weight: 700 !important;
    color: var(--muted) !important;
    padding: 10px 18px !important;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, rgba(94,231,223,0.16), rgba(124,131,255,0.22)) !important;
    color: white !important;
    border: 1px solid rgba(138,180,248,0.14) !important;
}

.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #5ee7df, #7c83ff, #9d4edd) !important;
}

.stExpander > details {
    background: rgba(14, 24, 42, 0.75) !important;
    border: 1px solid rgba(138, 180, 248, 0.10) !important;
    border-radius: 16px !important;
}

.stDataFrame, [data-testid="stDataFrame"] {
    background: rgba(14, 24, 42, 0.75) !important;
    border-radius: 16px !important;
    border: 1px solid rgba(138, 180, 248, 0.10) !important;
}

.stAlert > div {
    border-radius: 14px !important;
    border: 1px solid rgba(138, 180, 248, 0.10) !important;
    background: rgba(14, 24, 42, 0.82) !important;
}

hr {
    border-color: rgba(138,180,248,0.08) !important;
}

#MainMenu, footer, header {
    visibility: hidden;
}

@keyframes spin {
    to { transform: rotate(360deg); }
}

@keyframes pulseGlow {
    0%, 100% { box-shadow: 0 8px 20px rgba(94,231,223,0.06); }
    50% { box-shadow: 0 10px 26px rgba(124,131,255,0.14); }
}

@keyframes floatBadge {
    0%, 100% { transform: translateY(0px); }
    50% { transform: translateY(-2px); }
}

@media (max-width: 900px) {
    .hero-shell {
        padding: 24px 20px;
        border-radius: 22px;
    }

    .hero-title {
        font-size: 1.65rem;
    }

    .hero-subtitle {
        font-size: 0.92rem;
    }

    .metric-value {
        font-size: 1.7rem;
    }

    .section-card {
        padding: 18px;
    }

    .citation-row {
        flex-direction: column;
        align-items: flex-start;
    }
}
"""