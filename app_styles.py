# app_styles.py
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background-color: #f4f6fb; }
[data-testid="stSidebar"] { background: linear-gradient(180deg, #1a2b5e 0%, #0d1b3e 100%); }
[data-testid="stSidebar"] * { color: white !important; }
.header-banner { background: linear-gradient(135deg, #1a2b5e 0%, #0d47a1 50%, #1565c0 100%); padding: 28px 32px; border-radius: 16px; margin-bottom: 24px; display: flex; align-items: center; gap: 20px; box-shadow: 0 4px 20px rgba(26,43,94,0.3); }
.header-title { color: white; font-size: 1.8rem; font-weight: 700; margin: 0; }
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
.step-pending { background: #f3f4f6; color: #9ca3af; }
.step-running { background: #dbeafe; color: #1d4ed8; border: 1px solid #93c5fd; }
.step-done { background: #d1fae5; color: #065f46; }
.step-error { background: #fee2e2; color: #991b1b; }
.badge-valid { background: #d1fae5; color: #065f46; padding: 3px 10px; border-radius: 12px; font-size: 0.78rem; font-weight: 600; }
.badge-partial { background: #fef3c7; color: #92400e; padding: 3px 10px; border-radius: 12px; font-size: 0.78rem; font-weight: 600; }
.badge-hallucinated { background: #fee2e2; color: #991b1b; padding: 3px 10px; border-radius: 12px; font-size: 0.78rem; font-weight: 600; }
.review-box { background: #fafafa; border: 1px solid #e5e7eb; border-radius: 10px; padding: 20px; font-size: 0.92rem; line-height: 1.8; color: #374151; max-height: 420px; overflow-y: auto; white-space: pre-wrap; }
.citation-row { display: flex; align-items: center; justify-content: space-between; padding: 10px 14px; border-radius: 8px; background: #f9fafb; border: 1px solid #e5e7eb; margin-bottom: 8px; font-size: 0.88rem; }
.stButton > button { background: linear-gradient(135deg, #1a2b5e, #1565c0) !important; color: white !important; border: none !important; border-radius: 8px !important; padding: 10px 28px !important; font-weight: 600 !important; box-shadow: 0 2px 8px rgba(26,43,94,0.3) !important; }
.stTabs [data-baseweb="tab-list"] { background: white; border-radius: 10px; padding: 4px; gap: 4px; }
.stTabs [data-baseweb="tab"] { border-radius: 8px; font-weight: 500; color: #6b7280 !important; }
.stTabs [aria-selected="true"] { background: #1a2b5e !important; color: white !important; }
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header { visibility: hidden; }
</style>
"""