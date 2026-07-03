"""
app.py — run this with: streamlit run app.py

Interactive & Cinematic UI:
1. Paced Generator Streaming using typist text stream
2. Live Tug-of-War dominance bar indicator
3. Interactive user cross-examination directives injected via coroutines (yield/send)
4. Dynamic Mermaid evidence logic web
5. Clean, decisive purple-gold Judge Verdict dashboard
"""

import time
import streamlit as st
from debate_engine import run_debate_stream

st.set_page_config(page_title="Scientific Arena: Multi-Agent Debate", layout="wide")

# Custom CSS for an intense, highly styled "Scientific Arena" vibe
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800;900&family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Fira+Code:wght@400;500;600&display=swap');

    /* Global styling overrides */
    html, body, [data-testid="stAppViewContainer"] {
        background: linear-gradient(180deg, #040406 0%, #010102 100%);
        color: #e2e8f0;
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    /* Scrollbar overrides */
    ::-webkit-scrollbar {
        width: 6px;
        height: 6px;
    }
    ::-webkit-scrollbar-track {
        background: transparent;
    }
    ::-webkit-scrollbar-thumb {
        background: #111115;
        border-radius: 10px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #222228;
    }

    /* Side bar styling */
    [data-testid="stSidebar"] {
        background-color: #07070a;
        border-right: 1px solid #111115;
    }

    h1, h2, h3, h4, h5, h6 {
        font-family: 'Outfit', sans-serif;
        color: #ffffff;
        font-weight: 800;
        letter-spacing: -0.025em;
    }

    /* Streamlit Tabs custom styling */
    button[data-baseweb="tab"] {
        font-family: 'Outfit', sans-serif !important;
        font-size: 0.92rem !important;
        font-weight: 600 !important;
        color: #64748b !important;
        border: none !important;
        transition: all 0.2s ease !important;
        letter-spacing: 0.02em !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #ffffff !important;
    }
    div[data-baseweb="tab-highlight"] {
        background-color: #10b981 !important;
    }

    /* Streamlit input fields custom styling */
    div[data-testid="stTextArea"] textarea {
        background-color: #08080b !important;
        color: #f1f5f9 !important;
        border: 1px solid #14141a !important;
        border-radius: 12px !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        padding: 12px !important;
        transition: all 0.3s ease !important;
    }
    div[data-testid="stTextArea"] textarea:focus {
        border-color: #10b981 !important;
        box-shadow: 0 0 0 1px #10b981 !important;
    }

    /* Buttons styling */
    div.stButton > button {
        background-color: #08080a !important;
        color: #f8fafc !important;
        border: 1px solid #14141a !important;
        border-radius: 10px !important;
        padding: 10px 24px !important;
        font-weight: 600 !important;
        font-family: 'Outfit', sans-serif !important;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 4px 6px rgba(0,0,0,0.4) !important;
        letter-spacing: 0.02em !important;
    }
    div.stButton > button:hover {
        background-color: #111116 !important;
        border-color: #222228 !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 12px rgba(0,0,0,0.5) !important;
    }
    div.stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%) !important;
        border: none !important;
        box-shadow: 0 4px 15px rgba(16, 185, 129, 0.25) !important;
    }
    div.stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #34d399 0%, #10b981 100%) !important;
        box-shadow: 0 4px 25px rgba(16, 185, 129, 0.45) !important;
        transform: translateY(-2px) !important;
    }

    /* Heartbeat status indicator */
    .arena-status-bar {
        display: flex;
        align-items: center;
        justify-content: center;
        background-color: rgba(8, 8, 10, 0.85);
        backdrop-filter: blur(8px);
        border: 1px solid rgba(255, 255, 255, 0.01);
        padding: 12px 24px;
        border-radius: 50px;
        width: fit-content;
        margin: 0 auto 35px auto;
        box-shadow: 0 10px 30px rgba(0,0,0,0.5);
    }

    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.4); }
        70% { box-shadow: 0 0 0 8px rgba(16, 185, 129, 0); }
        100% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
    }
    
    .pulse-dot-active {
        width: 8px;
        height: 8px;
        background-color: #10b981;
        border-radius: 50%;
        margin-right: 8px;
        animation: pulse 2.5s infinite;
    }

    .pulse-dot-demo {
        width: 8px;
        height: 8px;
        background-color: #fbbf24;
        border-radius: 50%;
        margin-right: 8px;
        animation: pulse 2.5s infinite;
    }

    .status-label {
        font-family: 'Fira Code', monospace;
        font-size: 0.85rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        font-weight: 500;
    }

    /* Decisive judge verdict styling */
    .judge-panel {
        background: #0f1322;
        border: 1px solid rgba(139, 92, 246, 0.25);
        border-left: 4px solid #8b5cf6;
        border-radius: 12px;
        padding: 26px;
        margin-bottom: 40px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
    }

    .judge-header {
        display: flex;
        align-items: center;
        gap: 15px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
        padding-bottom: 18px;
        margin-bottom: 22px;
    }

    .judge-icon {
        font-size: 2.2rem;
    }

    .judge-title {
        font-size: 1.5rem;
        font-weight: 800;
        color: #f3e8ff;
        margin: 0;
        letter-spacing: -0.01em;
    }

    .verdict-box {
        background-color: rgba(10, 11, 18, 0.4);
        border: 1px solid rgba(139, 92, 246, 0.2);
        border-radius: 8px;
        padding: 18px 24px;
        margin-bottom: 24px;
        font-family: 'Fira Code', monospace;
    }

    .verdict-label {
        font-size: 0.78rem;
        color: #d8b4fe;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 6px;
    }

    .verdict-value {
        font-size: 1.25rem;
        font-weight: 700;
        color: #ffffff;
    }

    .judge-summary-label {
        font-weight: 700;
        color: #c084fc;
        font-family: 'Outfit', sans-serif;
        margin-bottom: 10px;
        font-size: 1.05rem;
        letter-spacing: 0.02em;
    }

    .judge-rationale-box {
        line-height: 1.65;
        color: #cbd5e1;
        font-size: 0.98rem;
    }

    /* Glassmorphism card styles with ambient backlights */
    .glass-card {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.05) 0%, rgba(255, 255, 255, 0.01) 50%, transparent 100%), rgba(8, 8, 10, 0.6) !important;
        backdrop-filter: blur(16px) !important;
        -webkit-backdrop-filter: blur(16px) !important;
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
        border-radius: 12px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.5), 0 0 15px rgba(255, 255, 255, 0.01) !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .glass-card-a {
        border-left: 4px solid #10b981 !important;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.5), 0 0 25px rgba(16, 185, 129, 0.06) !important;
    }
    .glass-card-a:hover {
        border-color: rgba(16, 185, 129, 0.3) !important;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.6), 0 0 30px rgba(16, 185, 129, 0.12) !important;
        transform: translateY(-2px);
    }

    .glass-card-b {
        border-left: 4px solid #8b5cf6 !important;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.5), 0 0 25px rgba(139, 92, 246, 0.06) !important;
    }
    .glass-card-b:hover {
        border-color: rgba(139, 92, 246, 0.3) !important;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.6), 0 0 30px rgba(139, 92, 246, 0.12) !important;
        transform: translateY(-2px);
    }

    .glass-card-judge {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.05) 0%, rgba(255, 255, 255, 0.01) 50%, transparent 100%), rgba(8, 8, 10, 0.6) !important;
        backdrop-filter: blur(16px) !important;
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
        border-left: 4px solid #8b5cf6 !important;
        border-radius: 12px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.5), 0 0 25px rgba(139, 92, 246, 0.06) !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .glass-card-judge:hover {
        border-color: rgba(139, 92, 246, 0.3) !important;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.6), 0 0 30px rgba(139, 92, 246, 0.12) !important;
        transform: translateY(-2px);
    }

    /* Conversational Chat Bubbles with Diagonal Shine & Glowing Backlights */
    .chat-bubble-a {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.08) 0%, rgba(255, 255, 255, 0.01) 50%, transparent 100%), rgba(8, 8, 10, 0.5) !important;
        backdrop-filter: blur(16px) !important;
        -webkit-backdrop-filter: blur(16px) !important;
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
        border-left: 4px solid #10b981 !important;
        border-radius: 16px 16px 16px 4px !important;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.5), 0 0 30px rgba(16, 185, 129, 0.07) !important;
        width: 82% !important;
        padding: 20px 24px !important;
        margin: 12px auto 12px 0 !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .chat-bubble-a:hover {
        border-color: rgba(16, 185, 129, 0.3) !important;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.6), 0 0 35px rgba(16, 185, 129, 0.15) !important;
        transform: translateY(-2px);
    }

    .chat-bubble-b {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.08) 0%, rgba(255, 255, 255, 0.01) 50%, transparent 100%), rgba(8, 8, 10, 0.5) !important;
        backdrop-filter: blur(16px) !important;
        -webkit-backdrop-filter: blur(16px) !important;
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
        border-right: 4px solid #8b5cf6 !important;
        border-radius: 16px 16px 4px 16px !important;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.5), 0 0 30px rgba(139, 92, 246, 0.07) !important;
        width: 82% !important;
        padding: 20px 24px !important;
        margin: 12px 0 12px auto !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .chat-bubble-b:hover {
        border-color: rgba(139, 92, 246, 0.3) !important;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.6), 0 0 35px rgba(139, 92, 246, 0.15) !important;
        transform: translateY(-2px);
    }

    .badge-grounded {
        background-color: rgba(16, 185, 129, 0.06);
        color: #34d399;
        border: 1px solid rgba(52, 211, 153, 0.18);
        padding: 3px 12px;
        border-radius: 50px;
        font-size: 0.75rem;
        font-weight: 600;
        font-family: 'Fira Code', monospace;
    }

    .badge-unverified {
        background-color: rgba(239, 68, 68, 0.06);
        color: #f87171;
        border: 1px solid rgba(248, 113, 113, 0.18);
        padding: 3px 12px;
        border-radius: 50px;
        font-size: 0.75rem;
        font-weight: 600;
        font-family: 'Fira Code', monospace;
    }

    .badge-retry {
        background-color: rgba(245, 158, 11, 0.06);
        color: #fbbf24;
        border: 1px solid rgba(251, 191, 36, 0.18);
        padding: 3px 12px;
        border-radius: 50px;
        font-size: 0.75rem;
        font-weight: 600;
        font-family: 'Fira Code', monospace;
        margin-left: 10px;
    }
</style>
""", unsafe_allow_html=True)



# Sidebar Configurations
with st.sidebar:
    st.header("Arena Config Panel")
    topic = st.text_area(
        "Scientific Topic under dispute",
        value="What is the target region(s) of olfactory crypt neurons?",
        height=100,
    )
    rounds = st.slider("Max rounds", min_value=2, max_value=4, value=2)
    
    st.subheader("Simulated Run options")
    use_mock = st.checkbox(
        "Demo/Sandbox Mode",
        value=True,
        help="Simulates the entire research, adversarial debate, and judging loop with pre-cached high-fidelity literature. Bypass Google API rate limits (RESOURCE_EXHAUSTED) instantly.",
    )
    
    enable_adk_planner = st.checkbox(
        "🧭 Enable Google ADK Research Planner",
        value=True,
        help="Uses a Google ADK agent to check evidence coverage "
             "after opening arguments and fetch more literature if "
             "thin. Adds one extra Gemini API call per debate."
    )
    
    run_clicked = st.button("Initialize Stance & Run Debate", type="primary")

    # Reset button
    if st.button("Reset Arena"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# Initialize Session State Variables
if "debate_started" not in st.session_state:
    st.session_state.debate_started = False
    st.session_state.debate_paused = False
    st.session_state.debate_finished = False
    st.session_state.debate_status = "Status: Idle - Awaiting initialization"
    st.session_state.debate_score = 50
    st.session_state.debate_turns = []
    st.session_state.debate_hypotheses = {}
    st.session_state.debate_verdict = ""
    st.session_state.debate_challenge = ""
    st.session_state.adk_planner_decision = None

# Sleek Top Header Bar Function
header_placeholder = st.empty()

def update_header(status_text, is_concluded=False):
    status_class = "pulse-dot-demo" if use_mock else "pulse-dot-active"
    if is_concluded:
        dot_html = '<div class="pulse-dot-demo" style="background-color: #10b981; box-shadow: none; animation: none; width: 6px; height: 6px; border-radius: 50%; margin-right: 6px; display: inline-block;"></div>'
    else:
        dot_html = f'<div class="{status_class}" style="width: 6px; height: 6px; border-radius: 50%; margin-right: 6px; display: inline-block;"></div>'
        
    topic_str = topic if 'topic' in globals() else "Olfactory crypt neurons"
    if len(topic_str) > 48:
        topic_clipped = topic_str[:45] + "..."
    else:
        topic_clipped = topic_str

    header_placeholder.markdown(f"""
    <div style="display: flex; justify-content: space-between; align-items: center; background: rgba(8, 8, 10, 0.4); border: 1px solid rgba(255, 255, 255, 0.05); padding: 8px 16px; border-radius: 8px; font-family: 'Fira Code', monospace; font-size: 0.78rem; margin-bottom: 25px; letter-spacing: 0.05em; color: #94a3b8; width: 100%;">
        <div style="display: flex; align-items: center; gap: 10px; overflow: hidden; white-space: nowrap; text-overflow: ellipsis; max-width: 75%;">
            <span style="color: #10b981; font-weight: 700;">◆ DEBATE ARENA</span>
            <span style="color: rgba(255,255,255,0.08);">|</span>
            <span style="color: #cbd5e1; text-transform: uppercase;">Topic: "{topic_clipped}"</span>
        </div>
        <div style="display: flex; align-items: center; gap: 4px; shrink: 0;">
            {dot_html}
            <span style="color: #e2e8f0; font-weight: 600; text-transform: uppercase; font-size: 0.72rem;">{status_text}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Initial draw of header
update_header(st.session_state.debate_status)

# Tug of War dial helper
def render_tug_of_war():
    st.markdown(f"""
    <div style="background: rgba(8, 8, 10, 0.6); border: 1px solid rgba(255, 255, 255, 0.06); backdrop-filter: blur(16px); -webkit-backdrop-filter: blur(16px); border-radius: 12px; padding: 16px 20px; margin-bottom: 30px; box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.5);">
        <div style="display: flex; justify-content: space-between; font-family: 'Outfit', sans-serif; font-size: 0.8rem; font-weight: 700; letter-spacing: 0.08em; color: #64748b; margin-bottom: 10px;">
            <span style="color: #34d399;">◄ ADVOCATE A CREDIBILITY INDEX</span>
            <span style="color: #a78bfa;">ADVOCATE B CREDIBILITY INDEX ►</span>
        </div>
        <div style="background-color: #111116; border: 1px solid rgba(255,255,255,0.02); border-radius: 10px; height: 8px; width: 100%; position: relative; overflow: visible;">
            <!-- Left color track (Emerald/Purple Gradient) -->
            <div style="background: linear-gradient(to right, #10b981, #8b5cf6); width: {st.session_state.debate_score}%; height: 100%; border-radius: 10px 0 0 10px; position: absolute; left: 0; transition: width 0.8s ease-in-out;"></div>
            <!-- Pointer indicator -->
            <div style="background-color: #ffffff; width: 4px; height: 16px; border-radius: 2px; position: absolute; left: {st.session_state.debate_score}%; top: -4px; transform: translateX(-2px); box-shadow: 0 0 8px #ffffff; z-index: 5; transition: left 0.8s ease-in-out;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def update_tug_of_war(turn_data):
    speaker = turn_data["speaker"]
    grounded = turn_data["grounding"]["grounded"]
    retries = turn_data.get("retries", 0)
    current_score = st.session_state.debate_score
    
    if "A" in speaker:
        if grounded:
            current_score -= 6 if retries > 0 else 12
        else:
            current_score += 15
    else:
        if grounded:
            current_score += 6 if retries > 0 else 12
        else:
            current_score -= 15
            
    st.session_state.debate_score = max(5, min(95, current_score))

# Stance columns helper
def render_stances():
    if not st.session_state.debate_hypotheses:
        return
        
    col1, col2 = st.columns(2)
    
    with col1:
        if "Advocate A" in st.session_state.debate_hypotheses:
            data = st.session_state.debate_hypotheses["Advocate A"]
            st.markdown(f"""
            <div class="glass-card glass-card-a" style="padding: 24px;">
                <div style="display: inline-block; background: rgba(16, 185, 129, 0.08); color: #34d399; font-size: 0.72rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; padding: 4px 10px; border-radius: 6px; border: 1px solid rgba(16, 185, 129, 0.2); margin-bottom: 14px;">Advocate A Stance</div>
                <h3 style="margin: 0 0 10px 0; font-family: 'Outfit', sans-serif; font-size: 1.4rem; color: #ffffff;">Hypothesis A</h3>
                <p style="font-size: 1.1rem; font-weight: 500; color: #f1f5f9; line-height: 1.45; margin: 0 0 18px 0;">"{data['hypothesis']}"</p>
                <div style="border-top: 1px solid rgba(255,255,255,0.05); padding-top: 14px;">
                    <div style="font-size: 0.75rem; font-weight: 700; letter-spacing: 0.05em; color: #64748b; text-transform: uppercase; margin-bottom: 6px;">Initial Research Base</div>
                    <p style="font-size: 0.92rem; color: #94a3b8; line-height: 1.55; margin: 0;">{data['rationale']}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
    with col2:
        if "Advocate B" in st.session_state.debate_hypotheses:
            data = st.session_state.debate_hypotheses["Advocate B"]
            st.markdown(f"""
            <div class="glass-card glass-card-b" style="padding: 24px;">
                <div style="display: inline-block; background: rgba(139, 92, 246, 0.08); color: #a78bfa; font-size: 0.72rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; padding: 4px 10px; border-radius: 6px; border: 1px solid rgba(139, 92, 246, 0.2); margin-bottom: 14px;">Advocate B Stance (Adversarial)</div>
                <h3 style="margin: 0 0 10px 0; font-family: 'Outfit', sans-serif; font-size: 1.4rem; color: #ffffff;">Hypothesis B</h3>
                <p style="font-size: 1.1rem; font-weight: 500; color: #f1f5f9; line-height: 1.45; margin: 0 0 18px 0;">"{data['hypothesis']}"</p>
                <div style="border-top: 1px solid rgba(255,255,255,0.05); padding-top: 14px;">
                    <div style="font-size: 0.75rem; font-weight: 700; letter-spacing: 0.05em; color: #64748b; text-transform: uppercase; margin-bottom: 6px;">Initial Research Base</div>
                    <p style="font-size: 0.92rem; color: #94a3b8; line-height: 1.55; margin: 0;">{data['rationale']}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)

# Render speech turns helper
def render_adk_planner():
    decision = st.session_state.get("adk_planner_decision")
    if not decision:
        return
    
    need_more = decision.get("need_more_research", False)
    reasoning = decision.get("reasoning", "")
    query_used = decision.get("query_used")
    
    badge_html = (
        f'<span class="badge-retry" style="margin-left: 0;">🔍 Additional Search: {query_used}</span>'
        if need_more and query_used
        else '<span class="badge-grounded">Coverage sufficient ✅</span>'
    )
    
    status_msg = "Fetched more evidence to address thin coverage." if need_more else "No additional research required."
    
    st.markdown(f"""
    <div class="glass-card" style="padding: 22px; margin: 15px 0 25px 0; border-left: 4px solid #fbbf24 !important; box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.5), 0 0 25px rgba(251, 191, 36, 0.06) !important;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; font-family: 'Outfit', sans-serif;">
            <div style="font-weight: 700; font-size: 1.1rem; color: #fbbf24; letter-spacing: 0.02em; text-transform: uppercase;">🧭 Google ADK Research Planner</div>
            <div>{badge_html}</div>
        </div>
        <p style="font-size: 1.0rem; color: #f1f5f9; line-height: 1.5; margin: 0 0 10px 0;">{reasoning}</p>
        <div style="font-size: 0.85rem; color: #94a3b8; font-style: italic;">{status_msg}</div>
    </div>
    """, unsafe_allow_html=True)

def render_turns(stream_last=False):
    for i, turn in enumerate(st.session_state.debate_turns):
        is_adv_a = turn["speaker"] == "Advocate A"
        speaker_name = "Advocate A" if is_adv_a else "Advocate B"
        speaker_color = "#34d399" if is_adv_a else "#a78bfa"
        
        grounded = turn["grounding"]["grounded"]
        retries = turn.get("retries", 0)
        
        badge_html = '<span class="badge-grounded">✅ Grounded</span>' if grounded else '<span class="badge-unverified">⚠️ Unverified</span>'
        retry_html = f'<span class="badge-retry">🔄 Self-Corrected ({retries} retries)</span>' if retries > 0 else ''
        
        glass_class = "chat-bubble-a" if is_adv_a else "chat-bubble-b"
        
        if stream_last and i == len(st.session_state.debate_turns) - 1:
            with st.container():
                st.markdown(f"""
                <div class="{glass_class}">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; font-family: 'Outfit', sans-serif;">
                        <div style="font-weight: 700; font-size: 1.05rem; color: {speaker_color}; letter-spacing: 0.02em; text-transform: uppercase;">{speaker_name}</div>
                        <div>{badge_html}{retry_html}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Dynamic typewriter word generator
                def stream_words():
                    for word in turn["text"].split():
                        yield word + " "
                        time.sleep(0.03)
                st.write_stream(stream_words)
        else:
            st.markdown(f"""
            <div class="{glass_class}">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; font-family: 'Outfit', sans-serif;">
                    <div style="font-weight: 700; font-size: 1.05rem; color: {speaker_color}; letter-spacing: 0.02em; text-transform: uppercase;">{speaker_name}</div>
                    <div>{badge_html}{retry_html}</div>
                </div>
                <div style="line-height: 1.6; font-size: 1.0rem; color: #f1f5f9;">{turn['text']}</div>
            </div>
            """, unsafe_allow_html=True)
            
            if not grounded:
                with st.expander("Grounding audit details"):
                    st.info(turn["grounding"]["warning"])
                    
        if i == 1 and st.session_state.get("adk_planner_decision"):
            render_adk_planner()

# Render Judge Verdict helper
def render_verdict():
    if not st.session_state.debate_verdict:
        return
        
    verdict_lines = st.session_state.debate_verdict.split("\n\n")
    verdict_val = "Not resolved"
    summary_val = ""
    rationale_val = ""
    
    for line in verdict_lines:
        if line.startswith("**Verdict:**"):
            verdict_val = line.replace("**Verdict:**", "").strip()
        elif line.startswith("**Summary:**"):
            summary_val = line.replace("**Summary:**", "").strip()
        elif line.startswith("**Rationale:**"):
            rationale_val = line.replace("**Rationale:**", "").strip()

    st.markdown(f"""
    <div class="glass-card glass-card-judge" style="padding: 26px; margin-bottom: 40px;">
        <div class="judge-header">
            <div class="judge-icon">⚖️</div>
            <div class="judge-title">SCIENTIFIC DECISION TELEMETRY SUMMARY</div>
        </div>
        <div class="verdict-box">
            <div class="verdict-label">Official Deduction Declared</div>
            <div class="verdict-value">{verdict_val}</div>
        </div>
        <div style="margin-bottom: 20px;">
            <div class="judge-summary-label">Key Disagreement Mapping</div>
            <div style="line-height: 1.5; color: #e5e7eb;">{summary_val}</div>
        </div>
        <div class="judge-rationale-box">
            <div class="judge-summary-label">Deduction & Rationale</div>
            <p>{rationale_val}</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Render Mermaid logic connections web helper
@st.dialog("Logic & Evidence Connections Web", width="large")
def open_logic_dialog(mermaid_code):
    st.components.v1.html(f"""
    <div style="display: block; width: 100%; height: 100%; background-color: #090a10; padding: 20px; border-radius: 12px; border: 1px solid #1f2437; overflow: auto; box-sizing: border-box;">
        <pre class="mermaid" style="background-color: transparent; margin: 0; padding: 0; width: 100%; overflow: auto;">
{mermaid_code}
        </pre>
    </div>
    <script type="module">
      import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
      mermaid.initialize({{ startOnLoad: true, theme: 'dark', flowchart: {{ useMaxWidth: false }} }});
      mermaid.contentLoaded();
    </script>
    """, height=650, scrolling=True)

# Render Mermaid logic connections web helper
def render_mermaid_web():
    if not st.session_state.debate_hypotheses:
        return
        
    def clean_mermaid_label(text: str) -> str:
        return text.replace('"', "'").replace('[', '').replace(']', '').replace('(', '').replace(')', '').replace(':', ' ').replace(';', ' ')

    topic_clean = clean_mermaid_label(topic[:35])
    mermaid_code = "%%{init: {'theme': 'dark', 'themeVariables': { 'fontSize': '14px', 'fontFamily': 'Plus Jakarta Sans', 'lineColor': '#64748b' }}}%%\n"
    mermaid_code += "graph LR\n"
    mermaid_code += f'    Topic["Topic: {topic_clean}..."]\n'
    
    if "Advocate A" in st.session_state.debate_hypotheses:
        mermaid_code += '    HypA["Hypothesis A (Conserved)"]\n'
        mermaid_code += '    Topic --> HypA\n'
        for i, p in enumerate(st.session_state.debate_hypotheses["Advocate A"]["papers"], 1):
            title_escaped = clean_mermaid_label(p["title"][:22])
            mermaid_code += f'    PaperA{i}["Paper {i}: {title_escaped}..."]\n'
            mermaid_code += f'    PaperA{i} --> HypA\n'
            
    if "Advocate B" in st.session_state.debate_hypotheses:
        mermaid_code += '    HypB["Hypothesis B (Variable)"]\n'
        mermaid_code += '    Topic --> HypB\n'
        for i, p in enumerate(st.session_state.debate_hypotheses["Advocate B"]["papers"], 1):
            title_escaped = clean_mermaid_label(p["title"][:22])
            mermaid_code += f'    PaperB{i}["Paper {i}: {title_escaped}..."]\n'
            mermaid_code += f'    PaperB{i} --> HypB\n'
            
    if len(st.session_state.debate_turns) >= 3:
        mermaid_code += '    HypB --> HypA\n'
    if len(st.session_state.debate_turns) >= 4:
        mermaid_code += '    HypA --> HypB\n'
        
    # Add CSS styling mapping to design system
    mermaid_code += '    classDef topic fill:#0f172a,stroke:#334155,stroke-width:1.5px,color:#f8fafc;\n'
    mermaid_code += '    classDef advA fill:#0f1322,stroke:#0ea5e9,stroke-width:1.5px,color:#38bdf8;\n'
    mermaid_code += '    classDef advB fill:#0f1322,stroke:#d97706,stroke-width:1.5px,color:#fbbf24;\n'
    mermaid_code += '    classDef paper fill:#090a10,stroke:#1f2437,stroke-width:1px,color:#94a3b8;\n'
    
    mermaid_code += '    class Topic topic;\n'
    mermaid_code += '    class HypA advA;\n'
    mermaid_code += '    class HypB advB;\n'
    
    if "Advocate A" in st.session_state.debate_hypotheses:
        for i in range(1, len(st.session_state.debate_hypotheses["Advocate A"]["papers"]) + 1):
            mermaid_code += f'    class PaperA{i} paper;\n'
    if "Advocate B" in st.session_state.debate_hypotheses:
        for i in range(1, len(st.session_state.debate_hypotheses["Advocate B"]["papers"]) + 1):
            mermaid_code += f'    class PaperB{i} paper;\n'
        
    # Renders a neat centered button to open overlay dialog
    st.markdown("<div style='margin-bottom: 25px;'>", unsafe_allow_html=True)
    if st.button("🕸️ Inspect Evidence Logic Graph Overlay", use_container_width=True):
        open_logic_dialog(mermaid_code)
    st.markdown("</div>", unsafe_allow_html=True)


# Render complete literature tab bibliography
def render_bibliography():
    st.markdown("### 📚 Retrieved Literature Database")
    
    # Since both advocates currently query the same topic simultaneously, their paper base is shared.
    # We display a unified bibliography here instead of redundant tabs.
    if "Advocate A" in st.session_state.debate_hypotheses:
        papers = st.session_state.debate_hypotheses["Advocate A"]["papers"]
        for i, p in enumerate(papers, 1):
            url = p.get("url", "#")
            st.markdown(f"""
            <div class="glass-card" style="padding: 20px; margin-bottom: 16px; border-left: 1px solid rgba(139, 92, 246, 0.25) !important;">
                <div style="font-family: 'Outfit', sans-serif; font-size: 1.15rem; font-weight: 700; line-height: 1.35; margin-bottom: 6px;">
                    [{i}] {p['title']} 
                    <a href="{url}" target="_blank" style="color: #a78bfa; text-decoration: none; font-size: 0.95rem; margin-left: 8px;">🔗</a>
                </div>
                <div style="font-family: 'Fira Code', monospace; font-size: 0.8rem; color: #64748b; margin-bottom: 12px;">Year: {p['year']} | Citations: {p['citationCount']} | Source: {p.get('source', 'Unknown')}</div>
                <div style="border-left: 2px solid #334155; padding-left: 12px; margin-top: 10px; color: #94a3b8; font-style: italic; font-size: 0.9rem; line-height: 1.5;">
                    <strong>Abstract excerpt:</strong> {p['abstract'][:400]}...
                </div>
            </div>
            """, unsafe_allow_html=True)

# Main trigger execution
if run_clicked and not st.session_state.debate_started:
    st.session_state.debate_started = True
    st.session_state.debate_paused = False
    st.session_state.debate_finished = False
    st.session_state.debate_score = 50
    st.session_state.debate_turns = []
    st.session_state.debate_hypotheses = {}
    st.session_state.debate_verdict = ""
    st.session_state.adk_planner_decision = None
    st.session_state.debate_stream = run_debate_stream(topic, rounds=rounds, use_mock=use_mock, use_adk_planner=enable_adk_planner)
    st.rerun()

# Render Static Layout
if st.session_state.debate_started:
    render_tug_of_war()
    render_stances()

# Generator Consuming Loop
if st.session_state.debate_started and not st.session_state.debate_paused and not st.session_state.debate_finished:
    # Check if we are resuming from a saved next event
    initial_event = None
    if "next_event" in st.session_state and st.session_state.next_event is not None:
        initial_event = st.session_state.next_event
        st.session_state.next_event = None

    stream = st.session_state.debate_stream
    event = initial_event
    
    try:
        while True:
            if event is None:
                event = next(stream)
            
            # 1. Handle Status Ticker
            if event["type"] == "status":
                st.session_state.debate_status = event["message"]
                update_header(st.session_state.debate_status)
                
            # 2. Handle Hypothesis Proposals
            elif event["type"] == "hypothesis":
                speaker = event["speaker"]
                st.session_state.debate_hypotheses[speaker] = {
                    "hypothesis": event["hypothesis"],
                    "rationale": event["rationale"],
                    "papers": event["papers"]
                }
                st.rerun()  # Forces columns to render stance data immediately
                
            # 3. Handle Grounding Retries
            elif event["type"] == "grounding_retry":
                st.toast(f"⚠️ {event['speaker']} citation retry #{event['retry_num']}: {event['warning'][:120]}...", icon="⚠️")
                time.sleep(0.8) # Allow brief moment to read status before self-correction
                
            # 4. Handle Speech turns
            elif event["type"] == "turn":
                turn_data = event["data"]
                st.session_state.debate_turns.append(turn_data)
                update_tug_of_war(turn_data)
                st.rerun()  # Forces typewriter stream on the new turn
                
            # 5. Handle pause coroutines
            elif event["type"] == "pause":
                st.session_state.debate_paused = True
                st.session_state.debate_status = "Status: Intermission - Inject challenge directive"
                st.rerun()
                
            # 5b. Handle ADK planner decision
            elif event["type"] == "adk_planner":
                st.session_state.adk_planner_decision = event["data"]
                st.rerun()
                
            # 6. Handle Verdict Summaries
            elif event["type"] == "verdict":
                st.session_state.debate_verdict = event["val"]
                st.rerun()
                
            event = None
    except StopIteration:
        st.session_state.debate_finished = True
        st.session_state.debate_status = "Status: Debate Concluded"
        update_header(st.session_state.debate_status, is_concluded=True)
        st.rerun()

# Renders turns statically if not actively streaming them
if len(st.session_state.debate_turns) > 0:
    # If the last turn was just appended, stream it once, then let it draw statically on next redraws
    # We can detect if we are actively streaming by checking generator state
    is_actively_streaming = st.session_state.debate_started and not st.session_state.debate_paused and not st.session_state.debate_finished
    render_turns(stream_last=is_actively_streaming)

# Render Intermission UI if paused
if st.session_state.debate_paused:
    st.markdown("""
    <div style="background-color: #1e1b15; border: 1px solid #eab308; border-radius: 12px; padding: 22px; margin: 30px 0; box-shadow: 0 4px 20px rgba(234, 179, 8, 0.05);">
        <h3 style="color: #eab308; margin-top: 0; font-family: 'Outfit', sans-serif; font-size: 1.3rem;">💡 INTERMISSION: DIRECT CROSS-EXAMINATION</h3>
        <p style="font-size: 0.95rem; color: #cbd5e1; line-height: 1.5; margin: 0;">
            The Judge has paused the debate after the opening arguments. As the Judge's assistant, 
            you can now **inject a cross-examination challenge**. Select a pre-defined directive 
            or type your own to focus the advocates' rebuttals.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    challenge_options = [
        "Challenge Advocate A on Teleost Fish translation limits (animal vs human)",
        "Challenge Advocate B on Stochastic wiring guidance molecule variance",
        "Direct Advocates to address clinical relevance and medical translation",
        "Custom Directive (type below)"
    ]
    
    selected_option = st.radio("Select Cross-Examination Directive:", challenge_options)
    
    custom_input = ""
    if selected_option == "Custom Directive (type below)":
        custom_input = st.text_input("Enter your custom scientific directive/challenge:", placeholder="e.g. Projections in mammalian aging cohorts...")
        
    resume_btn = st.button("Inject Directive & Resume Debate", type="primary")
    
    if resume_btn:
        final_challenge = custom_input if selected_option == "Custom Directive (type below)" else selected_option
        st.session_state.debate_challenge = final_challenge
        
        # Resume generator by sending the user's final challenge!
        stream = st.session_state.debate_stream
        try:
            next_event = stream.send(final_challenge)
            st.session_state.debate_paused = False
            st.session_state.next_event = next_event
            st.rerun()
        except StopIteration:
            st.session_state.debate_paused = False
            st.rerun()

# Render Final Verdict summary
if st.session_state.debate_verdict:
    render_verdict()

# Render Mermaid Logic graph and bibliography database
if st.session_state.debate_started:
    render_mermaid_web()
    render_bibliography()

# Footnote captions
st.divider()
st.caption(
    "Capstone Project Arena. Powered by Google Gemini & SQLite Cache. "
    "Features live typewriter streams, programmatic grounding feedback loops, "
    "interactive coroutine cross-examinations, and dynamic logic web mapping."
)