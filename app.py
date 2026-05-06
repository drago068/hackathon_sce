import streamlit as st
import os
from dotenv import load_dotenv

from pdf_processor import extract_text_from_pdf
from rag_engine import RAGEngine
from llm_service import LLMService

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Vernacular AI Tutor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS — ChatGPT-dark + tab styling
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

*, html, body { font-family: 'Inter', sans-serif; box-sizing: border-box; }

/* ══ App ══ */
.stApp { background-color: #f5f7fa; min-height: 100vh; }
header[data-testid="stHeader"] { background: rgba(255,255,255,0.85) !important; backdrop-filter: blur(10px); }
footer { display: none !important; }

/* ══ SIDEBAR TOGGLE — always visible ══ */
button[data-testid="stSidebarCollapsedControl"],
button[data-testid="baseButton-headerNoPadding"],
[data-testid="collapsedControl"] {
    display: flex !important;
    visibility: visible !important;
    opacity: 1 !important;
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 10px !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08) !important;
    color: #334155 !important;
    width: 40px !important;
    height: 40px !important;
    position: fixed !important;
    left: 12px !important;
    top: 14px !important;
    z-index: 999999 !important;
    align-items: center !important;
    justify-content: center !important;
    transition: all 0.2s ease !important;
    cursor: pointer !important;
}
button[data-testid="stSidebarCollapsedControl"]:hover,
button[data-testid="baseButton-headerNoPadding"]:hover,
[data-testid="collapsedControl"]:hover {
    background: #f1f5f9 !important;
    box-shadow: 0 4px 12px rgba(0,0,0,0.12) !important;
    transform: scale(1.05) !important;
}
button[data-testid="stSidebarCollapsedControl"] svg,
button[data-testid="baseButton-headerNoPadding"] svg,
[data-testid="collapsedControl"] svg {
    fill: #334155 !important;
    stroke: #334155 !important;
    color: #334155 !important;
    width: 20px !important;
    height: 20px !important;
}

/* ══ GLOBAL TEXT ══ */
body, .stApp, .main, .block-container,
div, span, p, li, label, small, em, strong {
    color: #334155;
}
h1, h2, h3, h4, h5, h6 { color: #0f172a !important; }
.stMarkdown, .stMarkdown p, .stMarkdown li,
.stMarkdown span { color: #334155 !important; }
.stMarkdown h1, .stMarkdown h2,
.stMarkdown h3, .stMarkdown h4 { color: #0f172a !important; }
.stCaption, .stCaption p { color: #94a3b8 !important; }

/* ══ SIDEBAR ══ */
section[data-testid="stSidebar"] {
    background: #ffffff !important;
    border-right: 1px solid #e2e8f0;
    box-shadow: 2px 0 12px rgba(0,0,0,0.04);
}
section[data-testid="stSidebar"],
section[data-testid="stSidebar"] *,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] div,
section[data-testid="stSidebar"] small { color: #334155 !important; }
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 { color: #0f172a !important; }
section[data-testid="stSidebar"] .stCaption,
section[data-testid="stSidebar"] .stCaption p { color: #94a3b8 !important; }

/* ══ FILE UPLOADER ══ */
[data-testid="stFileUploader"],
[data-testid="stFileUploader"] *,
[data-testid="stFileUploaderDropzone"],
[data-testid="stFileUploaderDropzone"] * {
    color: #334155 !important;
    background: #f8fafc !important;
}
[data-testid="stFileUploaderDropzone"] {
    border: 2px dashed #cbd5e1 !important;
    border-radius: 12px !important;
}
[data-testid="stFileUploader"] button,
[data-testid="stFileUploaderDropzone"] button {
    background: #e2e8f0 !important;
    color: #334155 !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 8px !important;
}

/* ══ TEXT INPUTS & TEXTAREAS ══ */
.stTextArea textarea,
.stTextInput input,
div[data-baseweb="textarea"] textarea,
div[data-baseweb="input"] input {
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 12px !important;
    color: #1e293b !important;
    caret-color: #3b82f6;
}
.stTextArea textarea:focus,
.stTextInput input:focus {
    border-color: #3b82f6 !important;
    box-shadow: 0 0 0 3px rgba(59,130,246,0.12) !important;
}
.stTextArea textarea::placeholder,
.stTextInput input::placeholder { color: #94a3b8 !important; }

/* ══ SELECTBOX & DROPDOWN ══ */
div[data-baseweb="select"],
div[data-baseweb="select"] *,
[data-testid="stSelectbox"],
[data-testid="stSelectbox"] * {
    background: #ffffff !important;
    color: #334155 !important;
}
div[data-baseweb="select"] svg { fill: #334155 !important; }
div[data-baseweb="popover"],
div[data-baseweb="popover"] *,
ul[data-baseweb="menu"],
ul[data-baseweb="menu"] li {
    background: #ffffff !important;
    color: #334155 !important;
}
ul[data-baseweb="menu"] li:hover { background: #f1f5f9 !important; }

/* ══ RADIO BUTTONS ══ */
.stRadio label, .stRadio span, .stRadio div,
[data-testid="stRadio"] label,
[data-testid="stRadio"] span { color: #334155 !important; }
.stRadio > div { display: flex !important; gap: 10px !important; flex-direction: row !important; }

/* ══ CHECKBOXES ══ */
.stCheckbox label, .stCheckbox span { color: #334155 !important; }

/* ══ EXPANDER ══ */
[data-testid="stExpander"],
[data-testid="stExpander"] *,
.streamlit-expanderHeader,
.streamlit-expanderHeader * { color: #334155 !important; background: #ffffff !important; }
[data-testid="stExpander"] { border: 1px solid #e2e8f0 !important; border-radius: 10px !important; }
[data-testid="stExpanderDetails"] { background: #ffffff !important; }

/* ══ METRIC BOXES ══ */
[data-testid="metric-container"] {
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 12px !important;
    padding: 14px 18px !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04) !important;
}
[data-testid="metric-container"] * { color: #334155 !important; }
[data-testid="stMetricLabel"] { color: #64748b !important; }
[data-testid="stMetricValue"] { color: #0f172a !important; }
[data-testid="stMetricDelta"] { color: #16a34a !important; }

/* ══ TABS ══ */
.stTabs [data-baseweb="tab-list"] {
    background: #ffffff !important;
    border-radius: 14px !important;
    padding: 4px !important;
    gap: 4px !important;
    border: 1px solid #e2e8f0 !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04) !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    border-radius: 10px !important;
    color: #94a3b8 !important;
    font-weight: 500 !important;
    font-size: 0.95rem !important;
    padding: 10px 28px !important;
    border: none !important;
}
.stTabs [aria-selected="true"] {
    background: #f1f5f9 !important;
    color: #0f172a !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.08) !important;
}
.stTabs [data-baseweb="tab-panel"] { padding-top: 20px !important; }

/* ══ BUTTONS ══ */
.stButton > button {
    background: linear-gradient(135deg, #10b981, #3b82f6) !important;
    color: #fff !important; border: none !important;
    border-radius: 10px !important; padding: 10px 22px !important;
    font-weight: 600 !important; transition: all 0.25s ease !important;
}
.stButton > button:hover {
    opacity: 0.9 !important; transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(16,185,129,0.25) !important;
}
.btn-outline .stButton > button {
    background: transparent !important; border: 1px solid #cbd5e1 !important;
    color: #64748b !important; transform: none !important; box-shadow: none !important;
}

/* ══ FORM SUBMIT BUTTON ══ */
[data-testid="stFormSubmitButton"] > button {
    background: linear-gradient(135deg, #10b981, #3b82f6) !important;
    color: #fff !important; border: none !important;
    border-radius: 10px !important; font-weight: 600 !important;
}

/* ══ SPINNER ══ */
.stSpinner > div { border-top-color: #10b981 !important; }
[data-testid="stSpinner"] p,
[data-testid="stSpinner"] span { color: #64748b !important; }

/* ══ ALERTS ══ */
.stAlert { border-radius: 10px !important; }
.stAlert p { color: inherit !important; }

/* ══ DIVIDER / SCROLLBAR ══ */
hr { border-color: #e2e8f0 !important; }
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 4px; }

/* ══ CHAT BUBBLES ══ */
.msg-row { display: flex; gap: 14px; margin: 18px 0; align-items: flex-start; }
.msg-avatar {
    width: 36px; height: 36px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 17px; flex-shrink: 0; margin-top: 2px;
}
.avatar-user { background: #3b82f6; }
.avatar-ai   { background: #10b981; }
.msg-bubble  { flex: 1; padding: 14px 18px; border-radius: 16px; font-size: 0.96rem; line-height: 1.75; max-width: 100%; word-wrap: break-word; }
.bubble-user { background: #eff6ff; color: #1e293b !important; border-radius: 16px 16px 4px 16px; border: 1px solid #dbeafe; }
.bubble-ai   { background: #ffffff; color: #334155 !important; border-radius: 16px 16px 16px 4px; border: 1px solid #e2e8f0; box-shadow: 0 1px 4px rgba(0,0,0,0.04); }
.msg-label   { font-size: 0.72rem; font-weight: 700; letter-spacing: 0.05em; margin-bottom: 6px; text-transform: uppercase; }
.label-user  { color: #3b82f6 !important; }
.label-ai    { color: #10b981 !important; }

/* ── Language badge ── */
.lang-badge { display: inline-block; font-size: 0.68rem; font-weight: 700; padding: 2px 10px; border-radius: 20px; margin-bottom: 8px; }
.badge-english  { background: #dbeafe; color: #1d4ed8 !important; }
.badge-tanglish { background: #ede9fe; color: #7c3aed !important; }

/* ── Verdict cards ── */
.verdict-card { border-radius: 10px; padding: 12px 16px; margin: 8px 0; font-size: 0.92rem; line-height: 1.6; }
.verdict-correct { background: #f0fdf4; border: 1px solid #86efac; color: #166534 !important; }
.verdict-partial { background: #fffbeb; border: 1px solid #fcd34d; color: #92400e !important; }
.verdict-wrong   { background: #fef2f2; border: 1px solid #fca5a5; color: #991b1b !important; }

/* ── Pending-Q banner ── */
.pending-q { border-radius: 14px; padding: 16px 20px; margin: 10px 0; background: #ffffff; box-shadow: 0 1px 4px rgba(0,0,0,0.06); }

/* ══ QUIZ CARDS ══ */
.quiz-card { background: #ffffff; border: 1px solid #e2e8f0; border-radius: 16px; padding: 22px 24px; margin: 14px 0; box-shadow: 0 1px 4px rgba(0,0,0,0.04); }
.chip { display: inline-block; font-size: 0.75rem; font-weight: 700; padding: 4px 14px; border-radius: 20px; margin-bottom: 12px; text-transform: uppercase; letter-spacing: 0.05em; }
.chip-easy       { background: #dcfce7; color: #166534 !important; }
.chip-medium     { background: #fef3c7; color: #92400e !important; }
.chip-conceptual { background: #e0e7ff; color: #3730a3 !important; }
.chip-followup   { background: #cffafe; color: #155e75 !important; }

/* ── History items ── */
.hist-item { background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 12px 18px; margin: 8px 0; display: flex; align-items: center; gap: 14px; box-shadow: 0 1px 3px rgba(0,0,0,0.04); }

/* ── Explanation box ── */
.explain-box { background: #f5f3ff; border-left: 4px solid #7c3aed; border-radius: 0 12px 12px 0; padding: 16px 20px; font-size: 1rem; color: #4c1d95 !important; line-height: 1.85; }

/* ── Welcome screen ── */
.welcome-box { text-align: center; padding: 56px 24px; }
.welcome-box h1 { color: #0f172a !important; font-size: 2.2rem; margin-bottom: 10px; }
.welcome-box p  { color: #64748b !important; max-width: 480px; margin: 0 auto 20px auto; }

</style>
""", unsafe_allow_html=True)



# ─────────────────────────────────────────────────────────────────────────────
# Session State
# ─────────────────────────────────────────────────────────────────────────────
def init_state():
    defaults = {
        "rag_engine":         None,
        "llm_service":        None,
        "pdf_processed":      False,
        "language":           "English",      # "English" | "Tanglish"
        # ── Chat tab ──
        "chat_history":       [],
        "chat_mode":          "chat",          # chat | awaiting_answer | awaiting_followup
        "chat_pending_q":     None,
        "chat_pending_diff":  None,
        "chat_pending_ctx":   None,
        "chat_pending_weak":  None,
        # ── Quiz tab ──
        "quiz_phase":         "generate",      # generate | answer | evaluated | teaching | followup | followup_result
        "quiz_questions":     None,
        "quiz_question":      None,
        "quiz_difficulty":    "medium",
        "quiz_context":       None,
        "quiz_eval":          None,
        "quiz_explanation":   None,
        "quiz_followup_q":    None,
        "quiz_followup_res":  None,
        "score":              0,
        "total_asked":        0,
        "history":            [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ── Init services ──────────────────────────────────────────────────────────────
if st.session_state.rag_engine is None:
    st.session_state.rag_engine = RAGEngine()
if st.session_state.llm_service is None:
    st.session_state.llm_service = LLMService()

rag: RAGEngine  = st.session_state.rag_engine
llm: LLMService = st.session_state.llm_service

LANG_MAP = {"English": "english", "Tanglish": "tanglish"}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def add_chat(role, content, lang="", meta=None):
    st.session_state.chat_history.append(
        {"role": role, "content": content, "lang": lang, "meta": meta or {}}
    )


def render_chat_message(msg):
    role = msg["role"]
    content = msg["content"]
    lang = msg.get("lang", "")
    meta = msg.get("meta", {})

    if role == "user":
        avatar_cls, bubble_cls, label_cls, label_text, icon = \
            "avatar-user", "bubble-user", "label-user", "You", "👤"
    else:
        avatar_cls, bubble_cls, label_cls, label_text, icon = \
            "avatar-ai", "bubble-ai", "label-ai", "Tutor AI", "🧠"

    badge = ""
    if lang == "English":
        badge = "<span class='lang-badge badge-english'>EN</span><br>"
    elif lang == "Tanglish":
        badge = "<span class='lang-badge badge-tanglish'>Tanglish</span><br>"

    verdict_html = ""
    if meta.get("verdict"):
        v = meta["verdict"]
        icons = {"Correct": "✅", "Partial": "⚠️", "Wrong": "❌"}
        verdict_html = f"""
<div class='verdict-card verdict-{v.lower()}'>
  <strong>{icons.get(v,'📝')} {v}</strong>
  {f"<br><small>Weak area: {meta['weak_concept']}</small>" if meta.get('weak_concept') and v != 'Correct' else ""}
</div>"""

    st.markdown(f"""
<div class="msg-row">
  <div class="msg-avatar {avatar_cls}">{icon}</div>
  <div class="msg-bubble {bubble_cls}">
    <div class="msg-label {label_cls}">{label_text}</div>
    {badge}{content}{verdict_html}
  </div>
</div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 Vernacular AI Tutor")
    st.markdown("*Your personalised study companion*")
    st.markdown("---")

    st.markdown(f"**LLM:** `{llm.get_backend_info()}`")
    st.markdown("---")

    # ── Language toggle ──────────────────────────────────────────────────────
    st.markdown("### 🌐 Explanation Language")
    lang_choice = st.radio(
        "lang", ["English", "Tanglish"],
        index=["English", "Tanglish"].index(st.session_state.language),
        label_visibility="collapsed"
    )
    if lang_choice != st.session_state.language:
        st.session_state.language = lang_choice
        st.rerun()

    captions = {
        "English":  "Clear explanations in simple English.",
        "Tanglish": "Tamil spoken phonetically in English letters — the WhatsApp way! 😄",
    }
    st.caption(captions[st.session_state.language])
    st.markdown("---")

    # ── PDF Upload ────────────────────────────────────────────────────────────
    st.markdown("### 📄 Upload Study Material")
    uploaded_file = st.file_uploader("PDF", type="pdf", label_visibility="collapsed")

    if uploaded_file and not st.session_state.pdf_processed:
        with st.spinner("🔍 Indexing…"):
            temp = "temp_upload.pdf"
            with open(temp, "wb") as f:
                f.write(uploaded_file.getbuffer())
            text = extract_text_from_pdf(temp)
            if text:
                ok = rag.process_text_and_index(text)
                if ok:
                    st.session_state.pdf_processed = True
                    st.session_state.chat_history  = []
                    st.session_state.score         = 0
                    st.session_state.total_asked   = 0
                    st.session_state.history       = []
                    st.session_state.quiz_phase    = "generate"
                    try: os.remove(temp)
                    except: pass
                    st.success("✅ PDF indexed!")
                    st.rerun()
                else:
                    st.error("❌ Index failed — text-based PDF?")
            else:
                st.error("❌ No text found.")

    if st.session_state.pdf_processed:
        s = rag.get_stats()
        st.info(f"📊 **{s['total_chunks']}** chunks indexed")

    st.markdown("---")

    # ── Score ─────────────────────────────────────────────────────────────────
    if st.session_state.total_asked > 0:
        acc = int(st.session_state.score / st.session_state.total_asked * 100)
        c1, c2 = st.columns(2)
        c1.metric("Score", f"{st.session_state.score}/{st.session_state.total_asked}")
        c2.metric("Accuracy", f"{acc}%")
        st.markdown("---")

    # ── Clear chat ────────────────────────────────────────────────────────────
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.chat_history   = []
        st.session_state.chat_mode      = "chat"
        st.session_state.chat_pending_q = None
        st.rerun()

    st.markdown("---")
    with st.expander("🔑 OpenRouter API Key", expanded=False):
        ak = st.text_input("Key", type="password", value=os.getenv("OPENROUTER_API_KEY", ""))
        if st.button("Apply", key="apply_key"):
            os.environ["OPENROUTER_API_KEY"] = ak
            st.session_state.llm_service = LLMService()
            llm = st.session_state.llm_service
            st.rerun()

    st.markdown("---")
    st.caption("💡 Chat commands: `/quiz` `/easy` `/medium` `/hard` `/explain <topic>` `/help`")


# ─────────────────────────────────────────────────────────────────────────────
# GATE: need PDF first
# ─────────────────────────────────────────────────────────────────────────────
if not st.session_state.pdf_processed:
    st.markdown("""
<div class='welcome-box'>
  <h1>🧠 Vernacular AI Tutor</h1>
  <p>Upload a PDF in the sidebar to start. I'll quiz you, explain weak spots<br>
  and teach concepts in <strong>English</strong> or <strong>Tanglish</strong>!</p>
  <p style='font-size:0.85rem;color:#555;'>
    Tanglish example: "Antha concept romba easy — phone charge maari thaan idhu work aguthu!"
  </p>
</div>""", unsafe_allow_html=True)
    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tab_chat, tab_quiz = st.tabs(["💬  Chat", "🎯  Quiz"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — CHAT (ChatGPT-style)
# ══════════════════════════════════════════════════════════════════════════════
with tab_chat:
    current_lang = st.session_state.language
    lang_key     = LANG_MAP[current_lang]
    mode         = st.session_state.chat_mode

    st.markdown(f"<p style='color:#6b7280;font-size:0.85rem;'>Language: <strong style='color:#ececec'>{current_lang}</strong> — switch in sidebar</p>",
                unsafe_allow_html=True)

    # ── Render conversation ──────────────────────────────────────────────────
    for msg in st.session_state.chat_history:
        render_chat_message(msg)

    # ── Pending question banner ──────────────────────────────────────────────
    if mode in ("awaiting_answer", "awaiting_followup"):
        q    = st.session_state.chat_pending_q
        diff = st.session_state.chat_pending_diff or "Follow-up"
        colours = {"easy": "#22c55e", "medium": "#f59e0b", "conceptual": "#6366f1", "Follow-up": "#0ea5e9"}
        c = colours.get(diff, "#6366f1")
        st.markdown(f"""
<div class='pending-q' style='border:1px solid {c}'>
  <span style='color:{c};font-size:0.75rem;font-weight:700;text-transform:uppercase;'>{diff}</span>
  <p style='color:#ececec;margin:8px 0 0;font-size:1rem;'>❓ {q}</p>
</div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── Input form ───────────────────────────────────────────────────────────
    placeholders = {
        "chat":              f"Ask anything about your PDF… ({current_lang})",
        "awaiting_answer":   "Type your answer to the question above…",
        "awaiting_followup": "Type your follow-up answer…",
    }
    with st.form("chat_form", clear_on_submit=True):
        col_i, col_b = st.columns([6, 1])
        with col_i:
            user_input = st.text_area(
                "msg", placeholder=placeholders[mode],
                height=80, label_visibility="collapsed"
            )
        with col_b:
            submitted = st.form_submit_button("Send ➤", use_container_width=True)

    # ── Process input ────────────────────────────────────────────────────────
    if submitted and user_input.strip():
        ut = user_input.strip()
        add_chat("user", ut, lang=current_lang)

        # ── AWAITING ANSWER ──────────────────────────────────────────────────
        if mode == "awaiting_answer":
            q   = st.session_state.chat_pending_q
            ctx = st.session_state.chat_pending_ctx

            with st.spinner("🤖 Evaluating…"):
                result = llm.evaluate_answer(question=q, user_answer=ut, context=ctx)

            st.session_state.total_asked += 1
            verdict = result["verdict"]
            weak    = result.get("weak_concept", "")
            if result["is_correct"]:
                st.session_state.score += 1

            if result["is_correct"]:
                add_chat("assistant",
                         f"🎉 **Correct!**\n\n{result['explanation']}",
                         lang=current_lang, meta={"verdict": "Correct"})
                st.session_state.chat_mode = "chat"
            else:
                add_chat("assistant",
                         f"**{verdict}**\n\n{result['explanation']}"
                         + (f"\n\n**Weak concept:** `{weak}`\n\nLet me explain in **{current_lang}**…"
                            if weak and weak.lower() != "none" else ""),
                         lang=current_lang,
                         meta={"verdict": verdict, "weak_concept": weak})

                retrieved = rag.retrieve_relevant_chunks(weak or q, k=3)
                best_ctx  = "\n\n".join(retrieved) if retrieved else ctx

                # Stream explanation
                toks = []
                with st.spinner(f"Explaining in {current_lang}…"):
                    for tok in llm.explain_stream(weak or q, best_ctx, lang_key):
                        toks.append(tok)
                add_chat("assistant", "".join(toks), lang=current_lang)

                # Auto follow-up
                with st.spinner("Generating follow-up…"):
                    fq = llm.generate_followup_question(weak or q, ctx)
                st.session_state.chat_pending_q    = fq
                st.session_state.chat_pending_diff = "Follow-up"
                st.session_state.chat_pending_weak = weak
                st.session_state.chat_mode         = "awaiting_followup"
                add_chat("assistant",
                         f"Let's see if that helped! Quick follow-up:\n\n**❓ {fq}**",
                         lang=current_lang)

        # ── AWAITING FOLLOW-UP ───────────────────────────────────────────────
        elif mode == "awaiting_followup":
            fq   = st.session_state.chat_pending_q
            weak = st.session_state.chat_pending_weak or fq

            with st.spinner("Checking…"):
                fu = llm.evaluate_followup(fq, ut, weak)

            if fu["improved"]:
                add_chat("assistant",
                         f"🌟 **Great improvement!** {fu['feedback']}\n\nKeep it up! 💪",
                         lang=current_lang, meta={"verdict": "Correct"})
                st.balloons()
            else:
                add_chat("assistant",
                         f"💪 {fu['feedback']}\n\nNo worries — type `/quiz` to try more questions!",
                         lang=current_lang, meta={"verdict": "Partial"})
            st.session_state.chat_mode = "chat"

        # ── FREE CHAT / COMMANDS ─────────────────────────────────────────────
        else:
            cmd = ut.lower()
            diff = None

            if any(k in cmd for k in ["/quiz", "quiz me", "give me a question", "ask me", "test me"]):
                diff = "medium"
            elif "/easy" in cmd or "easy question" in cmd:
                diff = "easy"
            elif any(k in cmd for k in ["/hard", "/conceptual", "hard question", "conceptual"]):
                diff = "conceptual"
            elif "/medium" in cmd or "medium question" in cmd:
                diff = "medium"

            if diff:
                with st.spinner("📖 Generating question…"):
                    chunk = rag.get_random_chunk()
                    if chunk:
                        qs = llm.generate_questions(chunk)
                        q  = qs.get(diff, qs.get("medium", "What is this passage about?"))

                        st.session_state.chat_pending_q    = q
                        st.session_state.chat_pending_diff = diff
                        st.session_state.chat_pending_ctx  = chunk
                        st.session_state.chat_mode         = "awaiting_answer"

                        icons = {"easy": "🟢", "medium": "🟠", "conceptual": "🔵"}
                        add_chat("assistant",
                                 f"{icons.get(diff,'🎯')} **{diff.upper()} Question:**\n\n❓ {q}\n\n*Type your answer below!*",
                                 lang=current_lang)
                    else:
                        add_chat("assistant", "❌ No content found. Please re-upload your PDF.", lang=current_lang)

            elif cmd.startswith("/explain") or ("explain" in cmd and len(ut.split()) > 1):
                concept = ut.replace("/explain", "").strip()
                if not concept:
                    add_chat("assistant", "Tell me what to explain. Example: `/explain photosynthesis`", lang=current_lang)
                else:
                    ctx = "\n\n".join(rag.retrieve_relevant_chunks(concept, k=3)) or "General knowledge"
                    toks = []
                    with st.spinner(f"Explaining in {current_lang}…"):
                        for tok in llm.explain_stream(concept, ctx, lang_key):
                            toks.append(tok)
                    add_chat("assistant", "".join(toks), lang=current_lang)

            elif any(k in cmd for k in ["tanglish", "switch tanglish"]):
                st.session_state.language = "Tanglish"
                add_chat("assistant",
                         "Switched to **Tanglish** mode! 😎 Ippo enna kaelunga, Tanglish-la explain pannuven!",
                         lang="Tanglish")

            elif any(k in cmd for k in ["english mode", "switch english", "in english"]):
                st.session_state.language = "English"
                add_chat("assistant",
                         "Switched to **English** mode! I'll explain everything clearly from now on. 👍",
                         lang="English")

            elif "/help" in cmd or "what can you do" in cmd:
                add_chat("assistant", f"""Here's what I can do:

| Command | Action |
|---|---|
| `/quiz` or "Quiz me" | Random question from your PDF |
| `/easy` `/medium` `/hard` | Difficulty-specific question |
| `/explain <topic>` | Explain any concept |
| "Switch to Tanglish" | Change to Tanglish mode |
| "Switch to English" | Change to English mode |

**Current language:** {current_lang} · **LLM:** {llm.get_backend_info()}""",
                         lang=current_lang)

            else:
                # RAG-powered answer
                ctx = "\n\n".join(rag.retrieve_relevant_chunks(ut, k=3)) or ""
                prompt = f"""You are a friendly AI tutor.
Student asked: "{ut}"

Relevant study material:
\"\"\"
{ctx[:1500]}
\"\"\"

Answer concisely (under 120 words) based on the material.
{'Write in simple English.' if lang_key == 'english' else 'Write ONLY in Tanglish (Tamil spoken words typed normally in English letters). WARNING: Do NOT use Tamil script (அ, இ). Do NOT spell out words with hyphens! Write normal complete words. Example: "Idhu romba simple, easily puriyum."'}
"""
                toks = []
                with st.spinner("Thinking…"):
                    for tok in llm._stream(prompt):
                        toks.append(tok)
                add_chat("assistant", "".join(toks), lang=current_lang)

        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — QUIZ (structured phase-based flow)
# ══════════════════════════════════════════════════════════════════════════════
with tab_quiz:
    current_lang = st.session_state.language
    lang_key     = LANG_MAP[current_lang]

    st.markdown("### 🎯 Quiz Yourself")
    st.markdown(f"<p style='color:#6b7280;font-size:0.85rem;'>Explanation language: <strong style='color:#ececec'>{current_lang}</strong> — switch in sidebar</p>",
                unsafe_allow_html=True)

    # ── Score strip ──────────────────────────────────────────────────────────
    total = st.session_state.total_asked
    score = st.session_state.score
    acc   = int(score / total * 100) if total > 0 else 0
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("✅ Correct",  score)
    c2.metric("📝 Total Q's", total)
    c3.metric("🎯 Accuracy",  f"{acc}%")
    c4.metric("📚 Chunks",    rag.get_stats()["total_chunks"])
    st.markdown("---")

    phase = st.session_state.quiz_phase

    # ════════════════════ PHASE: GENERATE ═══════════════════════════════════
    if phase == "generate":
        st.markdown("#### Pick a difficulty and generate your question:")

        col_easy, col_med, col_hard = st.columns(3)
        with col_easy:
            if st.button("🟢 Easy", use_container_width=True, key="q_easy"):
                st.session_state.quiz_difficulty = "easy"
        with col_med:
            if st.button("🟠 Medium", use_container_width=True, key="q_med"):
                st.session_state.quiz_difficulty = "medium"
        with col_hard:
            if st.button("🔵 Conceptual", use_container_width=True, key="q_hard"):
                st.session_state.quiz_difficulty = "conceptual"

        selected_diff = st.session_state.quiz_difficulty
        chip_map = {"easy": "chip-easy", "medium": "chip-medium", "conceptual": "chip-conceptual"}
        st.markdown(f"<p>Selected: <span class='chip {chip_map[selected_diff]}'>{selected_diff.upper()}</span></p>",
                    unsafe_allow_html=True)

        st.markdown("")
        if st.button("🎲 Generate Question", use_container_width=True, key="gen_q"):
            with st.spinner("Reading material and generating question…"):
                chunk = rag.get_random_chunk()
                if chunk:
                    qs = llm.generate_questions(chunk)
                    q  = qs.get(selected_diff, qs.get("medium", "What is this about?"))
                    st.session_state.quiz_questions = qs
                    st.session_state.quiz_question  = q
                    st.session_state.quiz_context   = chunk
                    st.session_state.quiz_eval      = None
                    st.session_state.quiz_explanation = None
                    st.session_state.quiz_followup_q  = None
                    st.session_state.quiz_phase     = "answer"
                    st.rerun()
                else:
                    st.error("❌ No chunks available. Re-upload the PDF.")

        # ── Recent history ───────────────────────────────────────────────────
        if st.session_state.history:
            st.markdown("---")
            st.markdown("#### 📜 Recent Questions")
            for h in reversed(st.session_state.history[-5:]):
                v = h["verdict"]
                icons = {"Correct": "✅", "Partial": "⚠️", "Wrong": "❌"}
                colour = {"Correct": "#19c37d", "Partial": "#fbbf24", "Wrong": "#ef4444"}.get(v, "#6b7280")
                st.markdown(f"""
<div class='hist-item'>
  <span style='color:{colour};font-size:1.2rem;'>{icons.get(v,'📝')}</span>
  <div>
    <span style='color:#9b9b9b;font-size:0.75rem;font-weight:700;text-transform:uppercase;'>{h['difficulty']}</span>
    <p style='color:#ececec;margin:2px 0 0;font-size:0.92rem;'>{h['question'][:110]}{'…' if len(h['question'])>110 else ''}</p>
  </div>
</div>""", unsafe_allow_html=True)

    # ════════════════════ PHASE: ANSWER ══════════════════════════════════════
    elif phase == "answer":
        q    = st.session_state.quiz_question
        diff = st.session_state.quiz_difficulty
        chip_map = {"easy": "chip-easy", "medium": "chip-medium", "conceptual": "chip-conceptual"}

        st.markdown(f"""
<div class='quiz-card'>
  <span class='chip {chip_map.get(diff,"chip-medium")}'>{diff.upper()}</span>
  <h3 style='color:#ececec;margin:10px 0 0;font-size:1.1rem;'>❓ {q}</h3>
</div>""", unsafe_allow_html=True)

        # Difficulty switcher
        st.markdown("**Switch difficulty:**")
        qs = st.session_state.quiz_questions or {}
        dc1, dc2, dc3 = st.columns(3)
        with dc1:
            if st.button("🟢 Easy", key="sw_easy"):
                st.session_state.quiz_difficulty = "easy"
                st.session_state.quiz_question   = qs.get("easy", q)
                st.rerun()
        with dc2:
            if st.button("🟠 Medium", key="sw_med"):
                st.session_state.quiz_difficulty = "medium"
                st.session_state.quiz_question   = qs.get("medium", q)
                st.rerun()
        with dc3:
            if st.button("🔵 Conceptual", key="sw_hard"):
                st.session_state.quiz_difficulty = "conceptual"
                st.session_state.quiz_question   = qs.get("conceptual", q)
                st.rerun()

        st.markdown("")
        user_ans = st.text_area("✏️ Your Answer:", height=130,
                                placeholder="Type your answer here…", key="quiz_ans_input")

        sub_col, skip_col = st.columns([4, 1])
        with sub_col:
            if st.button("📤 Submit Answer", use_container_width=True, key="quiz_submit"):
                if not user_ans.strip():
                    st.warning("Please type an answer first.")
                else:
                    with st.spinner("🤖 Evaluating…"):
                        result = llm.evaluate_answer(
                            question=st.session_state.quiz_question,
                            user_answer=user_ans,
                            context=st.session_state.quiz_context
                        )
                    st.session_state.quiz_eval   = result
                    st.session_state.total_asked += 1
                    if result["is_correct"]:
                        st.session_state.score += 1
                    st.session_state.history.append({
                        "question":   st.session_state.quiz_question,
                        "difficulty": st.session_state.quiz_difficulty,
                        "verdict":    result["verdict"],
                        "answer":     user_ans
                    })
                    st.session_state.quiz_phase = "evaluated"
                    st.rerun()

        with skip_col:
            if st.button("⏭ Skip", use_container_width=True, key="quiz_skip"):
                st.session_state.quiz_phase    = "generate"
                st.session_state.quiz_question = None
                st.rerun()

    # ════════════════════ PHASE: EVALUATED ═══════════════════════════════════
    elif phase == "evaluated":
        result  = st.session_state.quiz_eval
        verdict = result["verdict"]
        weak    = result.get("weak_concept", "")
        q       = st.session_state.quiz_question
        diff    = st.session_state.quiz_difficulty
        chip_map = {"easy": "chip-easy", "medium": "chip-medium", "conceptual": "chip-conceptual"}
        icons   = {"Correct": "✅", "Partial": "⚠️", "Wrong": "❌"}
        colours = {"Correct": "#19c37d", "Partial": "#fbbf24", "Wrong": "#ef4444"}

        st.markdown(f"""
<div class='quiz-card' style='border-color:{colours.get(verdict,"#2f2f2f")}'>
  <span class='chip {chip_map.get(diff,"chip-medium")}'>{diff.upper()}</span>
  <p style='color:#9b9b9b;font-size:0.88rem;margin:6px 0;'>❓ {q}</p>
  <div class='verdict-card verdict-{verdict.lower()}'>
    <strong>{icons.get(verdict,'📝')} {verdict}</strong><br>
    <span style='font-size:0.92rem;'>{result['explanation']}</span>
    {f"<br><small style='color:#fbbf24;'>Weak concept: {weak}</small>" if weak and weak.lower() != 'none' and verdict != 'Correct' else ""}
  </div>
</div>""", unsafe_allow_html=True)

        if verdict == "Correct":
            st.success("🎉 Excellent! You understood it perfectly.")
            if st.button("➡️ Next Question", key="next_q_correct"):
                st.session_state.quiz_phase    = "generate"
                st.session_state.quiz_question = None
                st.rerun()
        else:
            st.warning(f"Let me reteach **{weak or 'this concept'}** in {current_lang}!")
            if st.button(f"📚 Explain in {current_lang}", use_container_width=True, key="q_explain_btn"):
                retrieved = rag.retrieve_relevant_chunks(weak or q, k=3)
                best_ctx  = "\n\n".join(retrieved) if retrieved else st.session_state.quiz_context

                st.markdown(f"#### 💡 Explanation in {current_lang}")
                if current_lang == "Tanglish":
                    st.markdown("<p style='color:#7c3aed;font-size:0.82rem;'>Writing in Tanglish (Tamil sounds using A–Z English letters)…</p>",
                                unsafe_allow_html=True)

                streamed = st.write_stream(
                    llm.explain_stream(weak_concept=weak or q, context=best_ctx, language=lang_key)
                )
                st.session_state.quiz_explanation = streamed
                st.session_state.quiz_phase       = "teaching"
                st.rerun()

    # ════════════════════ PHASE: TEACHING ════════════════════════════════════
    elif phase == "teaching":
        result = st.session_state.quiz_eval
        weak   = result.get("weak_concept", "concept")
        expl   = st.session_state.quiz_explanation or ""

        st.markdown(f"#### 👨‍🏫 AI Tutor — Teaching in {current_lang}")
        st.markdown(f"**Concept:** `{weak}`")
        st.markdown(f"<div class='explain-box'>{expl}</div>", unsafe_allow_html=True)
        st.markdown("---")

        st.markdown("#### 🔁 Follow-Up Check")
        st.markdown("Answer one more quick question to confirm understanding!")

        if st.button("🎯 Generate Follow-Up Question", use_container_width=True, key="gen_fu"):
            with st.spinner("Generating follow-up…"):
                fq = llm.generate_followup_question(weak, st.session_state.quiz_context)
            st.session_state.quiz_followup_q = fq
            st.session_state.quiz_phase      = "followup"
            st.rerun()

        if st.button("⏭ Skip & Next Question", key="skip_fu"):
            st.session_state.quiz_phase    = "generate"
            st.session_state.quiz_question = None
            st.rerun()

    # ════════════════════ PHASE: FOLLOWUP ════════════════════════════════════
    elif phase == "followup":
        fq   = st.session_state.quiz_followup_q
        weak = st.session_state.quiz_eval.get("weak_concept", "")

        st.markdown(f"""
<div class='quiz-card' style='border-color:#0ea5e9'>
  <span class='chip chip-followup'>FOLLOW-UP</span>
  <p style='color:#9b9b9b;font-size:0.82rem;margin:6px 0;'>Testing: <em>{weak}</em></p>
  <h3 style='color:#ececec;margin:10px 0 0;font-size:1.05rem;'>❓ {fq}</h3>
</div>""", unsafe_allow_html=True)

        fu_ans = st.text_area("✏️ Your Answer:", height=110,
                              placeholder="Show what you learned…", key="fu_ans")
        if st.button("📤 Submit Follow-Up", use_container_width=True, key="fu_submit"):
            if not fu_ans.strip():
                st.warning("Please type an answer.")
            else:
                with st.spinner("Checking improvement…"):
                    fu = llm.evaluate_followup(fq, fu_ans, weak)
                st.session_state.quiz_followup_res = fu
                st.session_state.quiz_phase        = "followup_result"
                st.rerun()

    # ════════════════════ PHASE: FOLLOWUP RESULT ══════════════════════════════
    elif phase == "followup_result":
        fu = st.session_state.quiz_followup_res

        if fu["improved"]:
            st.balloons()
            st.markdown(f"""
<div class='quiz-card' style='border-color:#19c37d'>
  <span class='chip chip-easy'>IMPROVED ✅</span>
  <p style='color:#86efac;margin:10px 0 0;'>{fu['feedback']}</p>
</div>""", unsafe_allow_html=True)
            st.success("🌟 Excellent! Your understanding improved!")
        else:
            st.markdown(f"""
<div class='quiz-card' style='border-color:#f59e0b'>
  <span class='chip chip-medium'>NEEDS PRACTICE 📚</span>
  <p style='color:#fcd34d;margin:10px 0 0;'>{fu['feedback']}</p>
</div>""", unsafe_allow_html=True)
            st.warning("💪 Keep practising — you'll get it!")

        if st.button("➡️ Continue to Next Question", use_container_width=True, key="next_after_fu"):
            st.session_state.quiz_phase       = "generate"
            st.session_state.quiz_question    = None
            st.session_state.quiz_eval        = None
            st.session_state.quiz_explanation = None
            st.session_state.quiz_followup_q  = None
            st.rerun()
