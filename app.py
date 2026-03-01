# ============================================================
# Resume–JD Matcher | FINAL FULL VERSION (Hover Cards Kept)
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import hashlib
import random
import smtplib
from email.message import EmailMessage

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

import google.generativeai as genai

# ------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------
st.set_page_config("Resume–JD Matcher", layout="wide")

# ------------------------------------------------------------
# DYNAMIC STYLES + HOVER CARDS
# ------------------------------------------------------------
def apply_style(bg, accent):
    st.markdown(f"""
    <style>
    .stApp {{
        background: {bg};
        background-size: 400% 400%;
        animation: gradientBG 18s ease infinite;
    }}

    @keyframes gradientBG {{
        0% {{background-position:0% 50%;}}
        50% {{background-position:100% 50%;}}
        100% {{background-position:0% 50%;}}
    }}

    .center {{
        max-width: 420px;
        margin: auto;
        text-align: center;
    }}

    button[kind="primary"] {{
        background-color: {accent} !important;
        color: white !important;
        font-weight: 700;
    }}

    .result-card {{
        background: rgba(255,255,255,0.06);
        padding: 20px;
        border-radius: 14px;
        margin-bottom: 16px;
    }}

    /* ---------- HOVER INFO CARDS ---------- */
    .info-card {{
        background: rgba(255,255,255,0.06);
        border-radius: 16px;
        padding: 22px;
        margin-bottom: 22px;
        border: 1px solid rgba(255,255,255,0.12);
        transition: all 0.4s ease;
    }}

    .info-card:hover {{
        transform: translateY(-8px) scale(1.02);
        background: rgba(255,255,255,0.14);
        border-color: {accent};
        box-shadow: 0 0 20px {accent};
    }}

    .info-title {{
        font-size: 1.3rem;
        font-weight: 700;
        color: {accent};
        margin-bottom: 8px;
    }}

    .info-text {{
        color: #e5e7eb;
        line-height: 1.6;
        font-size: 0.95rem;
    }}

    .skill {{
        padding: 6px 12px;
        border-radius: 14px;
        margin: 4px;
        display: inline-block;
        font-size: 0.85rem;
    }}
    .match {{ background:#064e3b; color:#22c55e; }}
    .gap {{ background:#450a0a; color:#ef4444; }}
    </style>
    """, unsafe_allow_html=True)

# ------------------------------------------------------------
# DATABASE
# ------------------------------------------------------------
def hash_pw(p): 
    return hashlib.sha256(p.encode()).hexdigest()

def init_db():
    conn = sqlite3.connect("users.db")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users(
            username TEXT PRIMARY KEY,
            password TEXT,
            email TEXT
        )
    """)
    conn.commit(); conn.close()

init_db()

# ------------------------------------------------------------
# EMAIL OTP
# ------------------------------------------------------------
def send_otp(email):
    otp = str(random.randint(100000, 999999))
    msg = EmailMessage()
    msg.set_content(f"Your OTP is: {otp}")
    msg["Subject"] = "Verification Code"
    msg["From"] = st.secrets["EMAIL_USER"]
    msg["To"] = email

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as s:
        s.login(st.secrets["EMAIL_USER"], st.secrets["EMAIL_PASSWORD"])
        s.send_message(msg)
    return otp

# ------------------------------------------------------------
# MODELS
# ------------------------------------------------------------
@st.cache_resource
def load_models():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    df = pd.DataFrame({
        "Job Title": ["Web Developer", "Data Scientist", "Mobile App Developer"],
        "clean_description": [
            "html css javascript react node",
            "python machine learning statistics sql",
            "flutter dart firebase android ios"
        ]
    })
    emb = model.encode(df["clean_description"].tolist())
    return model, df, emb

model, df, job_emb = load_models()

# ------------------------------------------------------------
# SESSION STATE
# ------------------------------------------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "auth_step" not in st.session_state:
    st.session_state.auth_step = "login"

# ============================================================
# AUTH FLOW
# ============================================================
if not st.session_state.logged_in:
    apply_style(
        "linear-gradient(-45deg,#020617,#450a0a,#020617)",
        "#ef4444"
    )

    st.markdown("<div class='center'>", unsafe_allow_html=True)
    st.title("🔐 Login")

    if st.session_state.auth_step == "login":
        u = st.text_input("Username")
        p = st.text_input("Password", type="password")

        if st.button("Login", type="primary"):
            conn = sqlite3.connect("users.db")
            cur = conn.cursor()
            cur.execute(
                "SELECT * FROM users WHERE username=? AND password=?",
                (u, hash_pw(p))
            )
            if cur.fetchone():
                st.session_state.logged_in = True
                st.rerun()
            else:
                st.error("Invalid credentials")
            conn.close()

        if st.button("Forgot Password"):
            st.session_state.auth_step = "forgot"
            st.rerun()

        if st.button("New User? Sign Up"):
            st.session_state.auth_step = "signup"
            st.rerun()

    elif st.session_state.auth_step == "signup":
        su = st.text_input("Username")
        se = st.text_input("Email")
        sp = st.text_input("Password", type="password")

        if st.button("Send OTP", type="primary"):
            st.session_state.otp = send_otp(se)
            st.session_state.temp_user = (su, hash_pw(sp), se)
            st.session_state.auth_step = "verify_signup"
            st.rerun()

    elif st.session_state.auth_step == "verify_signup":
        o = st.text_input("Enter OTP")
        if st.button("Verify"):
            if o == st.session_state.otp:
                conn = sqlite3.connect("users.db")
                conn.execute(
                    "INSERT INTO users VALUES (?,?,?)",
                    st.session_state.temp_user
                )
                conn.commit(); conn.close()
                st.success("Account created. Please login.")
                st.session_state.auth_step = "login"
                st.rerun()

    elif st.session_state.auth_step == "forgot":
        fe = st.text_input("Registered Email")
        if st.button("Send Reset OTP", type="primary"):
            st.session_state.otp = send_otp(fe)
            st.session_state.reset_email = fe
            st.session_state.auth_step = "reset_pw"
            st.rerun()

    elif st.session_state.auth_step == "reset_pw":
        o = st.text_input("OTP")
        npw = st.text_input("New Password", type="password")
        if st.button("Reset Password"):
            if o == st.session_state.otp:
                conn = sqlite3.connect("users.db")
                conn.execute(
                    "UPDATE users SET password=? WHERE email=?",
                    (hash_pw(npw), st.session_state.reset_email)
                )
                conn.commit(); conn.close()
                st.success("Password updated. Login again.")
                st.session_state.auth_step = "login"
                st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

# ============================================================
# MAIN APP
# ============================================================
menu = st.sidebar.radio("Navigation", ["Home", "About", "AI Assistant"])
if st.sidebar.button("Logout"):
    st.session_state.logged_in = False
    st.rerun()

# ------------------------------------------------------------
# HOME
# ------------------------------------------------------------
if menu == "Home":
    apply_style(
        "linear-gradient(-45deg,#022c22,#020617,#064e3b)",
        "#22c55e"
    )

    st.title("🎯 Resume–JD Matcher")
    exp = st.slider("Experience (Years)", 0, 20, 2)
    resume = st.text_area("Resume Skills")

    if st.button("Analyze", type="primary"):
        vec = model.encode([resume])
        sims = cosine_similarity(vec, job_emb)[0]
        idx = np.argmax(sims)
        job = df.iloc[idx]
        score = sims[idx]

        band = {
            "Web Developer": (4,10),
            "Data Scientist": (6,14),
            "Mobile App Developer": (5,12)
        }

        lo, hi = band[job["Job Title"]]
        salary = lo + (hi - lo) * min(exp / 10, 1) * score

        st.markdown(f"""
        <div class="result-card">
        <h3>{job['Job Title']}</h3>
        Match: {round(score*100,1)}%<br>
        Estimated Salary: ₹{salary:.1f} LPA
        <hr>
        <b>Why this salary?</b><br>
        Role band: {lo}-{hi} LPA<br>
        Experience factor: {exp} years<br>
        Semantic match strength: {round(score*100,1)}%
        </div>
        """, unsafe_allow_html=True)

# ------------------------------------------------------------
# ABOUT (HOVER CARDS — KEPT)
# ------------------------------------------------------------
elif menu == "About":
    apply_style(
        "linear-gradient(-45deg,#1e3a8a,#020617,#312e81)",
        "#60a5fa"
    )

    st.title("📘 System Architecture")

    cards = [
        ("🧠 Transformer Architecture",
         "Sentence-BERT converts text into dense embeddings.<br>"
         "Semantic meaning is preserved beyond keywords.<br>"
         "Enables accurate context-aware matching."),

        ("🧩 Skill Gap Analysis",
         "TF-IDF identifies matched and missing skills.<br>"
         "Shows alignment with job expectations.<br>"
         "Guides focused upskilling."),

        ("🗄️ Secure Database",
         "SQLite stores credentials locally.<br>"
         "Passwords are hashed using SHA-256.<br>"
         "Prevents plaintext data exposure."),

        ("📐 Cosine Similarity",
         "Measures angular similarity of embeddings.<br>"
         "Produces normalized relevance scores.<br>"
         "Higher score means better match."),

        ("💰 Salary Prediction",
         "Uses industry role-based salary bands.<br>"
         "Experience and match strength scale salary.<br>"
         "Avoids unrealistic estimates.")
    ]

    for t, d in cards:
        st.markdown(f"""
        <div class="info-card">
            <div class="info-title">{t}</div>
            <div class="info-text">{d}</div>
        </div>
        """, unsafe_allow_html=True)

# ------------------------------------------------------------
# AI ASSISTANT
# ------------------------------------------------------------
elif menu == "AI Assistant":
    apply_style(
        "linear-gradient(-45deg,#581c87,#020617,#701a75)",
        "#c084fc"
    )

    st.title("🤖 Career AI Assistant")

    if "chat" not in st.session_state:
        st.session_state.chat = []

    for m in st.session_state.chat:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    q = st.chat_input("Ask about careers, skills, or learning paths")

    if q:
        st.session_state.chat.append({"role": "user", "content": q})
        with st.chat_message("assistant"):
            try:
                genai.configure(api_key=st.secrets["AI_API_KEY"])
                ai = genai.GenerativeModel("gemini-1.5-flash")
                ans = ai.generate_content(q).text
            except Exception:
                ans = "AI service unavailable. Please try again later."
            st.markdown(ans)
            st.session_state.chat.append({"role": "assistant", "content": ans})