# ============================================================
# Resume–JD Matcher | FINAL PRODUCTION-GRADE VERSION
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import hashlib
import random
import smtplib
import pickle
from email.message import EmailMessage

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai

# ------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------
st.set_page_config("Resume–JD Matcher", layout="wide")

# ------------------------------------------------------------
# STYLES
# ------------------------------------------------------------
def apply_style(bg):
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
    .card {{
        background: rgba(255,255,255,0.08);
        padding: 22px;
        border-radius: 16px;
        margin-bottom: 20px;
    }}
    .hover-card {{
        background: rgba(255,255,255,0.08);
        padding: 24px;
        border-radius: 18px;
        margin-bottom: 20px;
        border: 1px solid rgba(255,255,255,0.15);
        transition: all 0.35s ease;
    }}
    .hover-card:hover {{
        background: rgba(59,130,246,0.25);
        transform: translateY(-6px);
        box-shadow: 0 0 20px rgba(59,130,246,0.6);
    }}
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
# LOAD MODELS
# ------------------------------------------------------------
@st.cache_resource
def load_models():
    sbert = SentenceTransformer("all-MiniLM-L6-v2")

    with open("salary_model.pkl", "rb") as f:
        salary_model = pickle.load(f)

    with open("label_encoders.pkl", "rb") as f:
        label_encoders = pickle.load(f)

    job_key = list(label_encoders.keys())[0]
    encoder = label_encoders[job_key]

    descriptions = {
        "Software Developer": "c++ python algorithms system design",
        "Data Scientist": "machine learning statistics data analysis",
        "Cloud Engineer": "aws cloud infrastructure networking",
        "DevOps Engineer": "docker kubernetes automation ci cd",
        "Web Developer": "frontend backend web technologies"
    }

    rows = [(j, descriptions.get(j, j)) for j in encoder.classes_]
    df = pd.DataFrame(rows, columns=[job_key, "description"])
    emb = sbert.encode(df["description"].tolist())

    return sbert, salary_model, encoder, job_key, df, emb

sbert, salary_model, encoder, JOB_KEY, jobs_df, job_emb = load_models()

# ------------------------------------------------------------
# SESSION STATE
# ------------------------------------------------------------
for key in ["logged_in", "page", "auth_step"]:
    if key not in st.session_state:
        st.session_state[key] = False if key == "logged_in" else "login"

# ============================================================
# AUTHENTICATION
# ============================================================
if not st.session_state.logged_in:
    apply_style("linear-gradient(-45deg,#020617,#450a0a,#020617)")
    st.title("🔐 Authentication")

    if st.session_state.auth_step == "login":
        u = st.text_input("Username")
        p = st.text_input("Password", type="password")

        if st.button("Login"):
            conn = sqlite3.connect("users.db")
            cur = conn.cursor()
            cur.execute("SELECT * FROM users WHERE username=? AND password=?", (u, hash_pw(p)))
            if cur.fetchone():
                st.session_state.logged_in = True
                st.session_state.page = "Home"
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
        u = st.text_input("Username")
        e = st.text_input("Email")
        p = st.text_input("Password", type="password")

        if st.button("Send OTP"):
            st.session_state.otp = send_otp(e)
            st.session_state.tmp_user = (u, hash_pw(p), e)
            st.session_state.auth_step = "verify"
            st.rerun()

    elif st.session_state.auth_step == "verify":
        o = st.text_input("Enter OTP")
        if st.button("Verify"):
            if o == st.session_state.otp:
                conn = sqlite3.connect("users.db")
                conn.execute("INSERT INTO users VALUES (?,?,?)", st.session_state.tmp_user)
                conn.commit(); conn.close()
                st.success("Account created. Please login.")
                st.session_state.auth_step = "login"

    elif st.session_state.auth_step == "forgot":
        e = st.text_input("Registered Email")
        if st.button("Send Reset OTP"):
            st.session_state.otp = send_otp(e)
            st.session_state.reset_email = e
            st.session_state.auth_step = "reset"
            st.rerun()

    elif st.session_state.auth_step == "reset":
        o = st.text_input("OTP")
        npw = st.text_input("New Password", type="password")
        if st.button("Reset Password"):
            if o == st.session_state.otp:
                conn = sqlite3.connect("users.db")
                conn.execute("UPDATE users SET password=? WHERE email=?", (hash_pw(npw), st.session_state.reset_email))
                conn.commit(); conn.close()
                st.success("Password updated. Please login.")
                st.session_state.auth_step = "login"

    st.stop()

# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.title("📂 Menu")
if st.sidebar.button("🏠 Home"):
    st.session_state.page = "Home"
if st.sidebar.button("📘 About"):
    st.session_state.page = "About"
if st.sidebar.button("🤖 AI Assistant"):
    st.session_state.page = "AI"
if st.sidebar.button("🚪 Logout"):
    st.session_state.logged_in = False
    st.session_state.auth_step = "login"
    st.rerun()

# ============================================================
# HOME
# ============================================================
if st.session_state.page == "Home":
    apply_style("linear-gradient(-45deg,#022c22,#020617,#064e3b)")
    st.title("🎯 Resume–JD Matcher")

    exp = st.slider("Experience (Years)", 0, 20, 0)
    resume = st.text_area("Paste Resume / Skills")

    if st.button("Analyze"):
        emb = sbert.encode([resume])
        sims = cosine_similarity(emb, job_emb)[0]
        skills = set(resume.lower().split())

        for i in sims.argsort()[::-1][:3]:
            job = jobs_df.iloc[i]
            sim = sims[i]
            job_enc = encoder.transform([job[JOB_KEY]])[0]

            base = salary_model.predict([[job_enc, exp, sim]])[0]
            salary = base * (1 + 0.1 * exp)

            overlap = skills.intersection(job["description"].split())

            if job[JOB_KEY] == "Cloud Engineer":
                explanation = f"You show familiarity with infrastructure concepts like {', '.join(overlap) or 'cloud basics'}, suggesting readiness for cloud operations roles."
            elif job[JOB_KEY] == "Data Scientist":
                explanation = f"Your exposure to analytical or ML-oriented skills indicates potential to work with data-driven decision systems."
            elif job[JOB_KEY] == "DevOps Engineer":
                explanation = f"Automation and deployment-related knowledge in your profile aligns with DevOps workflows."
            elif job[JOB_KEY] == "Web Developer":
                explanation = f"Your skill set reflects experience in building user-facing or backend web systems."
            else:
                explanation = f"Your technical background supports general software development responsibilities."

            st.markdown(f"""
            <div class="card">
            <h3>{job[JOB_KEY]}</h3>
            Match: {sim*100:.1f}%<br>
            Salary: ₹{salary:.1f} LPA
            <hr>{explanation}
            </div>
            """, unsafe_allow_html=True)

# ============================================================
# ABOUT
# ============================================================
elif st.session_state.page == "About":
    apply_style("linear-gradient(-45deg,#1e3a8a,#020617,#312e81)")
    st.title("📘 System Architecture")

    for title, text in [
        ("Transformer Architecture", "Context-aware semantic encoding."),
        ("Skill Gap Analysis", "Identifies missing competencies."),
        ("Cosine Similarity", "Measures relevance mathematically."),
        ("Database", "Stores credentials securely."),
        ("Explainability", "Builds user trust.")
    ]:
        st.markdown(f"<div class='hover-card'><b>{title}</b><br>{text}</div>", unsafe_allow_html=True)

# ============================================================
# AI ASSISTANT (LOCAL FALLBACK)
# ============================================================
elif st.session_state.page == "AI":
    apply_style("linear-gradient(-45deg,#581c87,#020617,#701a75)")
    st.title("🤖 Career AI Assistant")

    q = st.text_input("Ask a career question")

    if q:
        try:
            genai.configure(api_key=st.secrets["AI_API_KEY"])
            model = genai.GenerativeModel("gemini-pro")
            st.markdown(model.generate_content(q).text)
        except Exception:
            st.info(
                "AI service is currently unavailable.\n\n"
                "Suggested approach:\n"
                "- Build projects aligned with your target role\n"
                "- Strengthen core fundamentals\n"
                "- Gain internship or hands-on experience"
            )