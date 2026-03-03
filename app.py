# ============================================================
# Resume–JD Matcher | FULL FINAL PRODUCTION VERSION
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import hashlib
import random
import smtplib
import time
import re
from email.message import EmailMessage
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai

# ------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------
st.set_page_config("Resume–JD Matcher", layout="wide")

# ------------------------------------------------------------
# MODERN UI STYLE
# ------------------------------------------------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(-45deg,#0f172a,#1e293b,#0f172a);
    background-size: 400% 400%;
    animation: gradientBG 15s ease infinite;
    color: white;
}
@keyframes gradientBG {
    0% {background-position:0% 50%;}
    50% {background-position:100% 50%;}
    100% {background-position:0% 50%;}
}
.card {
    backdrop-filter: blur(12px);
    background: rgba(255,255,255,0.05);
    padding: 25px;
    border-radius: 18px;
    margin-bottom: 25px;
    animation: fadeIn 0.8s ease-in-out;
    transition: 0.3s;
}
.card:hover {
    transform: scale(1.02);
    box-shadow: 0 0 25px rgba(59,130,246,0.5);
}
@keyframes fadeIn {
    from {opacity:0; transform: translateY(15px);}
    to {opacity:1; transform: translateY(0);}
}
.stButton>button {
    background: linear-gradient(90deg,#2563eb,#7c3aed);
    color:white;
    border-radius:10px;
    padding:10px 20px;
    border:none;
    transition:0.3s;
}
.stButton>button:hover {
    transform:scale(1.05);
}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# DATABASE SETUP
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
    conn.commit()
    conn.close()

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
# LOAD TRANSFORMER MODEL
# ------------------------------------------------------------
@st.cache_resource
def load_model():
    model = SentenceTransformer("all-MiniLM-L6-v2")

    descriptions = {
        "Software Developer":
        "java python c++ backend system_design algorithms data_structures api rest git microservices",

        "Data Scientist":
        "python machine_learning statistics pandas numpy data_analysis deep_learning sql r nlp",

        "Cloud Engineer":
        "aws cloud_computing docker kubernetes infrastructure devops terraform monitoring",

        "DevOps Engineer":
        "ci_cd automation docker kubernetes linux scripting terraform jenkins monitoring",

        "Web Developer":
        "html css javascript react frontend backend node express mongodb rest_api"
    }

    df = pd.DataFrame(list(descriptions.items()), columns=["Role", "description"])
    embeddings = model.encode(df["description"].tolist())
    return model, df, embeddings

model, jobs_df, job_embeddings = load_model()

# ------------------------------------------------------------
# SESSION STATE
# ------------------------------------------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "auth_step" not in st.session_state:
    st.session_state.auth_step = "login"
if "page" not in st.session_state:
    st.session_state.page = "Home"

# ============================================================
# AUTHENTICATION
# ============================================================
if not st.session_state.logged_in:

    st.title("🔐 Authentication")

    # LOGIN
    if st.session_state.auth_step == "login":
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login"):
            conn = sqlite3.connect("users.db")
            cur = conn.cursor()
            cur.execute("SELECT * FROM users WHERE username=? AND password=?",
                        (username, hash_pw(password)))
            if cur.fetchone():
                st.session_state.logged_in = True
                st.rerun()
            else:
                st.error("Invalid credentials")
            conn.close()

        col1, col2 = st.columns(2)
        if col1.button("New User? Sign Up"):
            st.session_state.auth_step = "signup"
            st.rerun()
        if col2.button("Forgot Password"):
            st.session_state.auth_step = "forgot"
            st.rerun()

    # SIGNUP
    elif st.session_state.auth_step == "signup":
        new_user = st.text_input("Choose Username")
        new_email = st.text_input("Email")
        new_pass = st.text_input("Password", type="password")

        if st.button("Send OTP"):
            st.session_state.otp = send_otp(new_email)
            st.session_state.temp_user = (new_user, hash_pw(new_pass), new_email)
            st.session_state.auth_step = "verify"
            st.success("OTP sent.")
            st.rerun()

    elif st.session_state.auth_step == "verify":
        otp_input = st.text_input("Enter OTP")
        if st.button("Verify"):
            if otp_input == st.session_state.otp:
                conn = sqlite3.connect("users.db")
                conn.execute("INSERT INTO users VALUES (?,?,?)",
                             st.session_state.temp_user)
                conn.commit()
                conn.close()
                st.success("Account created. Please login.")
                st.session_state.auth_step = "login"
            else:
                st.error("Invalid OTP")

    # FORGOT PASSWORD (USERNAME + EMAIL)
    elif st.session_state.auth_step == "forgot":

        reset_username = st.text_input("Username")
        reset_email = st.text_input("Registered Email")

        if st.button("Send Reset OTP"):

            conn = sqlite3.connect("users.db")
            cur = conn.cursor()
            cur.execute("SELECT * FROM users WHERE username=? AND email=?",
                        (reset_username, reset_email))
            user_exists = cur.fetchone()
            conn.close()

            if user_exists:
                st.session_state.otp = send_otp(reset_email)
                st.session_state.reset_username = reset_username
                st.session_state.reset_email = reset_email
                st.session_state.auth_step = "reset"
                st.success("OTP sent to registered email.")
                st.rerun()
            else:
                st.error("Username and Email do not match.")

    elif st.session_state.auth_step == "reset":

        otp_input = st.text_input("Enter OTP")
        new_password = st.text_input("New Password", type="password")

        if st.button("Reset Password"):

            if otp_input == st.session_state.otp:

                conn = sqlite3.connect("users.db")
                conn.execute(
                    "UPDATE users SET password=? WHERE username=? AND email=?",
                    (
                        hash_pw(new_password),
                        st.session_state.reset_username,
                        st.session_state.reset_email
                    )
                )
                conn.commit()
                conn.close()

                st.success("Password updated successfully.")
                st.session_state.auth_step = "login"
            else:
                st.error("Invalid OTP")

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

    st.title("🚀 Resume–JD Matcher")

    exp = st.slider("Experience (Years)", 0, 20, 0)
    resume = st.text_area("Paste Resume / Skills")

    if st.button("Analyze"):

        with st.spinner("🔍 Performing intelligent skill analysis..."):
            time.sleep(1)

            skills = re.split(r'[,\s]+', resume.lower().strip())
            skills = [s for s in skills if s]

            resume_emb = model.encode([resume])
            sims = cosine_similarity(resume_emb, job_embeddings)[0]

            for i in sims.argsort()[::-1][:3]:

                role = jobs_df.iloc[i]["Role"]
                desc = jobs_df.iloc[i]["description"]
                role_tokens = desc.split()
                role_embs = model.encode(role_tokens)

                positive, moderate, unmatched = [], [], []

                for skill in skills:
                    if skill in role_tokens:
                        positive.append((skill, 1.0))
                        continue

                    skill_emb = model.encode([skill])
                    scores = cosine_similarity(skill_emb, role_embs)[0]
                    max_score = np.max(scores)

                    if max_score > 0.55:
                        positive.append((skill, max_score))
                    elif max_score > 0.35:
                        moderate.append((skill, max_score))
                    else:
                        unmatched.append(skill)

                semantic_score = sims[i]
                skill_weight = len(positive)*0.6 + len(moderate)*0.3
                final_score = 0.6*semantic_score + 0.4*skill_weight

                base_salary = 3 + semantic_score*3
                experience_bonus = exp*0.7
                skill_bonus = len(positive)*0.5
                salary = base_salary + experience_bonus + skill_bonus

                explanation_lines = []
                for skill, score in positive:
                    explanation_lines.append(
                        f"• {skill} significantly contributes to your suitability for {role}."
                    )
                for skill, score in moderate:
                    explanation_lines.append(
                        f"• {skill} partially supports the role requirements."
                    )
                if not explanation_lines:
                    explanation_lines.append(
                        "• Your skills show general technical relevance but lack strong specialization for this role."
                    )

                explanation_html = "<br>".join(explanation_lines)

                st.markdown(f"""
                <div class="card">
                <h2>{role}</h2>
                <b>Match Score:</b> {final_score*100:.1f}%<br>
                <b>Predicted Salary:</b> ₹{salary:.1f} LPA
                <hr>
                <b>Explainability:</b><br>
                {explanation_html}
                <br><br>
                <b>Matched Skills:</b> {', '.join([s for s,_ in positive]) if positive else "None"}<br>
                <b>Unmatched Skills:</b> {', '.join(unmatched) if unmatched else "None"}
                <br><br>
                <b>Salary Breakdown:</b><br>
                Base: ₹{base_salary:.1f} LPA<br>
                Experience Impact: ₹{experience_bonus:.1f} LPA<br>
                Skill Bonus: ₹{skill_bonus:.1f} LPA
                </div>
                """, unsafe_allow_html=True)

# ============================================================
# ABOUT
# ============================================================
elif st.session_state.page == "About":

    st.title("📘 System Architecture")

    sections = {
        "Transformer Architecture":
        "We use SentenceTransformer (MiniLM) to convert resumes and job descriptions into contextual embeddings.",
        "Skill Gap Analysis":
        "Each user skill is compared against role tokens using exact and semantic similarity scoring.",
        "Cosine Similarity":
        "Cosine similarity measures angular distance between embedding vectors.",
        "Database":
        "User credentials are securely stored in SQLite using SHA-256 hashing.",
        "Explainability":
        "Each skill’s individual contribution is calculated and displayed dynamically."
    }

    for title, text in sections.items():
        st.markdown(f"<div class='card'><b>{title}</b><br>{text}</div>",
                    unsafe_allow_html=True)

# ============================================================
# AI ASSISTANT
# ============================================================
elif st.session_state.page == "AI":

    st.title("🤖 Career AI Assistant")

    question = st.text_input("Ask a career-related question")

    if question:
        try:
            genai.configure(api_key=st.secrets["AI_API_KEY"])
            model_ai = genai.GenerativeModel("gemini-pro")
            response = model_ai.generate_content(question)
            st.markdown(response.text)
        except Exception:
            st.error("AI service unavailable. Check API key.")