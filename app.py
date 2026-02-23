import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import hashlib
import smtplib
import random
import time
import re

from email.mime.text import MIMEText
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(page_title="Resume–JD Matching System", layout="wide")

# --------------------------------------------------
# Database & Auth Logic
# --------------------------------------------------
def init_db():
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password TEXT,
            email TEXT,
            first_name TEXT,
            middle_name TEXT,
            last_name TEXT,
            verified INTEGER
        )
    """)
    conn.commit()
    conn.close()

def hash_pw(pw):
    return hashlib.sha256(pw.encode()).hexdigest()

def check_login(username, password):
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute(
        "SELECT * FROM users WHERE username=? AND password=? AND verified=1",
        (username, hash_pw(password))
    )
    res = c.fetchone()
    conn.close()
    return res

def add_user(username, password, email, fname):
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    try:
        c.execute(
            "INSERT INTO users VALUES (?,?,?,?,?,?,1)",
            (username, hash_pw(password), email, fname, "", "")
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def get_user_email(username):
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("SELECT email FROM users WHERE username=?", (username,))
    res = c.fetchone()
    conn.close()
    return res[0] if res else None

def update_password(username, new_pw):
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute(
        "UPDATE users SET password=? WHERE username=?",
        (hash_pw(new_pw), username)
    )
    conn.commit()
    conn.close()

def send_otp(to_email, otp, subject="Verification Code"):
    sender = st.secrets["EMAIL_USER"]
    password = st.secrets["EMAIL_PASS"]

    msg = MIMEText(f"Your OTP is: {otp}")
    msg["From"] = sender
    msg["To"] = to_email
    msg["Subject"] = subject

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender, password)
            server.sendmail(sender, to_email, msg.as_string())
        return True
    except:
        return False

# --------------------------------------------------
# Skill Extraction
# --------------------------------------------------
def extract_skills(text, skills):
    text = text.lower()
    found = set()
    for skill in skills:
        pattern = r"\b" + re.escape(skill) + r"\b"
        if re.search(pattern, text):
            found.add(skill)
    return found

# --------------------------------------------------
# Load Resources
# --------------------------------------------------
@st.cache_resource
def load_all_resources():
    model = SentenceTransformer("all-MiniLM-L6-v2", token=st.secrets.get("HF_TOKEN"))
    df = pd.read_csv("jobs_processed.csv")
    job_embeddings = np.load("job_embeddings.npy")

    with open("skills.txt", "r", encoding="utf-8") as f:
        skills = [s.strip().lower() for s in f if s.strip()]

    with open("knowledge_base.txt", "r", encoding="utf-8") as f:
        paragraphs = [p.strip() for p in f.read().split("\n\n") if p.strip()]

    kb_embeddings = model.encode(paragraphs)

    return model, df, job_embeddings, skills, paragraphs, kb_embeddings

# --------------------------------------------------
# Init App State
# --------------------------------------------------
init_db()

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "section" not in st.session_state:
    st.session_state.section = "Home"
if "has_analyzed" not in st.session_state:
    st.session_state.has_analyzed = False

# --------------------------------------------------
# AUTH UI
# --------------------------------------------------
if not st.session_state.logged_in:
    st.title("Resume–JD Matching System")

    u = st.text_input("Username")
    p = st.text_input("Password", type="password")

    if st.button("Login"):
        res = check_login(u, p)
        if res:
            st.session_state.logged_in = True
            st.session_state.user_name = res[3]
            st.rerun()
        else:
            st.error("Invalid credentials")

# --------------------------------------------------
# MAIN DASHBOARD
# --------------------------------------------------
else:
    model, df, job_embs, skills_list, kb_paragraphs, kb_embeddings = load_all_resources()

    # ---------------- Dynamic Background Config ----------------
    bg_configs = {
        "Home": {
            "gradient": "linear-gradient(-45deg, #020617, #4c1d95, #6d28d9, #020617)",
            "accent": "#c4b5fd"
        },
        "About": {
            "gradient": "linear-gradient(-45deg, #020617, #0f172a, #1e3a8a, #020617)",
            "accent": "#93c5fd"
        },
        "AI": {
            "gradient": "linear-gradient(-45deg, #020617, #7c2d12, #ea580c, #020617)",
            "accent": "#fdba74"
        }
    }

    conf = bg_configs.get(st.session_state.section, bg_configs["Home"])

    # ---------------- Inject Dynamic CSS ----------------
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: {conf['gradient']};
            background-size: 400% 400%;
            animation: gradientFlow 15s ease infinite;
            color: white;
        }}

        @keyframes gradientFlow {{
            0% {{ background-position: 0% 50%; }}
            50% {{ background-position: 100% 50%; }}
            100% {{ background-position: 0% 50%; }}
        }}

        [data-testid="stVerticalBlockBorderWrapper"] {{
            background: rgba(15, 23, 42, 0.85) !important;
            backdrop-filter: blur(18px);
            border-radius: 20px !important;
            border: 1px solid {conf['accent']}33 !important;
        }}

        h1, h2, h3 {{
            color: {conf['accent']};
            text-shadow: 0 0 12px {conf['accent']};
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

    # -------- Sidebar Navigation --------
    with st.sidebar:
        st.write(f"👋 Welcome {st.session_state.user_name}")
        if st.button("🏠 Home", use_container_width=True):
            st.session_state.section = "Home"
        if st.button("ℹ️ About", use_container_width=True):
            st.session_state.section = "About"
        if st.button("🤖 AI Assistant", use_container_width=True):
            st.session_state.section = "AI"
        st.markdown("---")
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.logged_in = False
            st.rerun()

    st.title("Resume–JD Matching System")

    # --------------------------------------------------
    # HOME
    # --------------------------------------------------
    if st.session_state.section == "Home":
        resume_text = st.text_area("Paste Resume Text", height=220)

        if st.button("Analyze"):
            if resume_text.strip():
                r_emb = model.encode(resume_text)
                sims = cosine_similarity([r_emb], job_embs)[0]

                df_local = df.copy()
                df_local["match_percentage"] = (sims * 100).round(2)

                st.session_state.user_skills = extract_skills(resume_text, skills_list)
                st.session_state.top_matches = (
                    df_local.sort_values("match_percentage", ascending=False)
                    .head(5)
                    .to_dict("records")
                )
                st.session_state.has_analyzed = True

        if st.session_state.has_analyzed:
            for row in st.session_state.top_matches:
                if row["match_percentage"] >= 15:
                    job_skills = extract_skills(row["clean_description"], skills_list)
                    matched = job_skills & st.session_state.user_skills
                    missing = job_skills - st.session_state.user_skills

                    with st.expander(f"{row['Job Title']} — {row['match_percentage']}%"):
                        st.write(row["clean_description"])
                        st.write("✅ Matched Skills:", ", ".join(matched) or "None")
                        st.write("❌ Missing Skills:", ", ".join(missing) or "None")

    # --------------------------------------------------
    # ABOUT
    # --------------------------------------------------
    elif st.session_state.section == "About":
        st.header("⚙️ Technology Overview")

        with st.expander("🔹 Transformer – MiniLM"):
            st.write(
                "The all-MiniLM-L6-v2 transformer converts resumes and job descriptions "
                "into semantic vectors using self-attention, enabling contextual matching."
            )

        with st.expander("🔹 Cosine Similarity"):
            st.write(
                "Cosine similarity measures angular distance between vectors, "
                "indicating semantic relevance rather than keyword overlap."
            )

        with st.expander("🔹 Skill Gap Analysis"):
            st.write(
                "Regex-based skill extraction identifies matched and missing skills, "
                "helping users optimize resumes for specific roles."
            )

    # --------------------------------------------------
    # AI ASSISTANT
    # --------------------------------------------------
    elif st.session_state.section == "AI":
        st.header("🤖 AI Assistant")
        q = st.text_input("Ask about the project")

        if st.button("Ask"):
            q_emb = model.encode(q)
            sims = cosine_similarity([q_emb], kb_embeddings)[0]
            if sims.max() > 0.45:
                st.info(kb_paragraphs[sims.argmax()])
            else:
                st.warning("No answer found.")