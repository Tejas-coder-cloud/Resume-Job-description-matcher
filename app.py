import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import hashlib
import time

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ==================================================
# PAGE CONFIG
# ==================================================
st.set_page_config(
    page_title="Resume–JD Matching System",
    layout="wide",
    initial_sidebar_state="collapsed"
)

MATCH_THRESHOLD = 30  # % threshold for showing skills

# ==================================================
# THEMES
# ==================================================
THEMES = {
    "Auth": (
        "linear-gradient(-45deg, #020617, #4c1d95, #6d28d9, #020617)",
        "#c4b5fd"
    ),
    "Home": (
        "linear-gradient(-45deg, #020617, #312e81, #4f46e5, #020617)",
        "#a5b4fc"
    ),
    "About": (
        "linear-gradient(-45deg, #022c22, #065f46, #047857, #022c22)",
        "#6ee7b7"
    ),
    "AI": (
        "linear-gradient(-45deg, #3b0a0a, #9a3412, #ea580c, #3b0a0a)",
        "#fdba74"
    ),
}

def inject_css(gradient, accent):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: {gradient};
            background-size: 400% 400%;
            animation: gradientBG 18s ease infinite;
        }}
        @keyframes gradientBG {{
            0% {{ background-position: 0% 50%; }}
            50% {{ background-position: 100% 50%; }}
            100% {{ background-position: 0% 50%; }}
        }}
        .glass {{
            background: rgba(15,23,42,0.75);
            backdrop-filter: blur(18px);
            padding: 2rem;
            border-radius: 18px;
        }}
        .skill-box {{
            display: inline-flex;
            padding: 6px 14px;
            margin: 6px 6px 6px 0;
            border-radius: 999px;
            font-size: 0.85rem;
            font-weight: 600;
        }}
        .matched {{
            background: rgba(34,197,94,0.15);
            border: 1px solid #22c55e;
            color: #86efac;
        }}
        .missing {{
            background: rgba(239,68,68,0.15);
            border: 1px solid #ef4444;
            color: #fca5a5;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# ==================================================
# DATABASE
# ==================================================
def init_db():
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password TEXT,
            email TEXT,
            verified INTEGER
        )
    """)
    conn.commit()
    conn.close()

def hash_pw(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()

def check_login(username, password):
    username = username.strip()
    password = password.strip()

    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute(
        "SELECT * FROM users WHERE username=? AND password=? AND verified=1",
        (username, hash_pw(password))
    )
    res = c.fetchone()
    conn.close()
    return res

def add_user(username, password, email):
    username = username.strip()
    password = password.strip()
    email = email.strip()

    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    try:
        c.execute(
            "INSERT INTO users VALUES (?,?,?,1)",
            (username, hash_pw(password), email)
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

# ==================================================
# NLP HELPERS
# ==================================================
def semantic_skill_match(text, skills, skill_emb, model, threshold=0.55):
    text_emb = model.encode(text)
    sims = cosine_similarity([text_emb], skill_emb)[0]
    return {
        skills[i]
        for i, score in enumerate(sims)
        if score >= threshold
    }

@st.cache_resource
def load_resources():
    model = SentenceTransformer("all-MiniLM-L6-v2")

    df = pd.read_csv("jobs_processed.csv")
    job_emb = np.load("job_embeddings.npy")

    with open("skills.txt") as f:
        skills = [s.strip().lower() for s in f if s.strip()]

    skill_emb = model.encode(skills)

    with open("knowledge_base.txt") as f:
        kb = [p.strip() for p in f.read().split("\n\n") if p.strip()]

    kb_emb = model.encode(kb)

    return model, df, job_emb, skills, skill_emb, kb, kb_emb

# ==================================================
# INIT
# ==================================================
init_db()
st.session_state.setdefault("logged_in", False)
st.session_state.setdefault("page", "🏠 Home")

# ==================================================
# AUTH PAGE
# ==================================================
if not st.session_state.logged_in:
    inject_css(*THEMES["Auth"])

    st.markdown("<div class='glass' style='width:420px;margin:auto;'>", unsafe_allow_html=True)
    st.title("Resume–JD Matcher")

    mode = st.radio("", ["Login", "Sign Up"], horizontal=True)

    if mode == "Login":
        u = st.text_input("Username").strip()
        p = st.text_input("Password", type="password").strip()

        if st.button("Login", type="primary"):
            if check_login(u, p):
                st.session_state.logged_in = True
                st.rerun()
            else:
                st.error("Invalid username or password")

    else:
        u = st.text_input("Username").strip()
        e = st.text_input("Email").strip()
        p = st.text_input("Password", type="password").strip()

        if st.button("Register", type="primary"):
            if add_user(u, p, e):
                st.success("Account created. Login now.")
            else:
                st.error("Username already exists")

    st.markdown("</div>", unsafe_allow_html=True)

# ==================================================
# DASHBOARD
# ==================================================
else:
    model, df, job_emb, skills, skill_emb, kb, kb_emb = load_resources()

    with st.sidebar:
        st.radio("Navigation", ["🏠 Home", "ℹ️ About", "🤖 AI Assistant"], key="page")
        if st.button("Logout"):
            st.session_state.logged_in = False
            st.rerun()

    # ---------------- HOME ----------------
    if st.session_state.page == "🏠 Home":
        inject_css(*THEMES["Home"])
        st.title("AI Resume Scanner")

        resume = st.text_area("Paste resume text here")

        if st.button("Analyze Resume", type="primary"):
            r_emb = model.encode(resume)
            sims = cosine_similarity([r_emb], job_emb)[0]

            df["match_percentage"] = (sims * 100).round(2)
            top = df.sort_values("match_percentage", ascending=False).head(5)

            resume_skills = semantic_skill_match(resume, skills, skill_emb, model)

            for _, row in top.iterrows():
                with st.expander(f"{row['Job Title']} — {row['match_percentage']}%"):

                    if row["match_percentage"] < MATCH_THRESHOLD:
                        st.warning("Match score too low to infer skills.")
                        continue

                    jd_skills = semantic_skill_match(
                        row["clean_description"], skills, skill_emb, model
                    )

                    matched = resume_skills & jd_skills
                    missing = jd_skills - resume_skills

                    st.subheader("Matched Skills")
                    if matched:
                        st.markdown(
                            "".join(f"<span class='skill-box matched'>{s}</span>" for s in sorted(matched)),
                            unsafe_allow_html=True
                        )
                    else:
                        st.info("No strong matches detected.")

                    st.subheader("Skill Gap")
                    if missing:
                        st.markdown(
                            "".join(f"<span class='skill-box missing'>{s}</span>" for s in sorted(missing)),
                            unsafe_allow_html=True
                        )
                    else:
                        st.success("No major skill gaps.")

    # ---------------- ABOUT ----------------
    elif st.session_state.page == "ℹ️ About":
        inject_css(*THEMES["About"])
        st.title("About the Project")

        st.write("""
        This system uses transformer-based semantic embeddings to compare resumes
        with job descriptions and identify skill gaps reliably.
        """)

    # ---------------- AI ASSISTANT ----------------
    else:
        inject_css(*THEMES["AI"])
        st.title("AI Assistant")

        q = st.text_input("Ask about the project")
        if st.button("Ask", type="primary"):
            q_emb = model.encode(q)
            sims = cosine_similarity([q_emb], kb_emb)[0]
            if sims.max() > 0.45:
                st.success(kb[sims.argmax()])
            else:
                st.warning("No answer found")