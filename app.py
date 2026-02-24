import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import hashlib
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ==================================================
# PAGE CONFIG
# ==================================================
st.set_page_config(page_title="Resume–JD Matcher", layout="wide")

# ==================================================
# DATABASE
# ==================================================
def init_db():
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS users(
            username TEXT PRIMARY KEY,
            password TEXT,
            email TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()

# ==================================================
# LOAD MODEL & DATA
# ==================================================
@st.cache_resource
def load_resources():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    df = pd.read_csv("jobs_processed.csv")
    job_emb = np.load("job_embeddings.npy")
    with open("skills.txt") as f:
        skills = [s.strip().lower() for s in f if s.strip()]
    skill_emb = model.encode(skills)
    return model, df, job_emb, skills, skill_emb

model, df, job_emb, skills_list, skill_emb = load_resources()

# ==================================================
# SESSION STATE INIT
# ==================================================
if "page" not in st.session_state:
    st.session_state.page = "Home"
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# ==================================================
# SIDEBAR MENU (DO NOT OVERRIDE LOGIN)
# ==================================================
st.sidebar.markdown("## 📌 Menu")

nav_page = st.sidebar.radio(
    "Navigate",
    ["Home", "About", "AI Assistant"],
    index=["Home", "About", "AI Assistant"].index(
        st.session_state.page if st.session_state.page in ["Home","About","AI Assistant"] else "Home"
    )
)

# Update page ONLY if not on Login
if st.session_state.page != "Login":
    st.session_state.page = nav_page

# ==================================================
# DYNAMIC THEMES (NOW ACTUALLY WORK)
# ==================================================
THEME = {
    "Home": "#22c55e",
    "About": "#3b82f6",
    "AI Assistant": "#a855f7",
    "Login": "#ef4444"
}

accent = THEME[st.session_state.page]

st.markdown(f"""
<style>
.main {{
    background: radial-gradient(circle at top, #020617, #000000);
}}

.job-card {{
    background: #020617;
    padding:22px;
    border-radius:16px;
    border-left:6px solid {accent};
    margin-bottom:22px;
    transition:0.3s;
}}

.job-card:hover {{
    transform: scale(1.02);
}}

.skill {{
    display:inline-block;
    padding:6px 14px;
    margin:4px;
    border-radius:20px;
    font-size:0.8rem;
}}

.match {{ background:#064e3b; color:#6ee7b7; }}
.gap {{ background:#7f1d1d; color:#fca5a5; }}
</style>
""", unsafe_allow_html=True)

# ==================================================
# AUTH UTILS
# ==================================================
def hash_pwd(p):
    return hashlib.sha256(p.encode()).hexdigest()

# ==================================================
# SKILL EXTRACTION
# ==================================================
def semantic_skill_extract(text, threshold=0.45):
    chunks = re.split(r'\n|•|-|\.', text)
    chunks = [c.strip() for c in chunks if len(c.strip()) > 5]
    if not chunks:
        return set()
    vecs = model.encode(chunks)
    sims = cosine_similarity(vecs, skill_emb)
    found = set()
    for row in sims:
        for i, s in enumerate(row):
            if s > threshold:
                found.add(skills_list[i])
    return found

def fallback_skill_extract(text):
    text = text.lower()
    return {s for s in skills_list if re.search(rf"\b{s}\b", text)}

def render_skills(skills, css):
    return "".join(
        f"<span class='skill {css}'>{s}</span>" for s in sorted(skills)
    )

# ==================================================
# TOP LOGIN BUTTON
# ==================================================
_, login_col = st.columns([8,1])
with login_col:
    if st.button("Login"):
        st.session_state.page = "Login"

st.divider()

# ==================================================
# LOGIN PAGE (NOW STABLE)
# ==================================================
if st.session_state.page == "Login":
    st.subheader("🔐 Account Access")

    tab1, tab2, tab3 = st.tabs(["Login", "Sign Up", "Forgot Password"])

    with tab1:
        u = st.text_input("Username", key="login_user")
        p = st.text_input("Password", type="password", key="login_pass")
        if st.button("Login Now"):
            conn = sqlite3.connect("users.db")
            c = conn.cursor()
            c.execute("SELECT * FROM users WHERE username=? AND password=?",
                      (u, hash_pwd(p)))
            if c.fetchone():
                st.session_state.logged_in = True
                st.success("Login successful")
                st.session_state.page = "Home"
            else:
                st.error("Invalid credentials")
            conn.close()

    with tab2:
        nu = st.text_input("New Username", key="signup_user")
        ne = st.text_input("Email", key="signup_email")
        npw = st.text_input("Password", type="password", key="signup_pass")
        if st.button("Create Account"):
            conn = sqlite3.connect("users.db")
            c = conn.cursor()
            try:
                c.execute("INSERT INTO users VALUES (?,?,?)",
                          (nu, hash_pwd(npw), ne))
                conn.commit()
                st.success("Account created")
            except:
                st.error("Username already exists")
            conn.close()

    with tab3:
        st.text_input("Registered Email", key="forgot_email")
        st.info("Password reset is demo-only")

# ==================================================
# HOME PAGE
# ==================================================
elif st.session_state.page == "Home":
    st.subheader("📄 Semantic Resume Matcher")

    resume = st.text_area("Paste your resume here", height=220)

    if st.button("Analyze Match"):
        if resume.strip():
            res_vec = model.encode([resume])
            df["semantic"] = (cosine_similarity(res_vec, job_emb)[0] * 100).round(1)

            res_skills = semantic_skill_extract(resume) or fallback_skill_extract(resume)

            for _, row in df.sort_values("semantic", ascending=False).head(5).iterrows():
                jd = row["clean_description"]
                jd_skills = semantic_skill_extract(jd) or fallback_skill_extract(jd)

                matched = res_skills & jd_skills
                missing = jd_skills - res_skills

                final_score = min(row["semantic"], 15.0) if not matched else \
                    round(0.6 * row["semantic"] +
                          0.4 * (len(matched)/len(jd_skills))*100, 1)

                st.markdown(f"""
                <div class="job-card">
                    <h3>{row['Job Title']}</h3>
                    <b>Final Match Score:</b> {final_score}%
                </div>
                """, unsafe_allow_html=True)

                with st.expander("Skill Analysis"):
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown(render_skills(matched, "match"), unsafe_allow_html=True)
                    with c2:
                        st.markdown(render_skills(missing, "gap"), unsafe_allow_html=True)

# ==================================================
# ABOUT PAGE
# ==================================================
elif st.session_state.page == "About":
    st.subheader("📘 About This Project")
    st.markdown("""
    <div class="job-card">Semantic resume–JD matching using transformers</div>
    <div class="job-card">Hybrid scoring with skill overlap</div>
    <div class="job-card">No false positives — scores are capped</div>
    """, unsafe_allow_html=True)

# ==================================================
# AI ASSISTANT
# ==================================================
elif st.session_state.page == "AI Assistant":
    st.subheader("🤖 AI Assistant")
    q = st.text_input("Ask about the system")
    if q:
        st.success(
            "This assistant explains the system logic, scoring method, "
            "and skill extraction process used in this project."
        )