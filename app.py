import re
import streamlit as st
import sqlite3
import hashlib
import pdfplumber
import plotly.graph_objects as go
import yagmail
import google.generativeai as genai
import numpy as np
import random
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- CONFIG ---------------- #
GEMINI_API_KEY_PRIMARY = st.secrets.get("GEMINI_API_KEY_PRIMARY", "")
GEMINI_API_KEY_FALLBACK = st.secrets.get("GEMINI_API_KEY_FALLBACK", "")
EMAIL_USER = st.secrets.get("EMAIL_USER", "")
EMAIL_PASS = st.secrets.get("EMAIL_PASSWORD", "")

if GEMINI_API_KEY_PRIMARY:
    genai.configure(api_key=GEMINI_API_KEY_PRIMARY)

# ---------------- UI ---------------- #
st.set_page_config(page_title="Resume AI Matcher", layout="wide")

st.markdown("""
<style>
.stApp { background:#050505; color:#e0e0e0; }
.glow-text {
    text-align:center; 
    font-size:2.5rem; 
    color:white; 
    text-shadow:0 0 10px #6366f1,0 0 20px #6366f1;
    margin-bottom: 30px;
}
.info-card {
    background:#111; 
    padding:20px; 
    border-radius:15px; 
    border:1px solid #333; 
    transition:0.4s;
    margin-bottom: 15px;
}
.info-card:hover {
    transform:scale(1.02);
    box-shadow:0 0 25px rgba(99,102,241,0.5);
}
.badge {padding:5px 12px; margin:4px; border-radius:8px; display: inline-block; font-weight: bold;}
.match {background:#064e3b; color:#10b981; border: 1px solid #10b981;}
.miss {background:#450a0a; color:#f87171; border: 1px solid #f87171;}
</style>
""", unsafe_allow_html=True)

# ---------------- DATABASE ---------------- #
DB_PATH = "users.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("CREATE TABLE IF NOT EXISTS users(username TEXT UNIQUE, email TEXT UNIQUE, password TEXT)")
    conn.commit()
    conn.close()

init_db()

def hash_data(x):
    return hashlib.sha256(x.encode()).hexdigest()

def send_otp(email):
    otp = str(random.randint(100000, 999999))
    if EMAIL_USER and EMAIL_PASS:
        try:
            yagmail.SMTP(EMAIL_USER, EMAIL_PASS).send(email, "OTP Verification", f"Your OTP is: {otp}")
        except Exception:
            pass
    return otp, hash_data(otp)

def word_in_text(word, text):
    return re.search(r"\b" + re.escape(word.lower()) + r"\b", text.lower()) is not None

# ---------------- AI UTILITY ---------------- #
def genai_generate_with_fallback(prompt):
    models = ["gemini-2.5-flash", "gemini-2.5-pro"]
    keys = [GEMINI_API_KEY_PRIMARY, GEMINI_API_KEY_FALLBACK]
    
    for key in keys:
        if not key: continue
        genai.configure(api_key=key)
        for model_name in models:
            try:
                model_g = genai.GenerativeModel(model_name)
                response = model_g.generate_content(prompt)
                return response.text
            except Exception:
                continue
    return "AI insights currently unavailable. Check API keys."

# ---------------- RESOURCES ---------------- #
@st.cache_resource
def load_matcher():
    jobs = [
        {"role":"Software Engineer","skills":["python","java","c++","sql","git","dsa","api"]},
        {"role":"Data Scientist","skills":["python","ml","nlp","pandas","statistics","numpy"]},
        {"role":"Frontend Dev","skills":["react","js","html","css","ui"]},
        {"role":"ML Engineer","skills":["tensorflow","pytorch","deep learning","cnn"]},
        {"role":"Cloud Engineer","skills":["aws","docker","kubernetes","linux"]}
    ]
    texts = [" ".join(j["skills"]) for j in jobs]
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        emb = model.encode(texts)
        return model, jobs, emb
    except:
        vec = TfidfVectorizer(ngram_range=(1,2), stop_words='english')
        emb = vec.fit_transform(texts)
        return vec, jobs, emb

model, jobs, jd_emb = load_matcher()

# ---------------- APP STATE ---------------- #
if "user" not in st.session_state: st.session_state.user = None
if "resume" not in st.session_state: st.session_state.resume = ""
if "menu" not in st.session_state: st.session_state.menu = "Home"

# ---------------- AUTH UI ---------------- #
if not st.session_state.user:
    st.markdown("<h1 class='glow-text'>RESUME AI MATCH</h1>", unsafe_allow_html=True)
    t1, t2 = st.tabs(["Login", "Signup"])

    with t1:
        le = st.text_input("Email", key="l_email")
        lp = st.text_input("Password", type="password", key="l_pass")
        if st.button("Login", use_container_width=True):
            conn = sqlite3.connect(DB_PATH)
            cur = conn.cursor()
            cur.execute("SELECT * FROM users WHERE email=? AND password=?", (le, hash_data(lp)))
            res = cur.fetchone()
            conn.close()
            if res:
                st.session_state.user = le
                st.rerun()
            else:
                st.error("Invalid Email or Password")

    with t2:
        su = st.text_input("Username", key="s_user")
        se = st.text_input("Email", key="s_email")
        sp = st.text_input("Password", type="password", key="s_pass")
        
        if st.button("Get OTP"):
            if se:
                otp_code, otp_hash = send_otp(se)
                st.session_state.reg_otp_hash = otp_hash
                st.info(f"Verification Code: {otp_code}") # Visible for testing
            else: st.error("Email required")

        so = st.text_input("Enter OTP")
        if st.button("Register Account", use_container_width=True):
            if hash_data(so) == st.session_state.get("reg_otp_hash"):
                try:
                    conn = sqlite3.connect(DB_PATH)
                    cur = conn.cursor()
                    cur.execute("INSERT INTO users VALUES (?,?,?)", (su, se, hash_data(sp)))
                    conn.commit()
                    conn.close()
                    st.success("Registration Successful! Now go to the Login tab.")
                    time.sleep(0.5) # Give DB time to breath
                except sqlite3.IntegrityError:
                    st.error("User or Email already exists.")
            else: st.error("Incorrect OTP.")

# ---------------- DASHBOARD ---------------- #
else:
    st.sidebar.title(f"Hi, {st.session_state.user}")
    if st.sidebar.button("Home"): st.session_state.menu = "Home"
    if st.sidebar.button("Analytics"): st.session_state.menu = "Analytics"
    if st.sidebar.button("Salary"): st.session_state.menu = "Salary"
    if st.sidebar.button("AI Assistant"): st.session_state.menu = "AI"
    if st.sidebar.button("About"): st.session_state.menu = "About"
    if st.sidebar.button("Logout"):
        st.session_state.user = None
        st.rerun()

    menu = st.session_state.menu

    if menu == "Home":
        st.markdown("<h1 class='glow-text'>RESUME UPLOADER</h1>", unsafe_allow_html=True)
        up = st.file_uploader("Upload PDF", type="pdf")
        if up:
            text = ""
            with pdfplumber.open(up) as pdf:
                for p in pdf.pages: text += p.extract_text() or ""
            st.session_state.resume = text.lower()
            
            # Match Logic
            emb = model.encode([st.session_state.resume]) if hasattr(model, "encode") else model.transform([st.session_state.resume])
            sims = cosine_similarity(emb, jd_emb)[0]
            
            results = []
            for i, job in enumerate(jobs):
                m = [s for s in job["skills"] if word_in_text(s, text)]
                ms = [s for s in job["skills"] if not word_in_text(s, text)]
                score = (0.7 * sims[i] + 0.3 * (len(m)/len(job["skills"]))) * 100
                results.append((job, score, m, ms))
            
            for job, score, m, ms in sorted(results, key=lambda x: x[1], reverse=True):
                with st.expander(f"{job['role']} - {score:.1f}% Match"):
                    st.markdown(f"""<div class='info-card'>
                        <p><b>Matched:</b> {' '.join([f'<span class="badge match">{x}</span>' for x in m])}</p>
                        <p><b>Missing:</b> {' '.join([f'<span class="badge miss">{x}</span>' for x in ms])}</p>
                    </div>""", unsafe_allow_html=True)
            
            if st.button("Generate AI Insights"):
                with st.spinner("AI is reading your resume..."):
                    resp = genai_generate_with_fallback(f"Give 3 career tips for this resume:\n{text[:1500]}")
                    st.info(resp)

    elif menu == "Analytics":
        st.markdown("<h1 class='glow-text'>SKILL CHART</h1>", unsafe_allow_html=True)
        if st.session_state.resume:
            t = st.session_state.resume
            d = {"Dev": sum(word_in_text(x, t) for x in ["python","java","c++"]), 
                 "Web": sum(word_in_text(x, t) for x in ["html","css","js"]),
                 "Data": sum(word_in_text(x, t) for x in ["sql","ml","ai"])}
            fig = go.Figure(go.Pie(labels=list(d.keys()), values=list(d.values()), hole=.4))
            fig.update_layout(template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)
        else: st.warning("Upload resume first!")

    elif menu == "Salary":
        st.markdown("<h1 class='glow-text'>SALARY ESTIMATOR</h1>", unsafe_allow_html=True)
        r = st.selectbox("Role", [j["role"] for j in jobs])
        e = st.slider("Years", 0, 15, 2)
        base = {"Software Engineer": 500000, "Data Scientist": 600000, "Frontend Dev": 450000, "ML Engineer": 700000, "Cloud Engineer": 650000}
        val = int(base[r] * (1 + (e * 0.12)))
        st.metric("Estimated Salary", f"₹{val:,}")

    elif menu == "AI":
        st.markdown("<h1 class='glow-text'>AI CHAT</h1>", unsafe_allow_html=True)
        q = st.text_input("Ask about your career...")
        if q: st.write(genai_generate_with_fallback(q))

    elif menu == "About":
        st.markdown("<h1 class='glow-text'>FEATURES</h1>", unsafe_allow_html=True)
        feats = [("Skill Gap", "Analyzes missing keywords."), ("Semantic Match", "Deep context understanding."), 
                 ("Secure", "SHA-256 password hashing."), ("AI", "Gemini 1.5 integration.")]
        for t, d in feats:
            st.markdown(f"<div class='info-card'><h4>{t}</h4><p>{d}</p></div>", unsafe_allow_html=True)