import streamlit as st
import pandas as pd
import sqlite3
import hashlib
import pdfplumber
import plotly.graph_objects as go
import yagmail
import google.generativeai as genai
import numpy as np
import pickle
import random
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# --- CONFIG & SECRETS ---
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
EMAIL_USER = st.secrets["EMAIL_USER"]
EMAIL_PASS = st.secrets["EMAIL_PASSWORD"]

genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel('gemini-2.5-flash')

# --- UI OVERRIDE ---
st.set_page_config(page_title="Resume AI Pro", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #050505; color: #e0e0e0; }
    div.stButton > button {
        background-color: #111; color: white; border: 2px solid #333;
        border-radius: 12px; width: 100%; height: 55px; font-weight: 600;
        transition: 0.3s; white-space: nowrap;
    }
    div.stButton > button:hover {
        border-color: #6366f1; box-shadow: 0 0 20px rgba(99, 102, 241, 0.6);
        color: #6366f1; transform: scale(1.02);
    }
    .glow-text {
        color: #ffffff; text-shadow: 0 0 10px #6366f1, 0 0 20px #6366f1;
        text-align: center; font-weight: 800; font-size: 2.5rem; margin-bottom: 20px;
    }
    .info-card {
        background: #111; padding: 20px; border-radius: 15px;
        border: 1px solid #333; transition: 0.5s; margin-bottom: 15px;
    }
    .badge { display: inline-block; padding: 4px 10px; border-radius: 6px; margin: 3px; font-size: 0.8rem; font-weight: bold; }
    .matched { background: #064e3b; color: #10b981; border: 1px solid #10b981; }
    .missing { background: #450a0a; color: #f87171; border: 1px solid #f87171; }
</style>
""", unsafe_allow_html=True)

# --- DATABASE LOGIC ---
conn = sqlite3.connect("users.db", check_same_thread=False)
c = conn.cursor()
c.execute("CREATE TABLE IF NOT EXISTS users(username TEXT UNIQUE, email TEXT UNIQUE, password TEXT)")
conn.commit()

def hash_data(data): return hashlib.sha256(data.encode()).hexdigest()

def send_otp_email(receiver_email):
    otp = str(random.randint(100000, 999999))
    try:
        yag = yagmail.SMTP(EMAIL_USER, EMAIL_PASS)
        yag.send(to=receiver_email, subject="Your OTP Verification", contents=f"Your OTP is {otp}")
        return hash_data(otp)
    except Exception as e:
        st.error(f"Failed to send email: {e}")
        return None

# --- LOAD ASSETS ---
@st.cache_resource
def load_ml_assets():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    job_embeddings = np.load("job_embeddings.npy")
    # Expanded Skillset for better matching
    jd_list = [
        {"Role": "Software Engineer", "Skills": ["Python", "C++", "DSA", "Git", "SQL", "APIs", "Java", "Docker", "Linux"]},
        {"Role": "Data Scientist", "Skills": ["Python", "Machine Learning", "NLP", "Pandas", "Math", "Statistics", "Scikit-Learn", "Matplotlib"]},
        {"Role": "Full Stack Developer", "Skills": ["React", "Node.js", "HTML", "CSS", "JS", "MongoDB", "Express", "TypeScript"]},
        {"Role": "ML Engineer", "Skills": ["PyTorch", "TensorFlow", "Keras", "Scikit-Learn", "CNN", "Deep Learning", "Deployment"]},
        {"Role": "Cloud Architect", "Skills": ["AWS", "Azure", "Docker", "Kubernetes", "Linux", "Terraform", "Cloud Security"]}
    ]
    return model, job_embeddings, pd.DataFrame(jd_list)

st_model, jd_embeddings, jd_df = load_ml_assets()

# --- AUTHENTICATION ---
if "user" not in st.session_state: st.session_state.user = None
if "menu" not in st.session_state: st.session_state.menu = "Home"
if "signup_otp" not in st.session_state: st.session_state.signup_otp = None

if st.session_state.user is None:
    st.markdown("<h1 class='glow-text'>RESUME AI-MATCH</h1>", unsafe_allow_html=True)
    t1, t2, t3 = st.tabs(["Login", "Sign Up", "Forgot Password"])
    
    with t1:
        login_id = st.text_input("Username or Email", key="l_id").strip()
        lp = st.text_input("Password", type="password", key="lp")
        if st.button("Login"):
            c.execute("SELECT username FROM users WHERE (email=? OR username=?) AND password=?", (login_id, login_id, hash_data(lp)))
            res = c.fetchone()
            if res: st.session_state.user = res[0]; st.rerun()
            else: st.error("Invalid credentials.")

    with t2:
        su = st.text_input("Username", key="su").strip()
        se = st.text_input("Email", key="se").lower().strip()
        sp = st.text_input("Password", type="password", key="sp")
        if st.button("Send Verification OTP"):
            if su and se and sp:
                st.session_state.signup_otp = send_otp_email(se)
                if st.session_state.signup_otp: st.success("OTP sent!")
            else: st.warning("Fill all fields.")
        
        otp_in = st.text_input("Enter OTP", key="otp_reg")
        if st.button("Verify & Register"):
            if st.session_state.signup_otp and hash_data(otp_in) == st.session_state.signup_otp:
                try:
                    c.execute("INSERT INTO users VALUES (?,?,?)", (su, se, hash_data(sp)))
                    conn.commit(); st.success("Created! Please Login.")
                except: st.error("User exists.")
            else: st.error("Invalid OTP.")

# --- MAIN APP ---
else:
    with st.sidebar:
        st.markdown(f"<h2 class='glow-text'>{st.session_state.user}</h2>", unsafe_allow_html=True)
        if st.button("🏠 Home"): st.session_state.menu = "Home"
        if st.button("💰 Salary Predictor"): st.session_state.menu = "Salary"
        if st.button("📊 Analytics"): st.session_state.menu = "Analytics"
        if st.button("🤖 AI Assistant"): st.session_state.menu = "AI"
        if st.button("ℹ️ About"): st.session_state.menu = "About"
        st.divider()
        if st.button("🚪 Logout"): st.session_state.user = None; st.rerun()

    if st.session_state.menu == "Home":
        st.markdown("<h1 class='glow-text'>RESUME MATCHING</h1>", unsafe_allow_html=True)
        file = st.file_uploader("Upload Resume", type="pdf")
        if file:
            with pdfplumber.open(file) as pdf:
                text = " ".join([p.extract_text() for p in pdf.pages if p.extract_text()]).strip()
            st.session_state.resume_text = text
            
            u_emb = st_model.encode([text])
            raw_scores = cosine_similarity(u_emb, jd_embeddings)[0]
            top_idx = raw_scores.argsort()[::-1]

            st.write("### TOP JOB MATCHES")
            cols = st.columns(3)
            for i in range(min(3, len(jd_df))):
                idx = int(top_idx[i]) % len(jd_df)
                role, skills = jd_df.iloc[idx]['Role'], jd_df.iloc[idx]['Skills']
                
                matched = [s for s in skills if s.lower() in text.lower()]
                missing = [s for s in skills if s.lower() not in text.lower()]
                
                # BUG FIX: Hard-floor logic. If no skills match, score is 0.
                final_score = raw_scores[top_idx[i]] * 100 if len(matched) > 0 else 0.0
                
                with cols[i]:
                    st.markdown(f"""
                    <div class='info-card'>
                        <h3 style='color:#6366f1;'>{role}</h3>
                        <h2 style='color:#10b981;'>{final_score:.1f}%</h2>
                        <p><b>Matched:</b> {' '.join([f'<span class="badge matched">{s}</span>' for s in matched]) if matched else 'None'}</p>
                        <p><b>Missing:</b> {' '.join([f'<span class="badge missing">{s}</span>' for s in missing])}</p>
                    </div>
                    """, unsafe_allow_html=True)

            st.divider()
            try:
                with st.spinner("Gemini 2.5 Analysis..."):
                    prompt = f"Analyze this resume for technical strengths: {text[:1000]}"
                    response = gemini_model.generate_content(prompt).text
                    st.info(response)
                    st.download_button("Download AI Explanation", response, file_name="analysis.txt")
            except: st.error("AI Quota exceeded.")

    elif st.session_state.menu == "Salary":
        st.markdown("<h1 class='glow-text'>SALARY PREDICTOR</h1>", unsafe_allow_html=True)
        if st.session_state.resume_text:
            exp = st.number_input("Years of Experience", 0, 40, 2)
            if st.button("Predict"):
                try:
                    res = gemini_model.generate_content(f"Predict 2026 salary for: {st.session_state.resume_text[:500]} with {exp} years exp in USD/INR.")
                    st.success(res.text)
                except: st.error("AI Quota reached.")
        else: st.warning("Upload resume first.")

    elif st.session_state.menu == "Analytics":
        st.markdown("<h1 class='glow-text'>SKILL ANALYSIS</h1>", unsafe_allow_html=True)
        if st.session_state.resume_text:
            t = st.session_state.resume_text.lower()
            vals = [sum(1 for s in ["python", "c++", "java", "javascript"] if s in t), 
                    sum(1 for s in ["ml", "nlp", "ai", "tensor", "pytorch"] if s in t), 
                    sum(1 for s in ["sql", "git", "aws", "docker"] if s in t)]
            fig = go.Figure(data=[go.Pie(labels=["Coding", "ML", "Tools"], values=vals, hole=.4, pull=[0.05, 0.05, 0.05], marker=dict(line=dict(color='#000000', width=2)))])
            fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig)

    elif st.session_state.menu == "AI":
        st.markdown("<h1 class='glow-text'>AI ASSISTANT</h1>", unsafe_allow_html=True)
        q = st.chat_input("Ask career questions...")
        if q:
            with st.chat_message("user"): st.write(q)
            try:
                res = gemini_model.generate_content(q)
                with st.chat_message("assistant"): st.write(res.text)
            except: st.error("Quota exceeded.")

    elif st.session_state.menu == "About":
        st.markdown("<h1 class='glow-text'>CORE CONCEPTS</h1>", unsafe_allow_html=True)
        concepts = [("Skill Gap", "Finding missing competencies."), ("Cosine Similarity", "Vector matching."), ("Transformers", "Context NLP."), ("SHA-256", "Secure OTP."), ("SQLite", "User DB.")]
        cols = st.columns(3)
        for i, (t, d) in enumerate(concepts):
            with cols[i % 3]: st.markdown(f"<div class='info-card'><h4 style='color:#6366f1;'>{t}</h4><p>{d}</p></div>", unsafe_allow_html=True)