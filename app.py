import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import hashlib
import smtplib
import random
import google.generativeai as genai
from email.message import EmailMessage
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ==================================================
# CONFIG & DYNAMIC STYLING
# ==================================================
st.set_page_config(page_title="Resume–JD Matcher", layout="wide")

def apply_style(accent_color):
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700&display=swap');
    
    .stApp {{
        background: linear-gradient(-45deg, #0f172a, #1e1b4b, #020617, #1e293b);
        background-size: 400% 400%;
        animation: gradient 15s ease infinite;
        transition: all 0.5s ease;
    }}
    @keyframes gradient {{
        0% {{ background-position: 0% 50%; }}
        50% {{ background-position: 100% 50%; }}
        100% {{ background-position: 0% 50%; }}
    }}
    .glowing-title {{
        font-size: 3.5rem; font-weight: 800; text-align: center; color: #fff;
        text-shadow: 0 0 15px {accent_color}, 0 0 30px {accent_color};
        margin-bottom: 2rem; font-family: 'Inter', sans-serif;
    }}
    [data-testid="stTextInput"], [data-testid="stPasswordInput"], [data-testid="stTextArea"] {{
        max-width: 500px; margin: 0 auto;
    }}
    div.stButton > button {{
        border-radius: 10px; font-weight: bold; width: 100%; transition: 0.3s;
    }}
    .login-btn button {{ background-color: #22c55e !important; color: white !important; border: none; }}
    .signup-btn button {{ background-color: #3b82f6 !important; color: white !important; border: none; }}
    
    .info-card {{
        background: rgba(255, 255, 255, 0.05); padding: 25px; border-radius: 15px;
        border: 1px solid rgba(255,255,255,0.1); margin-bottom: 20px;
        transition: all 0.4s ease; color: #e2e8f0;
    }}
    .info-card:hover {{
        transform: translateX(20px); background: rgba(255, 255, 255, 0.1);
        border-color: {accent_color}; box-shadow: -10px 0 15px {accent_color};
    }}
    .skill-pill {{
        display: inline-block; padding: 6px 12px; border-radius: 20px;
        margin: 4px; font-size: 0.8rem; font-weight: 600;
    }}
    .match {{ background: rgba(16, 185, 129, 0.2); color: #10b981; border: 1px solid #10b981; }}
    .gap {{ background: rgba(239, 68, 68, 0.2); color: #ef4444; border: 1px solid #ef4444; }}
    </style>
    """, unsafe_allow_html=True)

# ==================================================
# DATABASE & UTILS
# ==================================================
def init_db():
    conn = sqlite3.connect("users.db")
    conn.execute("CREATE TABLE IF NOT EXISTS users(username TEXT PRIMARY KEY, password TEXT, email TEXT)")
    conn.commit()
    conn.close()

def hash_pw(p): return hashlib.sha256(p.encode()).hexdigest()

init_db()

def send_otp(receiver_email):
    otp = str(random.randint(100000, 999999))
    try:
        sender = st.secrets["EMAIL_USER"]
        pw = st.secrets["EMAIL_PASSWORD"]
        msg = EmailMessage()
        msg.set_content(f"Your verification code is: {otp}")
        msg['Subject'] = 'Verification Code'
        msg['From'] = sender
        msg['To'] = receiver_email
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(sender, pw)
            smtp.send_message(msg)
        return otp
    except Exception as e:
        st.error(f"Mail Error: {e}")
        return None

# ==================================================
# CORE LOGIC
# ==================================================
@st.cache_resource
def load_resources():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    try:
        df = pd.read_csv("jobs_processed.csv")
        job_emb = np.load("job_embeddings.npy")
    except:
        df = pd.DataFrame({'Job Title': ['Software Engineer'], 'clean_description': ['python java sql react']})
        job_emb = model.encode(df['clean_description'].tolist())
    
    try:
        with open("skills.txt", "r") as f:
            skills = [s.strip().lower() for s in f if s.strip()]
    except:
        skills = ["python", "java", "react", "sql", "machine learning"]
    return model, df, job_emb, skills

model, df, job_emb, skills_list = load_resources()

# ==================================================
# SESSION STATE & AUTH FLOW
# ==================================================
if "auth_step" not in st.session_state: st.session_state.auth_step = "login"
if "logged_in" not in st.session_state: st.session_state.logged_in = False

st.markdown('<h1 class="glowing-title">RESUME-JD MATCHER</h1>', unsafe_allow_html=True)

if not st.session_state.logged_in:
    apply_style("#ef4444")
    _, center_col, _ = st.columns([1, 2, 1])
    
    with center_col:
        if st.session_state.auth_step == "login":
            st.subheader("🔐 Login")
            u = st.text_input("Username", key="l_u")
            p = st.text_input("Password", type="password", key="l_p")
            
            st.markdown('<div class="login-btn">', unsafe_allow_html=True)
            if st.button("Login"):
                conn = sqlite3.connect("users.db")
                c = conn.cursor()
                c.execute("SELECT * FROM users WHERE username=? AND password=?", (u, hash_pw(p)))
                if c.fetchone():
                    st.session_state.logged_in = True
                    st.rerun()
                else: st.error("Wrong credentials.")
                conn.close()
            st.markdown('</div>', unsafe_allow_html=True)

            if st.button("Forgot Password?"):
                st.session_state.auth_step = "forgot_req"
                st.rerun()
            
            st.markdown('<div class="signup-btn">', unsafe_allow_html=True)
            if st.button("New User? Sign Up"):
                st.session_state.auth_step = "signup"
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

        elif st.session_state.auth_step == "signup":
            st.subheader("📝 Sign Up")
            nu = st.text_input("Username")
            ne = st.text_input("Email")
            npw = st.text_input("Password", type="password")
            if st.button("Send OTP"):
                conn = sqlite3.connect("users.db")
                c = conn.cursor()
                c.execute("SELECT username FROM users WHERE username=?", (nu,))
                if c.fetchone():
                    st.error("Username already taken.")
                    conn.close()
                else:
                    conn.close()
                    otp = send_otp(ne)
                    if otp:
                        st.session_state.otp = otp
                        st.session_state.temp_user = (nu, hash_pw(npw), ne)
                        st.session_state.auth_step = "verify_signup"
                        st.rerun()

        elif st.session_state.auth_step == "verify_signup":
            v = st.text_input("Enter OTP from Gmail")
            if st.button("Verify & Sign Up"):
                if v == st.session_state.otp:
                    conn = sqlite3.connect("users.db")
                    try:
                        conn.execute("INSERT INTO users VALUES (?,?,?)", st.session_state.temp_user)
                        conn.commit()
                        st.success("Account Created! Please Login.")
                        st.session_state.auth_step = "login"
                        st.rerun()
                    except: st.error("Signup failed.")
                    finally: conn.close()
                else: st.error("Invalid OTP.")

# ==================================================
# MAIN INTERFACE
# ==================================================
else:
    menu = st.sidebar.radio("Navigation", ["Home", "About", "AI Assistant"])
    
    if menu == "Home":
        apply_style("#22c55e")
        st.subheader("🎯 Skill Recommendation")
        user_in = st.text_area("Enter Resume Skills", height=200)
        if st.button("Find Match"):
            if user_in:
                u_vec = model.encode([user_in])
                sims = cosine_similarity(u_vec, job_emb)[0]
                best = df.iloc[np.argmax(sims)]
                st.markdown(f"### Role: {best['Job Title']}")
                st.metric("Score", f"{round(np.max(sims)*100, 1)}%")
                
                u_sk = {s for s in skills_list if s in user_in.lower()}
                j_sk = {s for s in skills_list if s in best['clean_description'].lower()}
                matched, gap = u_sk & j_sk, j_sk - u_sk
                
                c1, c2 = st.columns(2)
                with c1: st.markdown("✅ **Matched**<br>" + " ".join([f"<span class='skill-pill match'>{s}</span>" for s in matched]), unsafe_allow_html=True)
                with c2: st.markdown("💡 **Gaps**<br>" + " ".join([f"<span class='skill-pill gap'>{s}</span>" for s in gap]), unsafe_allow_html=True)

    elif menu == "About":
        apply_style("#3b82f6")
        st.subheader("📘 System Components")
        cards = [
            ("🏗️ Transformers", "Semantic understanding using SBERT."),
            ("📉 Cosine Similarity", "Vector distance calculation."),
            ("🔍 Skill Gap", "Automated role comparison."),
            ("🔐 Database", "SQLite3 with SHA-256 security.")
        ]
        for t, d in cards:
            st.markdown(f'<div class="info-card"><h3>{t}</h3><p>{d}</p></div>', unsafe_allow_html=True)

    elif menu == "AI Assistant":
        apply_style("#a855f7")
        st.subheader("🤖 Career Coach")
        q = st.text_input("Ask anything:")
        if q:
            try:
                # REFINED CONFIG: Use the full model path to avoid 404 version errors
                genai.configure(api_key=st.secrets["AI_API_KEY"])
                ai = genai.GenerativeModel(model_name='gemini-1.5-flash')
                with st.spinner("Analyzing..."):
                    st.chat_message("assistant").write(ai.generate_content(q).text)
            except Exception as e:
                st.error(f"Gemini API Error: {e}")

    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.rerun()