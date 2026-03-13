# ============================================================
# Resume–JD Matcher | FINAL STABLE VERSION
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
st.set_page_config("Resume-JD Matcher", layout="wide")


# ------------------------------------------------------------
# DYNAMIC BACKGROUND FUNCTION
# ------------------------------------------------------------
def set_background(c1, c2):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: linear-gradient(-45deg,{c1},{c2},{c1});
            background-size: 400% 400%;
            animation: gradientBG 15s ease infinite;
            color:white;
        }}
        @keyframes gradientBG {{
            0% {{background-position:0% 50%;}}
            50% {{background-position:100% 50%;}}
            100% {{background-position:0% 50%;}}
        }}
        </style>
        """,
        unsafe_allow_html=True
    )


# ------------------------------------------------------------
# UI STYLE
# ------------------------------------------------------------
st.markdown("""
<style>
.card{
backdrop-filter: blur(12px);
background: rgba(255,255,255,0.05);
padding:25px;
border-radius:18px;
margin-bottom:25px;
transition:0.3s;
}
.card:hover{
transform:scale(1.05);
background:rgba(59,130,246,0.2);
box-shadow:0 0 25px rgba(59,130,246,0.6);
}
.chat-user{
background:rgba(37,99,235,0.3);
padding:10px;
border-radius:10px;
margin:6px 0;
}
.chat-ai{
background:rgba(124,58,237,0.3);
padding:10px;
border-radius:10px;
margin:6px 0;
}
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------
# GEMINI HELPER
# ------------------------------------------------------------
def get_ai_response(prompt):

    try:
        genai.configure(api_key=st.secrets["AI_API_KEY"])
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        return response.text

    except Exception as e:
        return f"AI error: {e}"


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
    conn.commit()
    conn.close()

init_db()


# ------------------------------------------------------------
# EMAIL OTP
# ------------------------------------------------------------
def send_otp(email):

    otp=str(random.randint(100000,999999))

    msg=EmailMessage()
    msg.set_content(f"Your OTP is: {otp}")
    msg["Subject"]="Verification Code"
    msg["From"]=st.secrets["EMAIL_USER"]
    msg["To"]=email

    with smtplib.SMTP_SSL("smtp.gmail.com",465) as s:
        s.login(st.secrets["EMAIL_USER"],st.secrets["EMAIL_PASSWORD"])
        s.send_message(msg)

    return otp


# ------------------------------------------------------------
# LOAD MODEL
# ------------------------------------------------------------
@st.cache_resource
def load_model():

    model=SentenceTransformer("all-MiniLM-L6-v2")

    descriptions={
    "Software Developer":
    "java python c++ backend system design algorithms data structures api rest git microservices",

    "Data Scientist":
    "python machine learning statistics pandas numpy deep learning sql r nlp",

    "Cloud Engineer":
    "aws docker kubernetes terraform infrastructure cloud monitoring",

    "DevOps Engineer":
    "ci cd docker kubernetes linux scripting terraform jenkins",

    "Web Developer":
    "html css javascript react node express mongodb frontend backend"
    }

    df=pd.DataFrame(list(descriptions.items()),columns=["Role","description"])
    embeddings=model.encode(df["description"].tolist())

    return model,df,embeddings

model,jobs_df,job_embeddings=load_model()


# ------------------------------------------------------------
# SESSION STATE
# ------------------------------------------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in=False

if "auth_step" not in st.session_state:
    st.session_state.auth_step="login"

if "page" not in st.session_state:
    st.session_state.page="Home"

if "chat" not in st.session_state:
    st.session_state.chat=[]


# ============================================================
# AUTH
# ============================================================
if not st.session_state.logged_in:

    st.title("Authentication")

    if st.session_state.auth_step=="login":

        with st.form("login"):

            username=st.text_input("Username")
            password=st.text_input("Password",type="password")

            submit=st.form_submit_button("Login")

        if submit:

            conn=sqlite3.connect("users.db")
            cur=conn.cursor()

            cur.execute("SELECT * FROM users WHERE username=? AND password=?",
            (username,hash_pw(password)))

            if cur.fetchone():

                st.session_state.logged_in=True
                st.rerun()

            else:
                st.error("Invalid credentials")

            conn.close()

        col1,col2=st.columns(2)

        if col1.button("New User? Sign Up"):
            st.session_state.auth_step="signup"
            st.rerun()

        if col2.button("Forgot Password"):
            st.session_state.auth_step="forgot"
            st.rerun()


    elif st.session_state.auth_step=="signup":

        user=st.text_input("Username")
        email=st.text_input("Email")
        pw=st.text_input("Password",type="password")

        if st.button("Send OTP"):

            st.session_state.otp=send_otp(email)
            st.session_state.temp=(user,hash_pw(pw),email)
            st.session_state.auth_step="verify"

            st.success("OTP sent")
            st.rerun()


    elif st.session_state.auth_step=="verify":

        otp=st.text_input("Enter OTP")

        if st.button("Verify"):

            if otp==st.session_state.otp:

                conn=sqlite3.connect("users.db")
                conn.execute("INSERT INTO users VALUES (?,?,?)",st.session_state.temp)
                conn.commit()
                conn.close()

                st.success("Account created")
                st.session_state.auth_step="login"

            else:
                st.error("Invalid OTP")


    elif st.session_state.auth_step=="forgot":

        user=st.text_input("Username")
        email=st.text_input("Email")

        if st.button("Send Reset OTP"):

            conn=sqlite3.connect("users.db")
            cur=conn.cursor()

            cur.execute("SELECT * FROM users WHERE username=? AND email=?",(user,email))
            if cur.fetchone():

                st.session_state.otp=send_otp(email)
                st.session_state.reset=(user,email)
                st.session_state.auth_step="reset"
                st.rerun()

            else:
                st.error("No matching user")


    elif st.session_state.auth_step=="reset":

        otp=st.text_input("OTP")
        newpw=st.text_input("New Password",type="password")

        if st.button("Reset"):

            if otp==st.session_state.otp:

                conn=sqlite3.connect("users.db")
                conn.execute("UPDATE users SET password=? WHERE username=?",
                (hash_pw(newpw),st.session_state.reset[0]))
                conn.commit()
                conn.close()

                st.success("Password updated")
                st.session_state.auth_step="login"

            else:
                st.error("Invalid OTP")

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.title("Menu")

if st.sidebar.button("Home"):
    st.session_state.page="Home"

if st.sidebar.button("About"):
    st.session_state.page="About"

if st.sidebar.button("AI Assistant"):
    st.session_state.page="AI"

if st.sidebar.button("Logout"):
    st.session_state.logged_in=False
    st.rerun()


# ============================================================
# HOME
# ============================================================
if st.session_state.page=="Home":

    set_background("#0f172a","#1e293b")

    st.title("Resume-JD Matcher")

    with st.form("resume"):

        exp=st.slider("Experience",0,20,0)
        resume=st.text_area("Paste Resume / Skills")

        submit=st.form_submit_button("Analyze")

    if submit:

        skills=list(set(re.findall(r"[a-zA-Z\+\#]{2,}",resume.lower())))

        resume_emb=model.encode([resume])
        sims=cosine_similarity(resume_emb,job_embeddings)[0]

        for i in sims.argsort()[::-1][:3]:

            role=jobs_df.iloc[i]["Role"]
            desc=jobs_df.iloc[i]["description"]
            role_tokens=desc.split()

            matched=[s for s in skills if s in role_tokens]
            unmatched=[s for s in skills if s not in role_tokens]

            score=sims[i]

            base_salary=3+score*3
            experience_bonus=exp*0.7
            salary=base_salary+experience_bonus

            prompt=f"""
You are a recruiter.

Role: {role}
Matched Skills: {matched}
Missing Skills: {unmatched}

Explain candidate suitability in 4 bullet points.
"""

            explanation=get_ai_response(prompt).replace("\n","<br>")

            st.markdown(f"""
            <div class="card">

            <h2>{role}</h2>

            Match Score: {score*100:.1f}%<br>
            Predicted Salary: ₹{salary:.1f} LPA

            <hr>

            <b>AI Explainability</b><br>
            {explanation}

            <br><br>

            <b>Matched Skills:</b> {', '.join(matched) if matched else "None"}<br>
            <b>Unmatched Skills:</b> {', '.join(unmatched) if unmatched else "None"}

            </div>
            """,unsafe_allow_html=True)


# ============================================================
# ABOUT
# ============================================================
elif st.session_state.page=="About":

    set_background("#1e3a8a","#0f172a")

    st.title("System Architecture")

    sections={
    "Transformer Architecture":
    "SentenceTransformer converts resumes and job descriptions into embeddings.",

    "Skill Gap Analysis":
    "Exact and semantic skill comparison identifies matches and gaps.",

    "Cosine Similarity":
    "Vector similarity determines job-role alignment.",

    "Database":
    "SQLite stores users securely with SHA256 hashed passwords.",

    "Explainability":
    "AI generates recruiter-style reasoning for matches."
    }

    for t,d in sections.items():

        st.markdown(f"""
        <div class="card">
        <b>{t}</b><br><br>{d}
        </div>
        """,unsafe_allow_html=True)


# ============================================================
# AI ASSISTANT
# ============================================================
elif st.session_state.page=="AI":

    set_background("#312e81","#0f172a")

    st.title("Career AI Assistant")

    for m in st.session_state.chat:

        st.markdown(f"<div class='chat-user'><b>You:</b> {m['u']}</div>",unsafe_allow_html=True)
        st.markdown(f"<div class='chat-ai'><b>AI:</b> {m['a']}</div>",unsafe_allow_html=True)

    with st.form("chat"):

        q=st.text_input("Ask a career question")

        send=st.form_submit_button("Send")

    if send and q:

        prompt=f"You are a career advisor. Answer: {q}"

        ans=get_ai_response(prompt)

        st.session_state.chat.append({"u":q,"a":ans})
        st.rerun()

    if st.button("Clear Chat History"):
        st.session_state.chat=[]
        st.rerun()