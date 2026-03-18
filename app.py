import streamlit as st
import sqlite3
import hashlib
import pdfplumber
import plotly.graph_objects as go
import yagmail
import google.generativeai as genai
import numpy as np
import random
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- CONFIG ---------------- #
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")
EMAIL_USER = st.secrets.get("EMAIL_USER", "")
EMAIL_PASS = st.secrets.get("EMAIL_PASSWORD", "")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

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
}

.info-card {
    background:#111;
    padding:20px;
    border-radius:15px;
    border:1px solid #333;
    transition:0.4s;
}
.info-card:hover {
    transform:scale(1.05);
    box-shadow:0 0 25px rgba(99,102,241,0.5);
}

.badge {padding:5px 10px; margin:3px; border-radius:8px;}
.match {background:#064e3b; color:#10b981;}
.miss {background:#450a0a; color:#f87171;}
</style>
""", unsafe_allow_html=True)

# ---------------- DB ---------------- #
conn = sqlite3.connect("users.db", check_same_thread=False)
c = conn.cursor()
c.execute("CREATE TABLE IF NOT EXISTS users(username TEXT, email TEXT, password TEXT)")
conn.commit()

def hash_data(x):
    return hashlib.sha256(x.encode()).hexdigest()

def send_otp(email):
    otp = str(random.randint(100000,999999))
    try:
        yagmail.SMTP(EMAIL_USER, EMAIL_PASS).send(email,"OTP",otp)
    except:
        st.write("OTP:",otp)
    return hash_data(otp)

# ---------------- MODEL ---------------- #
@st.cache_resource
def load():
    model = SentenceTransformer("all-MiniLM-L6-v2")

    jobs = [
        {"role":"Software Engineer","skills":["python","java","c++","sql","git","dsa","api"]},
        {"role":"Data Scientist","skills":["python","ml","nlp","pandas","statistics","numpy"]},
        {"role":"Frontend Dev","skills":["react","js","html","css","ui"]},
        {"role":"ML Engineer","skills":["tensorflow","pytorch","deep learning","cnn"]},
        {"role":"Cloud Engineer","skills":["aws","docker","kubernetes","linux"]}
    ]

    texts = [" ".join(j["skills"]) for j in jobs]
    emb = model.encode(texts)

    return model,jobs,emb

model,jobs,jd_emb = load()

# ---------------- SESSION ---------------- #
if "user" not in st.session_state:
    st.session_state.user=None
if "resume" not in st.session_state:
    st.session_state.resume=""

# ---------------- AUTH ---------------- #
if not st.session_state.user:
    st.markdown("<h1 class='glow-text'>RESUME AI MATCH</h1>", unsafe_allow_html=True)

    tab1,tab2=st.tabs(["Login","Signup"])

    with tab1:
        u=st.text_input("Email")
        p=st.text_input("Password",type="password")
        if st.button("Login"):
            c.execute("SELECT * FROM users WHERE email=? AND password=?", (u,hash_data(p)))
            if c.fetchone():
                st.session_state.user=u
                st.rerun()
            else:
                st.error("Invalid")

    with tab2:
        u=st.text_input("Email",key="su")
        p=st.text_input("Password",type="password",key="sp")

        if st.button("Send OTP"):
            st.session_state.otp=send_otp(u)

        otp=st.text_input("OTP")

        if st.button("Register"):
            if hash_data(otp)==st.session_state.get("otp"):
                c.execute("INSERT INTO users VALUES (?,?,?)",(u,u,hash_data(p)))
                conn.commit()
                st.success("Registered")
            else:
                st.error("Wrong OTP")

# ---------------- MAIN ---------------- #
else:
    menu=st.sidebar.radio("Menu",["Home","Analytics","AI","About"])

    if menu=="Home":
        st.markdown("<h1 class='glow-text'>UPLOAD RESUME</h1>", unsafe_allow_html=True)

        file=st.file_uploader("Upload PDF",type="pdf")

        if file:
            text=""
            with pdfplumber.open(file) as pdf:
                for p in pdf.pages:
                    text+=p.extract_text() or ""

            st.session_state.resume=text.lower()

            # ---------- MATCHING FIX ---------- #
            res_emb=model.encode([text])
            cos=cosine_similarity(res_emb,jd_emb)[0]

            results=[]
            for i,job in enumerate(jobs):
                skills=job["skills"]

                matched=[s for s in skills if s in text]
                missing=[s for s in skills if s not in text]

                skill_score=len(matched)/len(skills)

                # FINAL SCORE (REAL FIX)
                final_score = (0.7*cos[i] + 0.3*skill_score) * 100

                results.append((job,final_score,matched,missing))

            results=sorted(results,key=lambda x:x[1],reverse=True)

            cols=st.columns(3)
            for i in range(3):
                job,score,matched,missing=results[i]

                with cols[i]:
                    st.markdown(f"""
                    <div class="info-card">
                    <h3>{job['role']}</h3>
                    <h2>{score:.2f}%</h2>
                    <p><b>Matched:</b>{" ".join([f"<span class='badge match'>{m}</span>" for m in matched])}</p>
                    <p><b>Missing:</b>{" ".join([f"<span class='badge miss'>{m}</span>" for m in missing])}</p>
                    </div>
                    """, unsafe_allow_html=True)

            # ---------- AI FIX ---------- #
            report_text = "AI analysis unavailable."

            if GEMINI_API_KEY:
                try:
                    prompt = f"""
Detect resume language and respond in SAME language.

Give:
1. Detailed Analysis
2. Future Improvements

Resume:
{text[:1500]}
"""
                    model_g = genai.GenerativeModel("gemini-2.5-flash")
                    response = model_g.generate_content(prompt)

                    if response and hasattr(response, "text"):
                        report_text = response.text

                except Exception as e:
                    report_text = f"AI Error: {e}"

            st.info(report_text)

            # ALWAYS SHOW DOWNLOAD
            st.download_button(
                "Download Report",
                report_text,
                file_name="resume_report.txt"
            )

    elif menu=="Analytics":
        text=st.session_state.resume

        coding=sum(x in text for x in ["python","java","c++"])
        ml=sum(x in text for x in ["ml","ai","deep"])
        db=sum(x in text for x in ["sql","mongodb"])

        fig=go.Figure(data=[go.Scatter3d(
            x=[coding],y=[ml],z=[db],
            mode='markers',
            marker=dict(size=12,color='cyan')
        )])

        st.plotly_chart(fig)

    elif menu=="AI":
        q=st.text_input("Ask anything")
        if q and GEMINI_API_KEY:
            try:
                res=genai.GenerativeModel("gemini-2.5-flash").generate_content(q)
                st.write(res.text)
            except:
                st.error("AI error")

    elif menu=="About":
        cards=[
            ("Skill Gap","Find missing skills"),
            ("Cosine Similarity","Semantic matching"),
            ("Transformer","Context understanding"),
            ("OTP SHA256","Secure login"),
            ("Database","User storage")
        ]

        cols=st.columns(3)
        for i,(t,d) in enumerate(cards):
            with cols[i%3]:
                st.markdown(f"<div class='info-card'><h4>{t}</h4><p>{d}</p></div>",unsafe_allow_html=True)