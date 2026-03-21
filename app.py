import re
import os
import streamlit as st
import sqlite3
import hashlib
import pdfplumber
import plotly.graph_objects as go
import google.generativeai as genai
import random
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# NEW IMPORTS
from langdetect import detect
from deep_translator import GoogleTranslator

# ---------------- CONFIG ---------------- #
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY_PRIMARY", "")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

st.set_page_config(page_title="Resume AI Matcher", layout="wide")

# ---------------- STYLE ---------------- #
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
    margin-bottom:15px;
    transition:0.3s;
}
.info-card:hover {
    transform:scale(1.02);
    box-shadow:0 0 25px rgba(99,102,241,0.6);
}
</style>
""", unsafe_allow_html=True)

# ---------------- DB ---------------- #
DB = os.path.join(os.getcwd(), "users.db")

def init_db():
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS users(
        username TEXT,
        email TEXT UNIQUE,
        password TEXT
    )
    """)
    conn.commit()
    conn.close()

init_db()

def hash_data(x):
    return hashlib.sha256(x.encode()).hexdigest()

# ---------------- MULTILINGUAL ---------------- #
def detect_lang(text):
    try:
        return detect(text)
    except:
        return "en"

def to_english(text):
    try:
        return GoogleTranslator(source='auto', target='en').translate(text)
    except:
        return text

def from_english(text, lang):
    try:
        return GoogleTranslator(source='en', target=lang).translate(text)
    except:
        return text

# ---------------- OTP ---------------- #
def generate_otp():
    otp = str(random.randint(100000, 999999))
    st.session_state.otp = otp
    return otp

# ---------------- AI ---------------- #
def ai_generate(prompt):
    for _ in range(3):
        try:
            model = genai.GenerativeModel("gemini-2.5-flash")
            return model.generate_content(prompt).text
        except:
            time.sleep(2)
    return "⚠️ AI unavailable"

# ---------------- UTIL ---------------- #
def word_in_text(word, text):
    return re.search(r"\b" + re.escape(word.lower()) + r"\b", text.lower()) is not None

# ---------------- JOB DATA ---------------- #
jobs = [
    {"role":"Software Engineer","skills":["python","java","c++","sql","git","dsa"]},
    {"role":"Data Scientist","skills":["python","ml","pandas","numpy"]},
    {"role":"Frontend Dev","skills":["react","js","html","css"]},
    {"role":"ML Engineer","skills":["tensorflow","pytorch"]},
]

skill_map = {
    "java":["Android Dev","Backend Dev"],
    "python":["Backend Dev","Data Scientist"],
    "react":["Frontend Dev"]
}

def infer_roles(text):
    roles=set()
    for k,v in skill_map.items():
        if word_in_text(k,text):
            roles.update(v)
    return list(roles)

# ---------------- EMBEDDING ---------------- #
vec = TfidfVectorizer()
jd_emb = vec.fit_transform([" ".join(j["skills"]) for j in jobs])

# ---------------- SESSION ---------------- #
if "user" not in st.session_state:
    st.session_state.user=None
if "resume" not in st.session_state:
    st.session_state.resume=""
if "otp" not in st.session_state:
    st.session_state.otp=None

# ---------------- LOGIN ---------------- #
if not st.session_state.user:

    st.markdown("<h1 class='glow-text'>RESUME AI MATCH</h1>", unsafe_allow_html=True)

    tab1,tab2 = st.tabs(["Login","Signup"])

    with tab1:
        email = st.text_input("Email", key="login_email")
        password = st.text_input("Password", type="password", key="login_pass")

        if st.button("Login"):
            hashed = hash_data(password)

            conn = sqlite3.connect(DB)
            cur = conn.cursor()
            cur.execute("SELECT username FROM users WHERE email=? AND password=?",
                        (email, hashed))
            res = cur.fetchone()
            conn.close()

            if res:
                st.session_state.user = res[0]
                st.success("Login successful")
                st.rerun()
            else:
                st.error("Invalid credentials")

    with tab2:
        user = st.text_input("Username", key="signup_user")
        email = st.text_input("Email", key="signup_email")
        password = st.text_input("Password", type="password", key="signup_pass")

        if st.button("Send OTP"):
            st.success(f"OTP: {generate_otp()}")

        otp_input = st.text_input("Enter OTP", key="otp_input")

        if st.button("Register"):
            if otp_input != st.session_state.otp:
                st.error("Wrong OTP")
            else:
                try:
                    conn = sqlite3.connect(DB)
                    cur = conn.cursor()
                    cur.execute("INSERT INTO users VALUES (?,?,?)",
                                (user,email,hash_data(password)))
                    conn.commit()
                    conn.close()
                    st.success("Account created")
                except:
                    st.error("User exists")

# ---------------- MAIN ---------------- #
else:
    st.sidebar.title(f"Hi, {st.session_state.user}")
    menu = st.sidebar.radio("Menu",["Home","Analytics","Salary","AI","About"])

    # -------- HOME -------- #
    if menu=="Home":
        st.markdown("<h1 class='glow-text'>UPLOAD RESUME</h1>", unsafe_allow_html=True)

        file = st.file_uploader("Upload PDF", type="pdf")

        if file:
            raw_text=""
            with pdfplumber.open(file) as pdf:
                for p in pdf.pages:
                    raw_text+=p.extract_text() or ""

            # 🌍 LANGUAGE DETECT
            lang = detect_lang(raw_text)

            # 🌍 TRANSLATE TO ENGLISH
            text = to_english(raw_text) if lang!="en" else raw_text
            text = text.lower()

            st.session_state.resume=text

            emb = vec.transform([text])
            sims = cosine_similarity(emb,jd_emb)[0]

            results=[]
            for i,job in enumerate(jobs):
                matched=[s for s in job["skills"] if word_in_text(s,text)]
                missing=[s for s in job["skills"] if not word_in_text(s,text)]

                sim=max(0,sims[i])
                skill=len(matched)/len(job["skills"])

                score=0 if len(matched)==0 else min((0.6*sim+0.4*skill)*100,100)

                results.append((job,score,matched,missing))

            best=max(results,key=lambda x:x[1])

            st.success(f"Best Match: {best[0]['role']} ({best[1]:.1f}%)")

            st.info(f"Suggested Roles: {infer_roles(text)}")

            for job,score,m,ms in sorted(results,key=lambda x:x[1],reverse=True):
                with st.expander(f"{job['role']} - {score:.1f}%"):
                    st.progress(score/100)
                    st.write("Matched:",m)
                    st.write("Missing:",ms)

            if st.button("AI Insights"):
                response = ai_generate(f"Analyze resume:\n{text[:1000]}")

                # 🌍 TRANSLATE BACK
                if lang!="en":
                    response = from_english(response, lang)

                st.info(response)

    # -------- ANALYTICS -------- #
    elif menu=="Analytics":
        st.markdown("<h1 class='glow-text'>SKILL CHART</h1>", unsafe_allow_html=True)

        t=st.session_state.resume
        if t:
            d={
                "Dev":sum(word_in_text(x,t) for x in ["python","java"]),
                "Web":sum(word_in_text(x,t) for x in ["html","css"]),
                "Data":sum(word_in_text(x,t) for x in ["ml","sql"])
            }

            fig=go.Figure(data=[go.Pie(
                labels=list(d.keys()),
                values=list(d.values()),
                hole=0.6,
                marker=dict(colors=["#6366f1","#f43f5e","#10b981"]),
                textinfo="label+percent"
            )])
            fig.update_layout(paper_bgcolor="#050505",font=dict(color="white"))

            st.plotly_chart(fig)

    # -------- SALARY -------- #
    elif menu=="Salary":
        st.markdown("<h1 class='glow-text'>SALARY ESTIMATOR</h1>", unsafe_allow_html=True)

        role = st.selectbox("Role",[j["role"] for j in jobs], key="salary_role")
        exp = st.slider("Experience",0,15,2, key="salary_exp")

        base={"Software Engineer":500000,"Data Scientist":600000,"Frontend Dev":450000,"ML Engineer":700000}
        salary=int(base[role]*(1+0.12*exp))

        st.metric("Estimated Salary",f"₹{salary:,}")

    # -------- AI -------- #
    elif menu=="AI":
        q = st.text_input("Ask something", key="ai_query")
        if q:
            st.write(ai_generate(q))

    # -------- ABOUT -------- #
    elif menu=="About":
        st.markdown("<h1 class='glow-text'>FEATURES</h1>", unsafe_allow_html=True)
        st.write("Multilingual Resume Support Added 🌍")