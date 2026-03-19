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
from sentence_transformers import SentenceTransformer
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
c.execute("CREATE TABLE IF NOT EXISTS users(username TEXT UNIQUE, email TEXT UNIQUE, password TEXT)")
conn.commit()

def hash_data(x):
    return hashlib.sha256(x.encode()).hexdigest()

def send_otp(email):
    otp = str(random.randint(100000,999999))
    if EMAIL_USER and EMAIL_PASS:
        try:
            yagmail.SMTP(EMAIL_USER, EMAIL_PASS).send(email,"OTP",otp)
        except Exception:
            st.warning("Unable to send email OTP. Use the code shown in UI for testing.")
            st.write("OTP:", otp)
    else:
        st.warning("Email not configured. Using fallback OTP for testing.")
        st.write("OTP:", otp)
    return otp, hash_data(otp)

def word_in_text(word, text):
    return re.search(r"\b" + re.escape(word.lower()) + r"\b", text.lower()) is not None

# Gemini request wrapper with retries and fallback

def genai_generate_with_fallback(prompt, model_names=None, max_retries=2):
    if model_names is None:
        model_names = ["groq-1.0", "gemini-2.5-flash", "gemini-3.5-pro"]

    for api_key in [GEMINI_API_KEY_PRIMARY, GEMINI_API_KEY_FALLBACK]:
        if not api_key:
            continue
        genai.configure(api_key=api_key)

        for model_name in model_names:
            for attempt in range(1, max_retries + 1):
                try:
                    model_g = genai.GenerativeModel(model_name)
                    response = model_g.generate_content(prompt)
                    if response and hasattr(response, "text"):
                        return response.text
                    break
                except Exception as e:
                    msg = str(e).lower()
                    if "429" in msg or "rate limit" in msg:
                        if attempt < max_retries:
                            sleep_time = 2 ** attempt
                            st.warning(f"Rate limit hit for {model_name}. Retrying in {sleep_time}s...")
                            import time
                            time.sleep(sleep_time)
                            continue
                    # if model not found or unauthorized, move to next model key
                    if "not found" in msg or "unknown model" in msg or "unauthorized" in msg or "invalid" in msg:
                        break
                    # for other errors, log and try again if possible
                    if attempt < max_retries:
                        continue
                    break
    return "AI analysis unavailable due to API limits or configuration. Please verify your key and model."
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
if "forgot_pw" not in st.session_state:
    st.session_state.forgot_pw=False

# ---------------- AUTH ---------------- #
if not st.session_state.user:
    st.markdown("<h1 class='glow-text'>RESUME AI MATCH</h1>", unsafe_allow_html=True)

    tab1,tab2=st.tabs(["Login","Signup"])

    with tab1:
        login_email=st.text_input("Email", key="login_email")
        login_password=st.text_input("Password", type="password", key="login_password")

        if st.button("Login", key="login_button"):
            c.execute("SELECT * FROM users WHERE email=? AND password=?", (login_email, hash_data(login_password)))
            if c.fetchone():
                st.session_state.user=login_email
                st.session_state.forgot_pw=False
                st.rerun()
            else:
                st.error("Invalid email/password")

        if st.button("Forgot Password?", key="forgot_password_button"):
            st.session_state.forgot_pw=True

        if st.session_state.forgot_pw:
            fp_email=st.text_input("Enter your email to reset password", key="fp_email")
            if st.button("Send reset OTP", key="send_reset_otp"):
                c.execute("SELECT * FROM users WHERE email=?", (fp_email,))
                if c.fetchone():
                    otp, hashed = send_otp(fp_email)
                    st.session_state.password_reset_otp = hashed
                    st.success(f"OTP sent: {otp}")
                else:
                    st.error("Email not registered")

            reset_otp=st.text_input("Reset OTP", key="reset_otp")
            new_password=st.text_input("New password", type="password", key="reset_new_password")
            if st.button("Reset Password", key="reset_password"):
                if st.session_state.get("password_reset_otp") and hash_data(reset_otp) == st.session_state.get("password_reset_otp"):
                    c.execute("UPDATE users SET password=? WHERE email=?", (hash_data(new_password), fp_email))
                    conn.commit()
                    st.success("Password reset successful. Please login.")
                    st.session_state.forgot_pw=False
                else:
                    st.error("Invalid OTP")

    with tab2:
        su_username=st.text_input("Choose username", key="su_username")
        su_email=st.text_input("Email", key="su_email")
        su_password=st.text_input("Password", type="password", key="su_password")

        if st.button("Send OTP", key="send_signup_otp"):
            if su_email:
                otp, hashed = send_otp(su_email)
                st.session_state.signup_otp_secret = hashed
                st.success(f"OTP sent: {otp}")
            else:
                st.error("Please enter an email first")

        entered_otp = st.text_input("OTP", key="signup_otp_input")
        if st.button("Register", key="signup_register"):
            if not su_username:
                st.error("Please create a username")
            elif not su_email:
                st.error("Please enter email")
            elif not su_password:
                st.error("Please enter password")
            elif not entered_otp:
                st.error("Please enter OTP")
            elif hash_data(entered_otp) != st.session_state.get("signup_otp_secret"):
                st.error("Wrong OTP")
            else:
                c.execute("SELECT * FROM users WHERE username=? OR email=?", (su_username, su_email))
                if c.fetchone():
                    st.error("Username or email already exists")
                else:
                    c.execute("INSERT INTO users VALUES (?,?,?)", (su_username, su_email, hash_data(su_password)))
                    conn.commit()
                    st.success("Registered successfully! Please login.")

# ---------------- MAIN ---------------- #
else:
    st.sidebar.markdown("### Navigation")
    if "menu" not in st.session_state:
        st.session_state.menu="Home"

    if st.sidebar.button("Home"):
        st.session_state.menu="Home"
    if st.sidebar.button("Analytics"):
        st.session_state.menu="Analytics"
    if st.sidebar.button("Salary"):
        st.session_state.menu="Salary"
    if st.sidebar.button("AI"):
        st.session_state.menu="AI"
    if st.sidebar.button("About"):
        st.session_state.menu="About"
    if st.sidebar.button("Logout"):
        st.session_state.user=None
        st.session_state.menu="Home"
        st.experimental_rerun()

    menu = st.session_state.menu

    if menu=="Home":
        st.markdown("<h1 class='glow-text'>UPLOAD RESUME</h1>", unsafe_allow_html=True)

        file=st.file_uploader("Upload PDF",type="pdf")

        if file:
            text=""
            with pdfplumber.open(file) as pdf:
                for p in pdf.pages:
                    text+=p.extract_text() or ""

            resume_text = text.lower()
            st.session_state.resume=resume_text

            # ---------- MATCHING FIX ---------- #
            res_emb=model.encode([resume_text])
            cos=cosine_similarity(res_emb,jd_emb)[0]

            results=[]
            for i,job in enumerate(jobs):
                skills=job["skills"]

                matched=[s for s in skills if word_in_text(s, text)]
                missing=[s for s in skills if not word_in_text(s, text)]

                skill_score=len(matched)/len(skills)

                if len(matched) == 0:
                    final_score = 0.0
                else:
                    # make sure cosine is in [0,1] and apply safe scaling
                    cos_score = max(0, min(1, cos[i]))
                    final_score = (0.7 * cos_score + 0.3 * skill_score) * 100

                results.append((job, final_score, matched, missing))

            results=sorted(results,key=lambda x:x[1],reverse=True)

            if all(score==0 for _,score,_,_ in results):
                st.warning("No skills matched. Try a resume with explicit job/technology keywords (python, aws, ml, sql, etc.).")

            for job,score,matched,missing in results:
                with st.expander(f"{job['role']} - {score:.2f}% match", expanded=False):
                    st.markdown(f"""
                    <div class="info-card">
                    <h3>{job['role']}</h3>
                    <h2>{score:.2f}%</h2>
                    <p><b>Matched:</b>{" ".join([f"<span class='badge match'>{m}</span>" for m in matched]) or 'None'}</p>
                    <p><b>Missing:</b>{" ".join([f"<span class='badge miss'>{m}</span>" for m in missing]) or 'None'}</p>
                    </div>
                    """, unsafe_allow_html=True)

            # ---------- AI FIX ---------- #
            report_text = "AI analysis unavailable."

            if GEMINI_API_KEY_PRIMARY or GEMINI_API_KEY_FALLBACK:
                try:
                    prompt = f"""
Detect resume language and respond in SAME language.

Give:
1. Detailed Analysis
2. Future Improvements

Resume:
{text[:1500]}
"""
                    report_text = genai_generate_with_fallback(prompt)
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

        coding=sum(word_in_text(x, text) for x in ["python","java","c++"])
        ml=sum(word_in_text(x, text) for x in ["ml","ai","deep"])
        db=sum(word_in_text(x, text) for x in ["sql","mongodb"])

        if not text:
            st.warning("Upload your resume first to see analytics.")
        else:
            labels=["Coding","ML/AI","Database"]
            values=[coding, ml, db]
            if sum(values)==0:
                st.warning("No skill keywords found in resume. Please upload a resume with technical skills.")
                values=[1,1,1]
                textinfo='label+value'
            else:
                textinfo='label+percent'

            fig = go.Figure(go.Pie(
                labels=labels,
                values=values,
                hole=0.25,
                textinfo=textinfo,
                pull=[0.1,0.05,0.05],
                marker=dict(colors=['#636EFA','#EF553B','#00CC96'], line=dict(color='#111', width=2))
            ))
            fig.update_layout(
                title="3D-style Pie Chart Analytics",
                paper_bgcolor="#111",
                plot_bgcolor="#111",
                font=dict(color="white"),
                legend=dict(itemsizing='constant', font=dict(color='white')),
                margin=dict(t=50,b=20,l=20,r=20)
            )
            st.plotly_chart(fig, use_container_width=True)

    elif menu=="Salary":
        st.markdown("## Salary Prediction (INR)")
        exp = st.slider("Years of experience", 0, 25, 1)
        role = st.selectbox("Target role", ["Software Engineer", "Data Scientist", "Frontend Dev", "ML Engineer", "Cloud Engineer"])

        # Experience-based salary bands
        role_base = {
            "Software Engineer": 450000,
            "Data Scientist": 550000,
            "Frontend Dev": 420000,
            "ML Engineer": 600000,
            "Cloud Engineer": 500000
        }
        base = role_base.get(role, 450000)
        predicted = int(base * (1 + exp * 0.09))

        st.metric("Predicted CTC (INR)", f"₹{predicted:,}")

        explanation = (
            f"For a {role} with {exp} year{'s' if exp!=1 else ''} of experience, "
            f"the expected salary range in India is around ₹{predicted:,}. "
            "This estimate accounts for fresher-to-mid/senior increments and market trends. "
        )

        if exp < 2:
            explanation += "Entry-level roles typically start with training, coding and teamwork responsibilities."
        elif exp < 5:
            explanation += "At this stage, candidates often contribute independently and may lead small modules."
        elif exp < 10:
            explanation += "With strong experience, expectations include architecture, mentoring, and product impact."
        else:
            explanation += "Senior professionals often own strategy, technical leadership and cross-team execution."

        st.info(explanation)

    elif menu=="AI":
        q=st.text_input("Ask anything")
        if q:
            if not GEMINI_API_KEY_PRIMARY and not GEMINI_API_KEY_FALLBACK:
                st.warning("Gemini API key not configured. AI is unavailable.")
            else:
                try:
                    res_text = genai_generate_with_fallback(q)
                    st.write(res_text)
                except Exception as e:
                    st.error(f"AI error: {e}")

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