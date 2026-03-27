import re
import os
import sqlite3
import hashlib
import random
import pdfplumber
import streamlit as st
import plotly.graph_objects as go
import google.generativeai as genai
import spacy
from sentence_transformers import SentenceTransformer, util
from langdetect import detect

# ---------------- CONFIG & MODELS ---------------- #
st.set_page_config(page_title="Resume AI Pro", layout="wide")

GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY_PRIMARY", "")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

@st.cache_resource
def load_models():
    try:
        nlp = spacy.load("en_core_web_sm")
    except:
        os.system("python -m spacy download en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")
    transformer = SentenceTransformer('all-MiniLM-L6-v2')
    return nlp, transformer

nlp, transformer_model = load_models()

# ---------------- CSS STYLE ---------------- #
st.markdown("""
<style>
    .stApp { background:#0a0a0a; color:#f0f0f0; }
    .stButton>button { width: 100%; border-radius: 8px; margin-bottom: 5px; background-color: #1e1e1e; color: white; border: 1px solid #333; }
    .stButton>button:hover { border-color: #6366f1; color: #6366f1; }
    .job-card {
        background: #161616; padding: 20px; border-radius: 12px;
        border: 1px solid #333; margin-bottom: 15px;
    }
    .glow-card {
        background: #111; padding: 20px; border-radius: 15px;
        border: 1px solid #6366f1; text-align: center;
        box-shadow: 0 0 15px rgba(99, 102, 241, 0.3);
        height: 280px; transition: 0.4s;
    }
    .skill-tag {
        display: inline-block; padding: 3px 10px; border-radius: 15px;
        margin: 3px; font-size: 0.7rem;
    }
    .matched { background: #064e3b; color: #34d399; border: 1px solid #059669; }
    .missing { background: #450a0a; color: #f87171; border: 1px solid #991b1b; }
    .glow-text { text-align:center; color:white; text-shadow:0 0 10px #6366f1; }
    [data-testid="stSidebarNav"] {display: none;}
</style>
""", unsafe_allow_html=True)

# ---------------- DATABASE ---------------- #
DB_FILE = "users_v6.db"
def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("CREATE TABLE IF NOT EXISTS users (username TEXT PRIMARY KEY, email TEXT, password TEXT)")
    conn.commit()
    conn.close()
init_db()

def hash_data(data): return hashlib.sha256(data.encode()).hexdigest()

def ai_query(prompt):
    if not GEMINI_API_KEY: return "Error: No API Key"
    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(prompt)
        return response.text
    except Exception as e: return f"AI Error: {str(e)}"

# ---------------- SESSION STATE ---------------- #
for key in ["user", "resume_text", "dynamic_roles", "hashed_otp", "resume_lang"]:
    if key not in st.session_state: st.session_state[key] = None
if "page" not in st.session_state: st.session_state.page = "Home"

# ---------------- AUTH ---------------- #
if st.session_state.user is None:
    st.markdown("<h1 class='glow-text'>RESUME-JD MATCHER</h1>", unsafe_allow_html=True)
    auth_tab = st.tabs(["Login", "Sign Up", "Forgot Password"])
    
    with auth_tab[0]:
        l_user = st.text_input("Username")
        l_pass = st.text_input("Password", type="password")
        if st.button("Access Dashboard"):
            conn = sqlite3.connect(DB_FILE); cur = conn.cursor()
            cur.execute("SELECT username FROM users WHERE username=? AND password=?", (l_user, hash_data(l_pass)))
            if cur.fetchone():
                st.session_state.user = l_user
                st.rerun()
            else: st.error("Invalid credentials.")
            conn.close()

    with auth_tab[1]:
        s_email = st.text_input("Email")
        s_user = st.text_input("New Username")
        s_pass = st.text_input("New Password", type="password")
        if st.button("Get Registration OTP"):
            otp = str(random.randint(100000, 999999))
            st.session_state.hashed_otp = hash_data(otp)
            st.info(f"OTP: {otp}")
        v_otp = st.text_input("Enter OTP")
        if st.button("Register"):
            if st.session_state.hashed_otp == hash_data(v_otp):
                conn = sqlite3.connect(DB_FILE); cur = conn.cursor()
                try:
                    cur.execute("INSERT INTO users VALUES (?,?,?)", (s_user, s_email, hash_data(s_pass)))
                    conn.commit(); st.success("Account created!")
                except: st.error("User exists.")
                conn.close()
            else: st.error("Wrong OTP")

    with auth_tab[2]:
        f_user = st.text_input("Username for Reset")
        if st.button("Get Reset OTP"):
            st.session_state.hashed_otp = hash_data("123456")
            st.info("OTP: 123456")
        f_otp = st.text_input("OTP Code")
        new_p = st.text_input("New Pass", type="password")
        if st.button("Reset"):
            if st.session_state.hashed_otp == hash_data(f_otp):
                conn = sqlite3.connect(DB_FILE); cur = conn.cursor()
                cur.execute("UPDATE users SET password=? WHERE username=?", (hash_data(new_p), f_user))
                conn.commit(); conn.close(); st.success("Updated!")

# ---------------- MAIN APP ---------------- #
else:
    with st.sidebar:
        st.markdown(f"### 👤 {st.session_state.user}")
        if st.button("🏠 Home"): st.session_state.page = "Home"
        if st.button("📊 Analytics"): st.session_state.page = "Analytics"
        if st.button("💰 Salary Prediction"): st.session_state.page = "Salary"
        if st.button("🤖 AI Assistant"): st.session_state.page = "Assistant"
        if st.button("ℹ️ About"): st.session_state.page = "About"
        st.divider()
        if st.button("🚪 Logout"):
            st.session_state.user = None
            st.rerun()

    if st.session_state.page == "Home":
        st.subheader("Top 3 Role Matching & Improvement Suggestions")
        file = st.file_uploader("Upload Resume (PDF)", type="pdf")
        if file:
            with pdfplumber.open(file) as pdf:
                text = " ".join([p.extract_text() or "" for p in pdf.pages])
            st.session_state.resume_text = text
            st.session_state.resume_lang = detect(text)
            
            # Identify Top 3 Roles
            if not st.session_state.dynamic_roles:
                with st.spinner("Analyzing Career Path..."):
                    roles_prompt = f"Identify exactly 3 career paths for this resume text: {text[:1000]}. Provide ONLY the names of the roles, comma separated. Answer in English for processing."
                    roles_raw = ai_query(roles_prompt)
                    st.session_state.dynamic_roles = [r.strip() for r in roles_raw.split(",")][:3]
            
            res_emb = transformer_model.encode(text, convert_to_tensor=True)
            
            # Display Match Cards
            for role in st.session_state.dynamic_roles:
                jd_skills = ai_query(f"List 6 essential technical keywords for the role of {role}. Comma separated.")
                skills_list = [s.strip().lower() for s in jd_skills.split(",")]
                
                matched = [s for s in skills_list if s in text.lower()]
                missing = [s for s in skills_list if s not in text.lower()]
                
                # Logic: No match = 0%
                if not matched: score = 0.0
                else:
                    jd_emb = transformer_model.encode(" ".join(skills_list), convert_to_tensor=True)
                    score = float(util.cos_sim(res_emb, jd_emb)) * 100

                st.markdown(f"""
                <div class="job-card">
                    <h4 style="color:#6366f1;">{role}</h4>
                    <h2 style="margin: 0;">{score:.1f}% Match</h2>
                    <p style="margin-top:10px;"><b>Skills Detected:</b> {" ".join([f'<span class="skill-tag matched">{s}</span>' for s in matched]) if matched else 'None'}</p>
                    <p><b>Skills Missing:</b> {" ".join([f'<span class="skill-tag missing">{s}</span>' for s in missing]) if missing else 'None'}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Suggestions in user's language
            with st.expander("🛠️ Personalized Resume Improvements"):
                with st.spinner("Generating suggestions..."):
                    lang_name = "Hindi" if st.session_state.resume_lang == 'hi' else "the detected language of the resume"
                    improve_prompt = f"Based on this resume: {text[:1000]}, provide 3 specific points to improve it for ATS systems. Do NOT ask to translate it. Write the response ONLY in {lang_name}."
                    st.write(ai_query(improve_prompt))
            
            st.download_button("📥 Download Analysis", data=text, file_name="analysis.txt")

    elif st.session_state.page == "Analytics":
        st.subheader("Career Focus Distribution")
        if st.session_state.resume_text:
            st.write("This chart represents the primary areas of expertise found in your resume.")
            tech = ["Programming", "Cloud", "Data Analysis", "Management", "Communication", "Design"]
            counts = [st.session_state.resume_text.lower().count(t.lower()) + 1 for t in tech]
            
            fig = go.Figure(go.Pie(labels=tech, values=counts, hole=.5, marker_colors=['#6366f1','#818cf8','#a5b4fc','#c7d2fe','#e0e7ff']))
            fig.update_layout(template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)
        else: st.warning("Upload a resume first.")

    elif st.session_state.page == "Salary":
        st.subheader("Market Rate Estimation")
        if st.session_state.resume_text:
            role = st.selectbox("Select Target Role", st.session_state.dynamic_roles if st.session_state.dynamic_roles else ["Software Engineer"])
            exp = st.slider("Experience (Years)", 0, 20, 2)
            if st.button("Estimate Salary"):
                res = ai_query(f"What is the average USD salary for a {role} with {exp} years experience? Provide ONLY a single numeric value.")
                st.metric("Estimated Market Value", f"${res.strip()}")
        else: st.warning("Upload a resume first.")

    elif st.session_state.page == "Assistant":
        st.subheader("AI Career Chat")
        q = st.text_input("Ask about career paths, interview prep, or your resume:")
        if q:
            ans = ai_query(f"Context: {st.session_state.resume_text[:1000]}. Question: {q}")
            st.info(ans)

    elif st.session_state.page == "About":
        st.markdown("<h2 style='text-align:center;'>System Intelligence</h2>", unsafe_allow_html=True)
        cols = st.columns(5)
        cards = [
            ("Linguistic AI", "Detects resume language and provides localized career advice."),
            ("SBERT Embeddings", "Calculates semantic similarity between resume and job roles."),
            ("Gemini 2.5", "Generative intelligence for dynamic role inference and improvement tips."),
            ("SHA-256", "Secure cryptographic hashing for account and OTP authentication."),
            ("Visual Analytics", "Plotly-driven distribution charts for layman-friendly insights.")
        ]
        for i, (t, d) in enumerate(cards):
            with cols[i]:
                st.markdown(f"<div class='glow-card'><h4 style='color:#818cf8;'>{t}</h4><p>{d}</p></div>", unsafe_allow_html=True)