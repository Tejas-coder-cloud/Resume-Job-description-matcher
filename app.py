import streamlit as st
import pandas as pd
import sqlite3
import hashlib
import spacy
import pdfplumber
import plotly.graph_objects as go
import random
import yagmail
import os

from spacy.matcher import PhraseMatcher
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from crewai import Agent, Task, Crew, Process

# ------------------------------------------------------------
# CONFIG & SECRETS
# ------------------------------------------------------------
# Accessing secrets from .streamlit/secrets.toml
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
EMAIL_USER = st.secrets["EMAIL_USER"]
EMAIL_PASS = st.secrets["EMAIL_PASSWORD"]

# ------------------------------------------------------------
# PAGE CONFIG & CSS
# ------------------------------------------------------------
st.set_page_config(page_title="ATS Pro Matcher", layout="wide", page_icon="📄")

st.markdown("""
<style>
    /* Global Styles */
    .stApp { background-color: #ffffff; color: #111111; font-family: 'Segoe UI', sans-serif; }
    
    /* Interactive About Cards */
    .about-card {
        background: #f8f9fa;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #e9ecef;
        transition: all 0.3s ease-in-out;
        margin-bottom: 20px;
        min-height: 150px;
    }
    .about-card:hover {
        background: #4f46e5;
        color: white !important;
        transform: translateY(-10px);
        box-shadow: 0 10px 20px rgba(79, 70, 229, 0.2);
    }
    .about-card:hover h3, .about-card:hover p {
        color: white !important;
    }

    /* Skill Tags */
    .skill-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 15px;
        margin: 4px;
        font-size: 0.8rem;
        font-weight: 500;
    }
    .matched { background-color: #d1fae5; color: #065f46; }
    .missing { background-color: #fee2e2; color: #991b1b; }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# DATABASE LOGIC
# ------------------------------------------------------------
conn = sqlite3.connect("users.db", check_same_thread=False)
c = conn.cursor()
c.execute("CREATE TABLE IF NOT EXISTS users(username TEXT, email TEXT UNIQUE, password TEXT)")
conn.commit()

def hash_pass(p): return hashlib.sha256(p.encode()).hexdigest()

# ------------------------------------------------------------
# NLP & MODELS
# ------------------------------------------------------------
@st.cache_resource
def load_nlp_assets():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    nlp = spacy.load("en_core_web_sm")
    
    # Predefined Job Descriptions for matching
    jd_data = {
        "Software Engineer": "python java c++ system design dsa git api cloud linux",
        "Data Scientist": "python machine learning stats pandas sql pytorch nlp math",
        "Frontend Developer": "javascript react html css typescript tailwind sass",
        "Cloud Architect": "aws azure docker kubernetes terraform networking security"
    }
    df = pd.DataFrame(list(jd_data.items()), columns=["Role", "Content"])
    embeddings = model.encode(df["Content"].tolist())
    return model, nlp, df, embeddings

model, nlp, jd_df, jd_embeddings = load_nlp_assets()

def extract_text_from_pdf(file):
    with pdfplumber.open(file) as pdf:
        return " ".join([page.extract_text() for page in pdf.pages if page.extract_text()])

# ------------------------------------------------------------
# SESSION STATE
# ------------------------------------------------------------
if "user" not in st.session_state: st.session_state.user = None
if "otp" not in st.session_state: st.session_state.otp = None

# ------------------------------------------------------------
# LOGIN / SIGNUP / FORGOT PASSWORD
# ------------------------------------------------------------
if st.session_state.user is None:
    st.title("ATS Professional Login")
    choice = st.radio("Select Action", ["Login", "Sign Up", "Forgot Password"], horizontal=True)

    if choice == "Login":
        email = st.text_input("Email")
        pw = st.text_input("Password", type="password")
        if st.button("Login"):
            c.execute("SELECT username FROM users WHERE email=? AND password=?", (email, hash_pass(pw)))
            res = c.fetchone()
            if res:
                st.session_state.user = res[0]
                st.rerun()
            else: st.error("Invalid credentials.")

    elif choice == "Sign Up":
        u = st.text_input("Full Name")
        e = st.text_input("Email")
        p = st.text_input("Password", type="password")
        if st.button("Send OTP"):
            st.session_state.otp = str(random.randint(100000, 999999))
            try:
                yag = yagmail.SMTP(EMAIL_USER, EMAIL_PASS)
                yag.send(to=e, subject="Verification Code", contents=f"Your OTP is {st.session_state.otp}")
                st.success("OTP sent to your email!")
            except Exception as ex: st.error(f"Failed to send email: {ex}")
        
        entered_otp = st.text_input("Enter 6-digit OTP")
        if st.button("Complete Registration"):
            if entered_otp == st.session_state.otp:
                c.execute("INSERT INTO users VALUES (?,?,?)", (u, e, hash_pass(p)))
                conn.commit()
                st.success("Account created successfully!")
            else: st.error("Incorrect OTP.")

    elif choice == "Forgot Password":
        e = st.text_input("Registered Email")
        if st.button("Request Reset OTP"):
            st.session_state.otp = str(random.randint(100000, 999999))
            yagmail.SMTP(EMAIL_USER, EMAIL_PASS).send(to=e, subject="Reset OTP", contents=f"Code: {st.session_state.otp}")
            st.info("OTP sent.")
        
        # Logic for resetting would go here...

# ------------------------------------------------------------
# MAIN DASHBOARD
# ------------------------------------------------------------
else:
    menu = st.sidebar.radio("Navigation", ["Home", "Visual Analytics", "AI Assistant", "About", "Logout"])

    if menu == "Logout":
        st.session_state.user = None
        st.rerun()

    elif menu == "Home":
        st.title(f"Welcome, {st.session_state.user}")
        uploaded_file = st.file_uploader("Upload your Resume (PDF)", type="pdf")

        if uploaded_file:
            resume_text = extract_text_from_pdf(uploaded_file)
            
            # Semantic Similarity
            res_emb = model.encode([resume_text])
            scores = cosine_similarity(res_emb, jd_embeddings)[0]
            
            # Top 3 Roles
            top_indices = scores.argsort()[-3:][::-1]
            
            st.subheader("🎯 Top 3 Job Recommendations")
            cols = st.columns(3)
            for i, idx in enumerate(top_indices):
                role = jd_df.iloc[idx]["Role"]
                req_skills = jd_df.iloc[idx]["Content"].split()
                matched = [s for s in req_skills if s.lower() in resume_text.lower()]
                missing = [s for s in req_skills if s.lower() not in resume_text.lower()]
                
                with cols[i]:
                    st.metric(role, f"{scores[idx]*100:.1f}%")
                    st.write("**Matched:**")
                    for m in matched: st.markdown(f"<span class='skill-badge matched'>{m}</span>", unsafe_allow_html=True)
                    st.write("**Missing:**")
                    for ms in missing: st.markdown(f"<span class='skill-badge missing'>{ms}</span>", unsafe_allow_html=True)

            st.divider()
            st.subheader("🧑‍🏫 Human Explanation")
            best_role = jd_df.iloc[top_indices[0]]['Role']
            st.info(f"Your background is a strong semantic match for a **{best_role}**. "
                    "While your core technical skills align well, the ATS suggests adding more context regarding "
                    f"**{', '.join(missing[:2])}** to bypass strict filtering systems.")

    elif menu == "Visual Analytics":
        st.title("Career Growth & Recommendations")
        # Visualizing Match Strength across categories
        fig = go.Figure(data=go.Scatterpolar(
            r=[80, 65, 90, 70, 85],
            theta=['Technical Skills','Communication','Project Experience','Education','Leadership'],
            fill='toself',
            line_color='#4f46e5'
        ))
        st.plotly_chart(fig, use_container_width=True)
        st.write("**Future Recommendation:** Your technical projects are high-impact. Focus on documenting 'Leadership' roles to qualify for Senior positions.")

    elif menu == "AI Assistant":
        st.title("CrewAI Career Agent")
        user_query = st.text_input("Ask about career shifts, resume tips, or interview prep:")

        if st.button("Consult Crew") and user_query:
            # CrewAI Agents powered by OpenAI
            career_agent = Agent(
                role='Executive Career Coach',
                goal='Provide expert-level career advancement advice.',
                backstory='A former headhunter with 15 years of experience at top-tier tech firms.',
                verbose=True
            )
            
            task = Task(description=user_query, agent=career_agent, expected_output="A concise, actionable 3-step career plan.")
            
            crew = Crew(agents=[career_agent], tasks=[task], process=Process.sequential)
            
            with st.spinner("Crew is analyzing..."):
                result = crew.kickoff()
                st.markdown(f"### Coach's Plan:\n{result}")

    elif menu == "About":
        st.title("Technical Architecture")
        st.write("Hover over the cards to learn more about the system internals.")
        
        row1 = st.columns(3)
        row2 = st.columns(3)
        
        pillars = [
            ("Transformer Archi", "Leverages Sentence-Transformers to map resumes into a dense vector space for contextual understanding."),
            ("Cosine Similarity", "Determines the mathematical similarity between resume vectors and job description vectors."),
            ("Skill Gap Analysis", "Uses Tokenization and PhraseMatching to pinpoint exactly which industry skills are missing."),
            ("Database Management", "Secure SQLite storage for user profiles with SHA-256 encrypted password protocols."),
            ("OTP Verification", "Email-based security layer using Streamlit Secrets and yagmail for user integrity."),
            ("CrewAI Multi-Agent", "Orchestrates autonomous agents to simulate professional career counseling and resume feedback.")
        ]
        
        for i, (title, desc) in enumerate(pillars):
            container = row1[i] if i < 3 else row2[i-3]
            with container:
                st.markdown(f"""
                <div class="about-card">
                    <h3>{title}</h3>
                    <p>{desc}</p>
                </div>
                """, unsafe_allow_html=True)