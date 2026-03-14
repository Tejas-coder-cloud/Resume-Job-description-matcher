# ============================================================
# Resume–JD Matcher | FINAL COMPLETE VERSION
# ============================================================

import streamlit as st
import pandas as pd
import sqlite3
import hashlib
import random
import smtplib
import re
import spacy
import plotly.graph_objects as go

from spacy.matcher import PhraseMatcher
from email.message import EmailMessage
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from openai import OpenAI
except:
    OpenAI=None


# ------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------

st.set_page_config("Resume-JD Matcher",layout="wide")


# ------------------------------------------------------------
# GLOBAL CSS
# ------------------------------------------------------------

st.markdown("""
<style>

.glow-title{
text-align:center;
font-size:44px;
font-weight:bold;
color:white;
text-shadow:0 0 10px #60a5fa,0 0 20px #3b82f6,0 0 40px #2563eb;
margin-bottom:20px;
}

.card{
backdrop-filter:blur(10px);
background:rgba(255,255,255,0.06);
padding:20px;
border-radius:14px;
margin-bottom:20px;
transition:0.35s;
}

.card:hover{
transform:translateY(-6px) scale(1.02);
box-shadow:0 0 18px rgba(59,130,246,0.7);
}

.about-red{background:rgba(239,68,68,0.35)}
.about-blue{background:rgba(59,130,246,0.35)}
.about-green{background:rgba(34,197,94,0.35)}
.about-purple{background:rgba(168,85,247,0.35)}
.about-orange{background:rgba(249,115,22,0.35)}

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
""",unsafe_allow_html=True)


# ------------------------------------------------------------
# BACKGROUND FUNCTION
# ------------------------------------------------------------

def set_background(c1,c2):

    st.markdown(f"""
    <style>
    .stApp {{
    background: linear-gradient(-45deg,{c1},{c2},{c1});
    background-size:400% 400%;
    animation:gradientBG 12s ease infinite;
    color:white;
    }}

    @keyframes gradientBG {{
    0%{{background-position:0% 50%}}
    50%{{background-position:100% 50%}}
    100%{{background-position:0% 50%}}
    }}
    </style>
    """,unsafe_allow_html=True)


# ------------------------------------------------------------
# OPENAI HELPER
# ------------------------------------------------------------

def get_ai_response(prompt):

    try:
        api_key=st.secrets.get("OPENAI_API_KEY")
    except:
        api_key=None

    if not api_key or OpenAI is None:
        return None

    try:
        client=OpenAI(api_key=api_key)

        response=client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role":"system","content":"You are a helpful career advisor."},
                {"role":"user","content":prompt}
            ],
            max_tokens=200
        )

        return response.choices[0].message.content

    except:
        return None


# ------------------------------------------------------------
# EXPLANATION GENERATOR
# ------------------------------------------------------------

def generate_explanation(role,matched,missing):

    text=[]

    if matched:
        text.append(
        f"The candidate demonstrates relevant exposure for the {role} role through skills such as {', '.join(matched)}.")

    if missing:
        text.append(
        f"Important technologies like {', '.join(missing)} are recommended to improve the matching score.")

    text.append(
    "Building practical projects and highlighting them clearly in the resume can increase the match percentage.")

    return "<br>• " + "<br>• ".join(text)


# ------------------------------------------------------------
# MODEL
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
# NLP SKILL EXTRACTION
# ------------------------------------------------------------

nlp=spacy.load("en_core_web_sm")

skills_db=[
"python","java","c++","aws","docker","kubernetes",
"machine learning","deep learning","nlp",
"pandas","numpy","sql","terraform","linux",
"react","node","mongodb","html","css","javascript","express"
]

matcher=PhraseMatcher(nlp.vocab)
patterns=[nlp(skill) for skill in skills_db]
matcher.add("SKILLS",patterns)

def extract_skills(text):

    doc=nlp(text.lower())
    matches=matcher(doc)

    skills=set()

    for match_id,start,end in matches:
        skills.add(doc[start:end].text)

    return list(skills)


# ------------------------------------------------------------
# SESSION STATE
# ------------------------------------------------------------

if "page" not in st.session_state:
    st.session_state.page="Home"

if "chat_history" not in st.session_state:
    st.session_state.chat_history=[]


# ------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------

st.sidebar.title("Menu")

if st.sidebar.button("Home"):
    st.session_state.page="Home"

if st.sidebar.button("About"):
    st.session_state.page="About"

if st.sidebar.button("AI Assistant"):
    st.session_state.page="AI"


# ============================================================
# HOME PAGE
# ============================================================

if st.session_state.page=="Home":

    set_background("#0f172a","#1e293b")

    st.markdown("<div class='glow-title'>Resume-JD Matcher</div>",unsafe_allow_html=True)

    exp=st.slider("Experience",0,20,0)

    resume=st.text_area("Paste Resume / Skills")

    if st.button("Analyze"):

        skills=extract_skills(resume)

        resume_emb=model.encode([resume])

        sims=cosine_similarity(resume_emb,job_embeddings)[0]

        for i in sims.argsort()[::-1][:3]:

            role=jobs_df.iloc[i]["Role"]

            desc=jobs_df.iloc[i]["description"]

            role_tokens=desc.split()

            matched=[s for s in role_tokens if s in skills]

            missing=[s for s in role_tokens if s not in skills]

            similarity_score=sims[i]

            skill_overlap=len(matched)/len(role_tokens)

            score=0.7*similarity_score+0.3*skill_overlap

            base_salary=3+score*3
            salary=base_salary+exp*0.7

            explanation=generate_explanation(role,matched,missing)

            st.markdown(f"""
            <div class='card'>

            <h3>{role}</h3>

            Match Score: {score*100:.1f}%<br>
            Predicted Salary: ₹{salary:.1f} LPA

            <hr>

            <b>Explanation</b><br>
            {explanation}

            <br>

            <b>Matched Skills:</b> {', '.join(matched) if matched else "None"}<br>
            <b>Recommended Skills:</b> {', '.join(missing) if missing else "None"}

            </div>
            """,unsafe_allow_html=True)

            fig=go.Figure()

            fig.add_bar(x=["Matched Skills"],y=[len(matched)],marker_color="green")

            fig.add_bar(x=["Missing Skills"],y=[len(missing)],marker_color="red")

            fig.update_layout(title="Skill Gap Analysis",height=350)

            st.plotly_chart(fig,use_container_width=True)


# ============================================================
# ABOUT PAGE
# ============================================================

elif st.session_state.page=="About":

    set_background("#111827","#111827")

    st.markdown("<div class='glow-title'>System Architecture</div>",unsafe_allow_html=True)

    sections=[

        ("Transformer Architecture",
         "SentenceTransformer converts resumes and job descriptions into embeddings.",
         "about-purple"),

        ("Skill Gap Analysis",
         "Compares resume skills with job requirements to identify missing technologies.",
         "about-red"),

        ("Cosine Similarity",
         "Measures similarity between embedding vectors to find the best job match.",
         "about-blue"),

        ("Database",
         "SQLite stores user credentials securely using SHA256 hashing.",
         "about-green"),

        ("Explainability",
         "The system generates recruiter style reasoning for job matches.",
         "about-orange")
    ]

    for title,desc,color in sections:

        st.markdown(f"""
        <div class='card {color}'>
        <b>{title}</b><br><br>{desc}
        </div>
        """,unsafe_allow_html=True)


# ============================================================
# AI ASSISTANT
# ============================================================

elif st.session_state.page=="AI":

    set_background("#6d28d9","#9333ea")

    st.markdown("<div class='glow-title'>Career AI Assistant</div>",unsafe_allow_html=True)

    for m in st.session_state.chat_history:

        st.markdown(f"<div class='chat-user'><b>You:</b> {m['u']}</div>",unsafe_allow_html=True)

        st.markdown(f"<div class='chat-ai'><b>AI:</b> {m['a']}</div>",unsafe_allow_html=True)

    q=st.text_input("Ask a question")

    if st.button("Send") and q:

        ans=get_ai_response(q)

        if ans is None:
            ans="Sorry I am unable to answer that right now."

        st.session_state.chat_history.append({"u":q,"a":ans})

        st.rerun()

    if st.button("Clear Chat History"):
        st.session_state.chat_history.clear()
        st.rerun()