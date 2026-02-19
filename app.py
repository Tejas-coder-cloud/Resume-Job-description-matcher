import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import os

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Resume–JD Matching System",
    layout="wide"
)

# --------------------------------------------------
# Optimized Resource Loading (Reduces Latency)
# --------------------------------------------------
@st.cache_resource
def load_all_resources():
    # Load Model
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Load Data
    df = pd.read_csv("jobs_processed.csv")
    job_embs = np.load("job_embeddings.npy")
    
    # Load Skills
    with open("skills.txt", "r", encoding="utf-8") as f:
        skills = [s.strip().lower() for s in f if s.strip()]
        
    # Load KB and pre-calculate embeddings to stop lag in AI section
    with open("knowledge_base.txt", "r", encoding="utf-8") as f:
        paragraphs = [p.strip() for p in f.read().split("\n\n") if p.strip()]
    kb_embs = model.encode(paragraphs)
    
    return model, df, job_embs, skills, paragraphs, kb_embs

# Global Access to pre-loaded data
model, df, job_embeddings, skills_list, kb_paragraphs, kb_embeddings = load_all_resources()

# --------------------------------------------------
# Sidebar Navigation
# --------------------------------------------------
if "section" not in st.session_state:
    st.session_state.section = "Home"

def set_section(name):
    st.session_state.section = name

bg_configs = {
    "Home": {"gradient": "linear-gradient(-45deg, #020617, #0f172a, #1e293b, #020617)", "accent": "#00f5ff"},
    "About": {"gradient": "linear-gradient(-45deg, #020617, #1e1b4b, #312e81, #020617)", "accent": "#c084fc"},
    "AI": {"gradient": "linear-gradient(-45deg, #020617, #064e3b, #022c22, #020617)", "accent": "#4ade80"}
}
conf = bg_configs.get(st.session_state.section, bg_configs["Home"])

with st.sidebar:
    st.markdown("## ☰ Menu")
    st.button("🏠 Home", on_click=set_section, args=("Home",), use_container_width=True, 
              type="primary" if st.session_state.section == "Home" else "secondary")
    st.button("ℹ️ About", on_click=set_section, args=("About",), use_container_width=True,
              type="primary" if st.session_state.section == "About" else "secondary")
    st.button("🤖 AI Assistant", on_click=set_section, args=("AI",), use_container_width=True,
              type="primary" if st.session_state.section == "AI" else "secondary")

# --------------------------------------------------
# Dynamic CSS (Background & Hover)
# --------------------------------------------------
st.markdown(f"""
<style>
.stApp {{
    background: {conf['gradient']};
    background-size: 400% 400%;
    animation: gradientFlow 15s ease infinite;
    color: white;
}}
@keyframes gradientFlow {{
    0% {{ background-position: 0% 50%; }}
    50% {{ background-position: 100% 50%; }}
    100% {{ background-position: 0% 50%; }}
}}

/* Dynamic Sidebar Backgrounds */
div[data-testid="stSidebar"] button[kind="primary"] {{
    background-color: {conf['accent']} !important;
    color: #020617 !important;
    box-shadow: 0 0 20px {conf['accent']};
    border: none !important;
}}

/* Text Hover Background Effect */
h1, h2, h3, p, span {{
    transition: 0.3s;
    border-radius: 5px;
}}
h1:hover, h2:hover, h3:hover, p:hover {{
    background: rgba(255, 255, 255, 0.05);
    color: {conf['accent']} !important;
}}

[data-testid="stVerticalBlockBorderWrapper"] {{
    background: rgba(15, 23, 42, 0.75) !important;
    backdrop-filter: blur(10px);
    border: 1px solid {conf['accent']}55 !important;
    border-radius: 20px !important;
}}

.skill-chip {{
    display: inline-block;
    padding: 4px 12px;
    border-radius: 50px;
    margin: 3px;
    font-size: 13px;
    border: 1px solid;
}}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Helper Functions
# --------------------------------------------------
def extract_skills(text):
    text = text.lower()
    return {s for s in skills_list if s in text}

# --------------------------------------------------
# Main UI
# --------------------------------------------------
st.markdown(f'<div style="font-size:42px; font-weight:800; text-align:center; color:{conf["accent"]}; text-shadow: 0 0 15px {conf["accent"]}; margin-bottom:20px;">Resume–JD Matching System</div>', unsafe_allow_html=True)

if st.session_state.section == "Home":
    with st.container(border=True):
        st.write("### 📝 Paste your resume content")
        resume_text = st.text_area("Input", height=200, label_visibility="collapsed", placeholder="Enter skills and experience...")
        
        if st.button("🚀 Analyze Matching & Skills", use_container_width=True):
            if resume_text.strip():
                # 1. Matching Logic
                r_emb = model.encode(resume_text)
                sims = cosine_similarity([r_emb], job_embeddings)[0]
                df["match_percentage"] = (sims * 100).round(2)
                
                # 2. Skill Extraction
                user_skills = extract_skills(resume_text)
                
                st.write("### Career Match Results")
                for _, row in df.sort_values("match_percentage", ascending=False).head(3).iterrows():
                    with st.expander(f"{row['Job Title']} — {row['match_percentage']}% Match"):
                        job_skills = extract_skills(row['clean_description'])
                        
                        # Display Missing Skills (The fix)
                        missing = job_skills - user_skills
                        matched = job_skills & user_skills
                        
                        st.write("**Matched Skills:**")
                        if matched:
                            for s in matched:
                                st.markdown(f'<span class="skill-chip" style="color:{conf["accent"]}; border-color:{conf["accent"]};">{s}</span>', unsafe_allow_html=True)
                        else: st.write("None detected.")

                        st.write("**Missing Skills (Unmatched):**")
                        if missing:
                            for s in missing:
                                st.markdown(f'<span class="skill-chip" style="color:#ff4b4b; border-color:#ff4b4b;">{s}</span>', unsafe_allow_html=True)
                        else: st.write("No missing skills! You're a great fit.")
            else:
                st.warning("Please enter text to analyze.")

elif st.session_state.section == "About":
    
    with st.container(border=True):
        st.write("### ⚙️ Technology Overview")
        with st.expander("🔹 Transformer: all-MiniLM-L6-v2"):
            st.write("- Maps text into a high-dimensional vector space where similar meanings are closer together.")
        with st.expander("🔹 Cosine Similarity"):
            st.write("- Calculates the match by measuring the cosine of the angle between your resume vector and the job description vector.")
        with st.expander("🔹 Skill Gap Analysis"):
            st.write("- Compares a predefined dictionary of tech skills against your text to identify missing competencies.")

elif st.session_state.section == "AI":
    with st.container(border=True):
        st.write("### 🤖 AI Assistant")
        query = st.text_input("Ask a question about the project...", placeholder="How does semantic matching work?")
        if query:
            q_emb = model.encode(query)
            sims = cosine_similarity([q_emb], kb_embeddings)[0]
            st.info(f"**Answer:** {kb_paragraphs[sims.argmax()]}")