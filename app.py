# ============================================================
# Resume–JD Matcher | FINAL STABLE VERSION (UPDATED)
# ============================================================
#
# Changes made:
# - Configure google.generativeai once at startup (if API key present).
# - Replaced get_ai_response with a robust helper that:
#     * Tries multiple common client call patterns (generate_text, model.generate, model.generate_content)
#     * On error attempts to list available models to provide helpful diagnostics
#     * Returns a readable string for the UI instead of raw exceptions
# - Added a local model-listing helper (used for diagnostics)
# - Minor logging and safer access to st.secrets
#
# Note: Keep your credentials in Streamlit secrets or environment variables.
# Add these keys to .streamlit/secrets.toml or your Streamlit configuration:
#   AI_API_KEY = "..."
#   AI_MODEL = "gemini-1.5-flash"   # set this after running the model lister if needed
#
# If you intend to use OpenAI instead of Google Vertex/Gemini, replace the google.generativeai
# usage with the OpenAI client calls accordingly.
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import hashlib
import random
import smtplib
import re
from email.message import EmailMessage
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai  # keep if using Google generative AI

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
# GENERATIVE AI CONFIGURATION & HELPERS
# ------------------------------------------------------------
# We'll configure the generative client once (if an API key is available in st.secrets).
# We also provide helpers to list models and get responses in a robust way.

# Global flag so we don't attempt to re-configure unnecessarily
_GENAI_CONFIGURED = False

def configure_genai_from_secrets():
    global _GENAI_CONFIGURED
    if _GENAI_CONFIGURED:
        return True

    # st.secrets might be absent in some contexts; use .get to avoid KeyError
    api_key = None
    try:
        api_key = st.secrets.get("AI_API_KEY")
    except Exception:
        # st.secrets may not exist or be inaccessible; fall back to env var if needed
        api_key = None

    if not api_key:
        # Not configured - do not treat as fatal here; callers will surface helpful message
        return False

    try:
        genai.configure(api_key=api_key)
        _GENAI_CONFIGURED = True
        return True
    except Exception as e:
        # configuration failed, keep flag false
        st.warning(f"Warning: failed to configure generative client: {e}")
        return False


def list_available_models(api_key_override=None):
    """
    Attempts to return a list of available model ids/names using the configured client.
    Returns a list (possibly empty) or raises a RuntimeError if listing is not possible.
    """
    # If an override API key is provided, configure with it for the listing call
    if api_key_override:
        try:
            genai.configure(api_key=api_key_override)
        except Exception as e:
            raise RuntimeError(f"Failed to configure genai with provided key: {e}")

    # Ensure client is configured (if not, try to configure from secrets)
    if not _GENAI_CONFIGURED:
        configure_genai_from_secrets()

    try:
        # client versions vary; try common listing entrypoints
        resp = None
        try:
            resp = genai.list_models()
        except Exception:
            try:
                resp = genai.get_models()
            except Exception:
                resp = None

        if resp is None:
            # As a last attempt some clients expose 'models' as attribute on genai
            if hasattr(genai, "models"):
                try:
                    resp = genai.models.list()
                except Exception:
                    resp = None

        models = []

        if resp is None:
            raise RuntimeError("Model listing API not available for the configured client/version.")

        # Normalize several response shapes
        if isinstance(resp, dict):
            for key in ("models", "data"):
                if key in resp:
                    for m in resp[key]:
                        if isinstance(m, dict):
                            models.append(m.get("name") or m.get("id") or str(m))
                        else:
                            # object-like entry
                            models.append(getattr(m, "name", None) or getattr(m, "id", None) or str(m))
                    break
        elif isinstance(resp, list):
            for m in resp:
                if isinstance(m, dict):
                    models.append(m.get("name") or m.get("id") or str(m))
                else:
                    models.append(getattr(m, "name", None) or getattr(m, "id", None) or str(m))
        else:
            # object-like response with attributes .models or .data
            arr = getattr(resp, "models", None) or getattr(resp, "data", None)
            if arr:
                for m in arr:
                    models.append(getattr(m, "name", None) or getattr(m, "id", None) or str(m))

        return models

    except Exception as e:
        raise RuntimeError(f"Failed to list models: {e}")


def _extract_text_from_response(response):
    """
    Attempt to extract a readable text string from different response shapes.
    """
    if response is None:
        return ""
    # Many Python client objects expose .text
    if hasattr(response, "text"):
        try:
            return response.text
        except Exception:
            pass
    # Another common field
    if hasattr(response, "output"):
        try:
            return getattr(response, "output")
        except Exception:
            pass
    # If it's a dict, try common keys
    if isinstance(response, dict):
        # candidates -> output/content
        if "candidates" in response and response["candidates"]:
            c = response["candidates"][0]
            if isinstance(c, dict):
                return c.get("output") or c.get("content") or str(c)
            else:
                return str(c)
        for key in ("content", "text", "output"):
            if key in response:
                return response.get(key)
        # fallback to stringifying
        return str(response)
    # Fallback: any attribute that looks like text
    for attr in ("response", "content", "result"):
        if hasattr(response, attr):
            try:
                return getattr(response, attr)
            except Exception:
                pass
    return str(response)


def get_ai_response(prompt, model_name=None, max_tokens=256):
    """
    Robust AI helper: tries multiple call patterns and returns a string.
    If the model is not found or a client error occurs, attempts to list available models
    and returns an informative error string that is safe to display in the UI.
    """
    # Ensure client is configured (if possible)
    configured = configure_genai_from_secrets()

    # Determine model_name: prefer provided, then secrets, then fallback default
    preferred_model = model_name or st.secrets.get("AI_MODEL") if hasattr(st, "secrets") else None
    preferred_model = preferred_model or "gemini-1.5-flash"

    if not configured:
        # Provide immediate helpful feedback rather than raising
        return ("AI error: generative client not configured. "
                "Set AI_API_KEY in Streamlit secrets or environment. "
                f"Attempted model: {preferred_model}")

    try:
        response = None

        # Try common call patterns in order (client versions differ)
        # Pattern 1: top-level convenience method (generate_text)
        try:
            response = genai.generate_text(model=preferred_model, prompt=prompt, max_output_tokens=max_tokens)
        except Exception:
            response = None

        # Pattern 2: construct a model object and call generate / generate_content
        if response is None:
            try:
                model_obj = genai.GenerativeModel(preferred_model)
                # try model_obj.generate (some versions)
                try:
                    response = model_obj.generate(prompt=prompt, max_output_tokens=max_tokens)
                except Exception:
                    # try generate_content as another possible method name
                    response = model_obj.generate_content(prompt)
            except Exception:
                response = None

        # Pattern 3: some modern clients support .responses.create or similar shapes
        if response is None:
            try:
                # Example: genai.responses.create(model=..., input=...)
                if hasattr(genai, "responses") and hasattr(genai.responses, "create"):
                    response = genai.responses.create(model=preferred_model, input=prompt, max_output_tokens=max_tokens)
            except Exception:
                response = None

        # If still None, raise an error so that listing logic runs
        if response is None:
            raise RuntimeError("No supported client method succeeded for model call.")

        # Extract text safely
        text = _extract_text_from_response(response)
        # Some responses may include structured lists or newline tokens; return as string
        if text is None:
            text = ""
        return str(text)

    except Exception as e:
        # On error, attempt to list available models for diagnostics
        try:
            models = list_available_models()
            models_summary = ", ".join(models[:20]) if isinstance(models, list) else str(models)
        except Exception as list_err:
            models_summary = f"Failed to list models: {list_err}"

        return f"AI error: {e}. Available models: {models_summary}"


# ------------------------------------------------------------
# DATABASE
# ------------------------------------------------------------
def hash_pw(p):
    return hashlib.sha256(p.encode()).hexdigest()


def init_db():

    conn = sqlite3.connect("users.db")
    cur = conn.cursor()

    cur.execute("""
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

    otp = str(random.randint(100000, 999999))

    msg = EmailMessage()
    msg.set_content(f"Your OTP is: {otp}")
    msg["Subject"] = "Verification Code"
    msg["From"] = st.secrets.get("EMAIL_USER", "")
    msg["To"] = email

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as s:
        s.login(st.secrets.get("EMAIL_USER"), st.secrets.get("EMAIL_PASSWORD"))
        s.send_message(msg)

    return otp


# ------------------------------------------------------------
# LOAD MODEL
# ------------------------------------------------------------
@st.cache_resource
def load_model():

    model = SentenceTransformer("all-MiniLM-L6-v2")

    descriptions = {
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

    df = pd.DataFrame(list(descriptions.items()), columns=["Role", "description"])
    embeddings = model.encode(df["description"].tolist())

    return model, df, embeddings


model, jobs_df, job_embeddings = load_model()


# ------------------------------------------------------------
# SESSION STATE
# ------------------------------------------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "auth_step" not in st.session_state:
    st.session_state.auth_step = "login"

if "page" not in st.session_state:
    st.session_state.page = "Home"

if "chat" not in st.session_state:
    st.session_state.chat = []


# ============================================================
# AUTH
# ============================================================
if not st.session_state.logged_in:

    st.title("Authentication")

    if st.session_state.auth_step == "login":

        with st.form("login"):

            username = st.text_input("Username")
            password = st.text_input("Password", type="password")

            submit = st.form_submit_button("Login")

        if submit:

            conn = sqlite3.connect("users.db")
            cur = conn.cursor()

            cur.execute(
                "SELECT * FROM users WHERE username=? AND password=?",
                (username, hash_pw(password))
            )

            if cur.fetchone():

                st.session_state.logged_in = True
                st.rerun()

            else:
                st.error("Invalid credentials")

            conn.close()

        col1, col2 = st.columns(2)

        if col1.button("New User? Sign Up"):
            st.session_state.auth_step = "signup"
            st.rerun()

        if col2.button("Forgot Password"):
            st.session_state.auth_step = "forgot"
            st.rerun()


    elif st.session_state.auth_step == "signup":

        user = st.text_input("Username")
        email = st.text_input("Email")
        pw = st.text_input("Password", type="password")

        if st.button("Send OTP"):

            st.session_state.otp = send_otp(email)
            st.session_state.temp = (user, hash_pw(pw), email)
            st.session_state.auth_step = "verify"

            st.success("OTP sent")
            st.rerun()


    elif st.session_state.auth_step == "verify":

        otp = st.text_input("Enter OTP")

        if st.button("Verify"):

            if otp == st.session_state.otp:

                conn = sqlite3.connect("users.db")
                conn.execute(
                    "INSERT INTO users VALUES (?,?,?)",
                    st.session_state.temp
                )
                conn.commit()
                conn.close()

                st.success("Account created")
                st.session_state.auth_step = "login"

            else:
                st.error("Invalid OTP")


    elif st.session_state.auth_step == "forgot":

        user = st.text_input("Username")
        email = st.text_input("Email")

        if st.button("Send Reset OTP"):

            conn = sqlite3.connect("users.db")
            cur = conn.cursor()

            cur.execute(
                "SELECT * FROM users WHERE username=? AND email=?",
                (user, email)
            )

            if cur.fetchone():

                st.session_state.otp = send_otp(email)
                st.session_state.reset = (user, email)
                st.session_state.auth_step = "reset"
                st.rerun()

            else:
                st.error("User not found. Please sign up first.")


    elif st.session_state.auth_step == "reset":

        otp = st.text_input("OTP")
        newpw = st.text_input("New Password", type="password")

        if st.button("Reset"):

            if otp == st.session_state.otp:

                conn = sqlite3.connect("users.db")
                conn.execute(
                    "UPDATE users SET password=? WHERE username=?",
                    (hash_pw(newpw), st.session_state.reset[0])
                )
                conn.commit()
                conn.close()

                st.success("Password updated")
                st.session_state.auth_step = "login"

            else:
                st.error("Invalid OTP")

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.title("Menu")

if st.sidebar.button("Home"):
    st.session_state.page = "Home"

if st.sidebar.button("About"):
    st.session_state.page = "About"

if st.sidebar.button("AI Assistant"):
    st.session_state.page = "AI"

if st.sidebar.button("Logout"):
    st.session_state.logged_in = False
    st.rerun()


# ============================================================
# HOME
# ============================================================
if st.session_state.page == "Home":

    set_background("#0f172a", "#1e293b")

    st.title("Resume-JD Matcher")

    with st.form("resume"):

        exp = st.slider("Experience", 0, 20, 0)
        resume = st.text_area("Paste Resume / Skills")

        submit = st.form_submit_button("Analyze")

    if submit:

        skills = list(set(re.findall(r"[a-zA-Z\+\#]{2,}", resume.lower())))

        resume_emb = model.encode([resume])
        sims = cosine_similarity(resume_emb, job_embeddings)[0]

        for i in sims.argsort()[::-1][:3]:

            role = jobs_df.iloc[i]["Role"]
            desc = jobs_df.iloc[i]["description"]
            role_tokens = desc.split()

            matched = [s for s in skills if s in role_tokens]
            unmatched = [s for s in skills if s not in role_tokens]

            score = sims[i]

            base_salary = 3 + score * 3
            experience_bonus = exp * 0.7
            salary = base_salary + experience_bonus

            prompt = f"""
You are a recruiter.

Role: {role}
Matched Skills: {matched}
Missing Skills: {unmatched}

Explain candidate suitability in 4 bullet points.
"""

            explanation = get_ai_response(prompt).replace("\n", "<br>")

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
            """, unsafe_allow_html=True)


# ============================================================
# ABOUT
# ============================================================
elif st.session_state.page == "About":

    set_background("#1e3a8a", "#0f172a")

    st.title("System Architecture")

    sections = {
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

    for t, d in sections.items():

        st.markdown(f"""
        <div class="card">
        <b>{t}</b><br><br>{d}
        </div>
        """, unsafe_allow_html=True)


# ============================================================
# AI ASSISTANT
# ============================================================
elif st.session_state.page == "AI":

    set_background("#312e81", "#0f172a")

    st.title("Career AI Assistant")

    for m in st.session_state.chat:
        st.markdown(f"<div class='chat-user'><b>You:</b> {m['u']}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='chat-ai'><b>AI:</b> {m['a']}</div>", unsafe_allow_html=True)

    with st.form("chat"):

        q = st.text_input("Ask a career question")

        send = st.form_submit_button("Send")

    if send and q:

        prompt = f"You are a career advisor. Answer: {q}"

        ans = get_ai_response(prompt)

        st.session_state.chat.append({"u": q, "a": ans})
        st.rerun()

    if st.button("Clear Chat History"):
        st.session_state.chat = []
        st.rerun()