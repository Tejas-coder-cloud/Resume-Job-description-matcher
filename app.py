import re
import time
import sqlite3
import smtplib
from email.mime.text import MIMEText
import hashlib
import random
import pdfplumber
import streamlit as st
import plotly.graph_objects as go
import google.generativeai as genai
import spacy
from sentence_transformers import SentenceTransformer, util
from langdetect import detect
from pymongo import MongoClient 
def is_valid_email(email):
    return re.match(r"[^@]+@[^@]+\.[^@]+", email)
# ✅ Send OTP via Gmail
def send_email_otp(receiver_email, otp):
    sender_email = st.secrets["EMAIL_USER"]
    app_password = st.secrets["EMAIL_PASS"]

    msg = MIMEText(f"Your OTP is: {otp}")
    msg["Subject"] = "OTP Verification"
    msg["From"] = sender_email
    msg["To"] = receiver_email

    server = smtplib.SMTP("smtp.gmail.com", 587)
    server.starttls()
    server.login(sender_email, app_password)
    server.sendmail(sender_email, receiver_email, msg.as_string())
    server.quit()
# ---------------- CONFIG ---------------- #
st.set_page_config(page_title="Resume-JD Matcher", layout="wide")
st.markdown("""
<style>
.glow-card {
    padding: 18px;
    border-radius: 14px;
    background: linear-gradient(145deg,#1e293b,#0f172a);

    /* ✅ Border + Glow */
    border: 1px solid rgba(129,140,248,0.5);
    box-shadow: 0 0 20px rgba(99,102,241,0.5);

    /* ✅ Spacing */
    margin-bottom: 18px;

    /* ✅ Smooth hover */
    transition: all 0.3s ease;
}

.glow-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 0 30px rgba(99,102,241,0.9);
}

/* ✅ Skill tags */
.tag {
    display:inline-block;
    padding:6px 10px;
    margin:5px;
    border-radius:8px;
    font-size:14px;
}

.match { 
    background:#16a34a; 
    color:white; 
}

.miss { 
    background:#dc2626; 
    color:white; 
}
</style>
""", unsafe_allow_html=True)
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY_PRIMARY", "")
st.write("API key found:", bool(GEMINI_API_KEY))
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# ---------------- MODELS ---------------- #
@st.cache_resource
def load_models():
    return spacy.load("en_core_web_sm"), SentenceTransformer('all-MiniLM-L6-v2')

nlp, transformer_model = load_models()

@st.cache_data
def cached_ai(prompt):
    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        return model.generate_content(prompt).text
    except Exception as e:
        st.error(f"AI Error: {str(e)}")
        return None

# ---------------- DATABASE ---------------- #
client = MongoClient(st.secrets["MONGO_URL"])
db = client["resume_matcher"]
users = db["users"]

def hash_pass(p): return hashlib.sha256(p.encode()).hexdigest()

# ---------------- SESSION ---------------- #
for key in ["user","resume_text","dynamic_roles","resume_lang"]:
    if key not in st.session_state:
        st.session_state[key] = None

if "page" not in st.session_state:
    st.session_state.page = "Home"

# ---------------- AUTH ---------------- #
if st.session_state.user is None:

    tabs = st.tabs(["Login","Signup","Forgot Password"])

    # ================= LOGIN ================= #
    with tabs[0]:
        u = st.text_input("Username", key="login_user")
        p = st.text_input("Password", type="password", key="login_pass")

        if st.button("Login", key="login_btn"):

            user = users.find_one({
                "username": u,
                "password": hash_pass(p)
            })
            if user:
                st.session_state.user = u
                st.success("Login successful ✅")
                st.rerun()
            else:
                st.error("Invalid credentials ❌")

    # ================= SIGNUP ================= #
    with tabs[1]:

        username = st.text_input("Username", key="signup_username")
        email = st.text_input("Email", key="signup_email")
        password = st.text_input("Password", type="password", key="signup_password")

    # 🔹 Send OTP
        if st.button("Send OTP", key="signup_send_otp"):

            if not username or not email or not password:
                st.warning("Fill all fields ❗")

            elif not is_valid_email(email):
                st.error("Enter valid email ❌")

            else:
                otp = str(random.randint(100000,999999))

                st.session_state.signup_otp = otp
                st.session_state.signup_verified = False
                st.session_state.signup_data = {
                "username": username,
                "email": email,
                "password": password
            }
                st.session_state.signup_time = time.time()

                try:
                    send_email_otp(email, otp)
                    st.success("OTP sent to email ✅")
                except Exception as e:
                    st.error(f"Email sending failed: {str(e)}")
                    st.warning("Email not configured, showing OTP below")
                    st.code(otp)

    # 🔹 Enter OTP
        otp_input = st.text_input("Enter OTP", key="signup_otp_input")

    # 🔹 Verify OTP  ✅ NOW INSIDE TAB
        if st.button("Verify OTP", key="signup_verify_btn"):

            if not st.session_state.get("signup_otp"):
                st.warning("Click 'Send OTP' first ❗")

            elif not otp_input:
                st.warning("Enter OTP ❗")

            elif time.time() - st.session_state.get("signup_time",0) > 60:
                st.error("OTP expired ❌")

            elif otp_input != st.session_state.get("signup_otp"):
                st.error("Invalid OTP ❌")

            else:
                st.session_state.signup_verified = True
                st.success("OTP Verified ✅")

    # 🔹 Register  ✅ NOW INSIDE TAB
        if st.button("Register", key="signup_register_btn"):

            if not st.session_state.get("signup_verified"):
                st.warning("Verify OTP first ❗")

            elif not st.session_state.get("signup_data"):
                st.error("Session expired ❌")

            else:
                data = st.session_state.signup_data

                if users.find_one({"username": data["username"]}):
                    st.error("Username already exists ❌")
                else:
                    users.insert_one({
                    "username": data["username"],
                    "email": data["email"],
                    "password": hash_pass(data["password"])
                })
                    st.success("Account created successfully 🎉")    


            # Reset session
                st.session_state.signup_otp = None
                st.session_state.signup_verified = False
                st.session_state.signup_data = None
    # ================= FORGOT PASSWORD ================= #
    with tabs[2]:


    # Username input
        u = st.text_input("Username", key="forgot_user")

    # Send OTP button
        if st.button("Send OTP", key="forgot_send_otp"):

            if not u:
                st.warning("Enter username first ❗")

            else:
                user = users.find_one({"username": u})

                if not user:
                    st.error("User does not exist ❌")

                else:
                    otp = str(random.randint(100000, 999999))

                # Store OTP information
                    st.session_state["forgot_otp"] = otp
                    st.session_state["otp_user"] = u
                    st.session_state["forgot_time"] = time.time()

                    st.success("OTP generated ✅")

                # For testing only
                    st.code(otp)

    # OTP and new password inputs
        otp_in = st.text_input(
        "Enter OTP",
        key="forgot_otp_input"
    )

        newp = st.text_input(
        "New Password",
        type="password",
        key="forgot_pass"
    )

    # Reset Password button
        if st.button("Reset Password", key="forgot_reset_btn"):

            if "forgot_otp" not in st.session_state:
                st.warning("Click 'Send OTP' first ❗")

            elif not otp_in or not newp:
                st.warning("Enter OTP and new password ❗")

            elif time.time() - st.session_state["forgot_time"] > 60:
                st.error("OTP expired ❌")

            elif otp_in != st.session_state["forgot_otp"]:
                st.error("Invalid OTP ❌")

            elif u != st.session_state["otp_user"]:
                st.error("Username mismatch ❌")

            else:
                users.update_one(
                    {"username:": u},
                    {
                        "$set": {
                            password: hash_pass(newp)
                        }
                    }
                )
                st.success("Password updated successfully ✅")


            # Clear OTP data
                del st.session_state["forgot_otp"]
                del st.session_state["otp_user"]
                del st.session_state["forgot_time"]
# ---------------- MAIN APP ---------------- #
else:

    # Sidebar
    with st.sidebar:
        st.write(f"👤 {st.session_state.user}")
        for p in ["Home","Analytics","Salary","Assistant","About"]:
            if st.button(p):
                st.session_state.page = p
                st.rerun()
        if st.button("Logout"):
            st.session_state.user=None
            st.rerun()

    # ================= HOME ================= #
    if st.session_state.page == "Home":
        st.subheader("Resume Analysis")
        file = st.file_uploader("Upload Resume", type="pdf")

        if file:

            text = " ".join(
        [p.extract_text() or "" for p in pdfplumber.open(file).pages]
    )

            st.session_state.resume_text = text

    # ================= LANGUAGE DETECTION =================

            raw_lang_code = detect(text)

            lang_map = {
        "hi": "Hindi",
        "es": "Spanish",
        "fr": "French",
        "de": "German",
        "mr": "Marathi",
        "en": "English"
    }

            target_lang = lang_map.get(raw_lang_code, "English")

    # ================= ROLE EXTRACTION =================

            roles_raw = cached_ai(
        f"""
        Return ONLY 3 suitable job roles
        in {target_lang}
        separated by commas.

        Resume:
        {text[:1000]}
        """
    )

            roles = (
        [r.strip() for r in roles_raw.split(",")]
        if roles_raw
        else ["Software Engineer"]
    )

            st.session_state.dynamic_roles = roles

    # ================= DOWNLOAD REPORT =================

            full_report_content = (
        f"RESUME ANALYSIS REPORT ({target_lang})\n"
        + "=" * 40 + "\n\n"
    )

    # ================= ROLE LOOP =================

            for role in roles:

        # ---------------- SKILL EXTRACTION ----------------

                skills_raw = cached_ai(
            f"""
            List exactly 6 technical skills
            required for {role}.

            Return ONLY comma separated values.
            """
        ) or "Python, SQL"

                skills_list = [
            s.strip().lower()
            for s in skills_raw.split(",")
            if s.strip()
        ]

        # ---------------- SBERT MATCHING ----------------

                matched = []
                missing = []

                resume_chunks = [
            chunk.strip()
                for chunk in re.split(
                r"[.\n,;:]",
                text.lower()
            )
                if chunk.strip()
        ]

            resume_embeddings = transformer_model.encode(
            resume_chunks,
            convert_to_tensor=True
        )

            for skill in skills_list:

                skill_embedding = transformer_model.encode(
                skill,
                convert_to_tensor=True
            )

                similarities = util.pytorch_cos_sim(
                skill_embedding,
                resume_embeddings
            )[0]

                max_similarity = similarities.max().item()

                if max_similarity >= 0.35:
                    matched.append(skill)
                else:
                    missing.append(skill)

            score = (
            len(matched)
            / len(skills_list)
            * 100
            if skills_list
            else 0
        )

        # ---------------- IMPROVEMENT TEXT ----------------

            improvement = cached_ai(
            f"""
            Act as a professional resume reviewer.

            Language: {target_lang}

            Role: {role}

            Write in plain text.

            Do NOT use:
            - markdown
            - bullets
            - headings
            - asterisks

            Explain:
            1. Resume improvements
            2. Missing skills
            3. Project suggestions

            Resume:
            {text[:1500]}
            """
        )

        # ---------------- UI CARD ----------------

            st.markdown(
            f"""
            <div class='glow-card'>

            <h3 style='color:#818cf8;'>
            {role} — {score:.1f}% Match
            </h3>

            <p>
            <b>Matched Skills:</b>
            {" ".join([f"<span class='tag match'>{m}</span>" for m in matched])}
            </p>

            <p>
            <b>Missing Skills:</b>
            {" ".join([f"<span class='tag miss'>{m}</span>" for m in missing])}
            </p>

            <div style='
                background:rgba(255,255,255,0.05);
                padding:15px;
                border-radius:10px;
                border-left:4px solid #818cf8;
                white-space:pre-wrap;
            '>

            {improvement}

            </div>

            </div>
            """,
            unsafe_allow_html=True
        )

        # ---------------- REPORT CONTENT ----------------

            full_report_content += (
            f"ROLE: {role}\n"
            f"MATCH SCORE: {score:.1f}%\n\n"
            f"MATCHED SKILLS:\n"
            f"{', '.join(matched)}\n\n"
            f"MISSING SKILLS:\n"
            f"{', '.join(missing)}\n\n"
            f"RECOMMENDATIONS:\n"
            f"{improvement}\n\n"
            + "=" * 40
            + "\n\n"
        )

    # ================= DOWNLOAD BUTTON =================

            st.download_button(
        label="📥 Download Full Analysis Report",
        data=full_report_content,
        file_name=f"Resume_Report_{raw_lang_code}.txt",
        mime="text/plain"
    )
    # ================= ANALYTICS ================= #
    elif st.session_state.page=="Analytics":
        if st.session_state.resume_text:
            def normalize_text(text):
                lang = detect(text)
                if lang != "en":
                    translated = cached_ai(f"Translate this to English:\n{text[:2000]}")
                    return translated.lower() if translated else text.lower()
                return text.lower()
            text = normalize_text(st.session_state.resume_text)
            tech={"Programming":["python","java"],"Cloud":["aws","azure"],"Data":["sql","pandas"]}
            counts=[sum(text.count(s) for s in v) for v in tech.values()]
            fig=go.Figure(go.Pie(labels=list(tech.keys()),values=counts))
            st.plotly_chart(fig)
        else:
            st.warning("Upload resume first")

    # ================= SALARY ================= #
    elif st.session_state.page == "Salary":
        
        if not st.session_state.resume_text:
             st.warning("Please upload your resume on the Home page first.")
        else:     

            role = st.selectbox(
            "Select Target Role",
            st.session_state.dynamic_roles
        )

            exp = st.slider(
            "Years of Experience",
            0,
            25,
            2
        )

            if st.button("Predict Market Salary"):

            # ================= AI PROMPT ================= #
                prompt = f"""
Give salary range in India for {role} with {exp} years experience.

STRICT FORMAT:
Only numbers, no symbols, no commas
Example: 600000-1200000

Do not write anything else.
"""

                res = cached_ai(prompt)
                success = False

            # ================= AI RESULT PARSING ================= #
                if res:

                    nums = re.findall(r"\d+", res)

                    if len(nums) >= 2:

                        low = int(nums[0])
                        high = int(nums[1])

                    # Fix small numbers (13 -> 13L)
                        if low < 100000:
                            low *= 100000
                            high *= 100000

                    # Sanity checks
                        if low < 200000:
                            low = 300000

                        if high < low:
                            high = low + 300000

                        st.balloons()

                        st.metric(
                        "Estimated Annual Package (AI Analyzed)",
                        f"₹{low:,} - ₹{high:,}"
                    )

                        st.caption(
                        "✨ Real-time market analysis based on your specific tech stack."
                    )

                        success = True

            # ================= FALLBACK ================= #
                if not success:

                    st.warning(
                    "AI service unavailable. Showing baseline estimate."
                )

                    base_val = 400000 + (exp * 250000)

                    if "Data" in role or "Senior" in role:
                        base_val += 200000

                    low = int(base_val * 0.8)
                    high = int(base_val * 1.3)

                    st.metric(
                    "Estimated Annual Package (Baseline)",
                    f"₹{low:,} - ₹{high:,}"
                )

                    st.info(
                    "This is a general estimate based on experience."
                )

    
    # ================= ASSISTANT ================= #
    elif st.session_state.page=="Assistant":
        q = st.text_input("Ask something")

        if st.button("Ask AI"):

            if not q:
                st.warning("Please enter a question")
            else:
                ans = cached_ai(q)

                if ans:
                    st.write(ans)
                else:
                    st.error("AI not responding. Check API key or internet.")

    # ================= ABOUT ================= #
    elif st.session_state.page=="About":
        st.markdown("<h2 style='text-align:center;'>System Intelligence</h2>", unsafe_allow_html=True)

        cards = [
        ("Linguistic AI", "Detects resume language and provides localized career advice."),
        ("SBERT Embeddings", "Calculates semantic similarity between resume and job roles."),
        ("Gemini 2.5", "Generative intelligence for dynamic role inference and improvement tips."),
        ("SHA-256", "Secure cryptographic hashing for account and OTP authentication."),
        ("Visual Analytics", "Plotly-driven distribution charts for layman-friendly insights.")
        ]

        for t, d in cards:
            st.markdown(f"""
        <div class='glow-card'>
            <h4 style='color:#818cf8; margin-bottom:8px;'>{t}</h4>
            <p style='color:#e5e7eb;'>{d}</p>
        </div>
        """, unsafe_allow_html=True)