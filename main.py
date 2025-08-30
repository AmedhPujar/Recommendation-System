import os
import re
import json
import datetime
import sqlite3
import pandas as pd
import streamlit as st
from passlib.hash import bcrypt
import jwt

# ====== NEW: AI (Gemini + LangChain) ======
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ========== CONFIGURATION ==========
DB_PATH = 'career_connector.db'
JWT_SECRET = os.getenv("JWT_SECRET", "change-me-in-prod")   # Prefer env var
JWT_ALGORITHM = 'HS256'
JOBDATA_PATH = os.path.join(os.path.dirname(__file__), 'Career_connector_app', 'JobData.csv')
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")  # Must be set for Gemini usage

# ========== DATABASE SETUP ==========
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    c = conn.cursor()
    # Recruiters
    c.execute('''CREATE TABLE IF NOT EXISTS recruiters (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        email TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL,
        company TEXT NOT NULL
    )''')
    # Jobs
    c.execute('''CREATE TABLE IF NOT EXISTS jobs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT NOT NULL,
        description TEXT NOT NULL,
        skills TEXT NOT NULL,
        salary TEXT NOT NULL,
        location TEXT NOT NULL,
        eligibility TEXT NOT NULL,
        recruiter_email TEXT NOT NULL
    )''')
    # NEW: Students (discoverable profiles)
    c.execute('''CREATE TABLE IF NOT EXISTS students (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        email TEXT UNIQUE NOT NULL,
        skills TEXT NOT NULL,
        discoverable INTEGER NOT NULL DEFAULT 1,
        updated_at TEXT NOT NULL
    )''')
    conn.commit()
    conn.close()

init_db()

# ========== AUTH HELPERS ==========
def hash_password(password):
    return bcrypt.hash(password)

def verify_password(password, hashed):
    return bcrypt.verify(password, hashed)

def create_jwt(payload):
    payload = dict(payload)
    payload['exp'] = datetime.datetime.utcnow() + datetime.timedelta(hours=6)
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

def decode_jwt(token):
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None

# ========== JOB DATA LOADING ==========
@st.cache_data(show_spinner=False)
def load_jobdata():
    try:
        df = pd.read_csv(JOBDATA_PATH)
        # Normalize column names expected: Domain, Job Role, Skills, Personality (as in your sample)
        # Fallbacks for missing columns
        if 'Skills' not in df.columns:
            # Try to infer or create
            df['Skills'] = ''
        # Fillna to avoid issues
        for col in df.columns:
            df[col] = df[col].fillna('')
        return df
    except Exception as e:
        st.sidebar.warning(f"JobData.csv failed to load: {e}")
        return pd.DataFrame(columns=['Domain','Job Role','Skills','Personality'])

jobdata_df = load_jobdata()

# ========== AI RECOMMENDER (HYBRID) ==========
def get_llm():
    """
    Returns a Gemini Flash LLM via LangChain.
    If GOOGLE_API_KEY is missing, returns None (falls back to TF-IDF only).
    """
    if not GOOGLE_API_KEY:
        return None
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.3,
            max_output_tokens=1024
        )
        return llm
    except Exception as e:
        st.error(f"Failed to initialize Gemini: {e}")
        return None

def tfidf_retrieve_top_k(student_skills: str, corpus: pd.DataFrame, k: int = 15):
    """
    Fast retrieval: find top-k jobs by cosine similarity on 'Skills'.
    Returns a DataFrame subset with a 'Score' column.
    """
    if corpus.empty:
        return corpus

    corpus_skills = corpus['Skills'].astype(str).fillna('')
    texts = corpus_skills.tolist() + [student_skills]
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf = vectorizer.fit_transform(texts)

    student_vec = tfidf[-1]
    job_vecs = tfidf[:-1]
    scores = cosine_similarity(student_vec, job_vecs).flatten()
    out = corpus.copy()
    out['Score'] = scores
    out = out.sort_values('Score', ascending=False).head(k)
    return out

def df_to_compact_list_of_dicts(df: pd.DataFrame, fields):
    items = []
    for _, row in df.iterrows():
        item = {f: str(row.get(f, "")) for f in fields}
        items.append(item)
    return items

def extract_json(text: str):
    """
    Extract first JSON array/object from model output.
    """
    try:
        # Try direct load
        return json.loads(text)
    except Exception:
        pass

    # Try fenced code block ```json ... ```
    m = re.search(r"```json\s*(.+?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass

    # Try any JSON-looking block
    m = re.search(r"(\[.*\]|\{.*\})", text, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass

    return None

# Prompt: Given student skills + candidate jobs (retrieved), select top-N with reasons and normalized score.
student_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are an AI career advisor. Given a student's skills and a small set of candidate jobs, "
     "rank the best matches and explain briefly why. Return STRICT JSON with fields: "
     "[{job_role, domain, skills, reason, fit_score}] where fit_score is 0-1."),
    ("human",
     "Student skills: {skills}\n\nCandidate jobs (list of dicts):\n{jobs}\n\n"
     "Output JSON only (no prose), top {topn} items, best first.")
])

# Prompt: Given a single job and many student profiles, select top students for that job.
recruiter_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are an AI recruiter assistant. Given one job and a list of students (name, email, skills), "
     "rank the best student matches and justify briefly. Return STRICT JSON with fields: "
     "[{name, email, reason, fit_score}] where fit_score is 0-1."),
    ("human",
     "Job: {job}\n\nStudents: {students}\n\nOutput JSON only (no prose), top {topn} items, best first.")
])

def recommend_jobs_for_student(student_skills: str, job_df: pd.DataFrame, topn: int = 5):
    """
    Hybrid: TF-IDF retrieve -> Gemini ranks & explains.
    Falls back to pure TF-IDF if no API key.
    """
    candidates = tfidf_retrieve_top_k(student_skills, job_df, k=max(10, topn*3))
    if candidates.empty:
        return []

    llm = get_llm()
    # If LLM unavailable -> return TF-IDF with simple reasons
    if llm is None:
        out = []
        for _, r in candidates.head(topn).iterrows():
            out.append({
                "job_role": str(r.get('Job Role', r.get('title',''))),
                "domain": str(r.get('Domain','')),
                "skills": str(r.get('Skills','')),
                "reason": "TF-IDF match based on overlapping skills.",
                "fit_score": float(r.get('Score', 0))
            })
        return out

    fields = ['Domain','Job Role','Skills']
    jobs_payload = df_to_compact_list_of_dicts(candidates, fields)
    chain = student_prompt | llm
    resp = chain.invoke({
        "skills": student_skills,
        "jobs": json.dumps(jobs_payload, ensure_ascii=False),
        "topn": topn
    })
    data = extract_json(resp.content or "")
    if not isinstance(data, list):
        # Fallback to TF-IDF topn
        out = []
        for _, r in candidates.head(topn).iterrows():
            out.append({
                "job_role": str(r.get('Job Role', r.get('title',''))),
                "domain": str(r.get('Domain','')),
                "skills": str(r.get('Skills','')),
                "reason": "TF-IDF match based on overlapping skills.",
                "fit_score": float(r.get('Score', 0))
            })
        return out
    # Coerce types
    for item in data:
        try:
            item["fit_score"] = float(item.get("fit_score", 0))
        except Exception:
            item["fit_score"] = 0.0
    return data[:topn]

def recommend_students_for_job(job_row: sqlite3.Row, students: list, topn: int = 5):
    """
    Given a recruiter job and discoverable students, return ranked student matches with reasons.
    """
    if not students:
        return []

    llm = get_llm()
    # Fallback: simple TF-IDF-ish scoring on skills overlap
    if llm is None:
        job_sk = set([s.strip().lower() for s in str(job_row['skills']).split(",") if s.strip()])
        scored = []
        for s in students:
            ss = set([x.strip().lower() for x in s['skills'].split(",") if x.strip()])
            overlap = len(job_sk & ss)
            score = overlap / max(1, len(job_sk))
            scored.append({"name": s['name'], "email": s['email'], "reason":"Skill overlap", "fit_score": round(score, 3)})
        scored.sort(key=lambda x: x["fit_score"], reverse=True)
        return scored[:topn]

    job_payload = {
        "title": job_row['title'],
        "skills": job_row['skills'],
        "location": job_row['location'],
        "description": job_row['description'],
        "eligibility": job_row['eligibility']
    }
    students_payload = [{"name": s["name"], "email": s["email"], "skills": s["skills"]} for s in students]
    chain = recruiter_prompt | llm
    resp = chain.invoke({
        "job": json.dumps(job_payload, ensure_ascii=False),
        "students": json.dumps(students_payload, ensure_ascii=False),
        "topn": topn
    })
    data = extract_json(resp.content or "")
    if not isinstance(data, list):
        # Fallback to simple scoring
        job_sk = set([s.strip().lower() for s in str(job_row['skills']).split(",") if s.strip()])
        scored = []
        for s in students:
            ss = set([x.strip().lower() for x in s['skills'].split(",") if x.strip()])
            overlap = len(job_sk & ss)
            score = overlap / max(1, len(job_sk))
            scored.append({"name": s['name'], "email": s['email'], "reason":"Skill overlap", "fit_score": round(score, 3)})
        scored.sort(key=lambda x: x["fit_score"], reverse=True)
        return scored[:topn]
    # Coerce types
    for item in data:
        try:
            item["fit_score"] = float(item.get("fit_score", 0))
        except Exception:
            item["fit_score"] = 0.0
    return data[:topn]


# ========== STREAMLIT UI ==========
st.set_page_config(page_title="Career Connector App (AI-powered)", layout="wide")
st.title("Career Connector App")
tabs = st.tabs(["Recruiter", "Student"])

# ========== RECRUITER TAB ==========
with tabs[0]:
    st.header("Recruiter Portal")
    menu = st.radio("Select Option", ["Signup", "Login", "Post Job", "My Jobs"], horizontal=True)

    if menu == "Signup":
        st.subheader("Create a Recruiter Account")
        name = st.text_input("Name", key="r_name")
        email = st.text_input("Email", key="r_email")
        password = st.text_input("Password", type="password", key="r_pass")
        company = st.text_input("Company", key="r_company")
        if st.button("Sign Up"):
            if not name or not email or not password or not company:
                st.warning("Please fill in all fields.")
            else:
                conn = get_db()
                c = conn.cursor()
                c.execute("SELECT 1 FROM recruiters WHERE email=?", (email,))
                if c.fetchone():
                    st.error("Email already exists.")
                else:
                    hashed = hash_password(password)
                    c.execute(
                        "INSERT INTO recruiters (name, email, password, company) VALUES (?, ?, ?, ?)",
                        (name, email, hashed, company)
                    )
                    conn.commit()
                    st.success("Recruiter created successfully!")
                conn.close()

    elif menu == "Login":
        st.subheader("Recruiter Login")
        email = st.text_input("Email", key="login_email")
        password = st.text_input("Password", type="password", key="login_pass")
        if st.button("Login"):
            conn = get_db()
            c = conn.cursor()
            c.execute("SELECT * FROM recruiters WHERE email=?", (email,))
            user = c.fetchone()
            conn.close()
            if not user or not verify_password(password, user['password']):
                st.error("Invalid credentials.")
            else:
                token = create_jwt({"email": user['email'], "name": user['name']})
                st.session_state['recruiter_token'] = token
                st.success("Login successful!")
                st.info("You can now post jobs and view your jobs.")

    elif menu == "Post Job":
        token = st.session_state.get('recruiter_token')
        user = decode_jwt(token) if token else None
        if not user:
            st.warning("Please login as a recruiter to post jobs.")
        else:
            st.subheader("Post a Job")
            title = st.text_input("Job Title", key="job_title")
            description = st.text_area("Job Description", key="job_desc")
            skills = st.text_input("Skills Required (comma separated)", key="job_skills")
            salary = st.text_input("Salary Offered", key="job_salary")
            location = st.text_input("Job Location", key="job_location")
            eligibility = st.text_area("Eligibility Criteria", key="job_eligibility")
            if st.button("Post Job"):
                if not title or not description or not skills or not salary or not location or not eligibility:
                    st.warning("Please fill in all fields.")
                else:
                    conn = get_db()
                    c = conn.cursor()
                    c.execute("""INSERT INTO jobs
                                 (title, description, skills, salary, location, eligibility, recruiter_email)
                                 VALUES (?, ?, ?, ?, ?, ?, ?)""",
                              (title, description, skills, salary, location, eligibility, user['email']))
                    conn.commit()
                    conn.close()
                    st.success("Job posted successfully!")

    elif menu == "My Jobs":
        token = st.session_state.get('recruiter_token')
        user = decode_jwt(token) if token else None
        if not user:
            st.warning("Please login as a recruiter to view your jobs.")
        else:
            st.subheader("Your Posted Jobs")
            conn = get_db()
            c = conn.cursor()
            c.execute("SELECT * FROM jobs WHERE recruiter_email=?", (user['email'],))
            jobs = c.fetchall()

            # Load discoverable students
            c.execute("SELECT name, email, skills FROM students WHERE discoverable=1 ORDER BY updated_at DESC LIMIT 100")
            students_raw = c.fetchall()
            conn.close()

            students = [{"name": s["name"], "email": s["email"], "skills": s["skills"]} for s in students_raw]

            if not jobs:
                st.info("No jobs posted yet.")
            else:
                for job in jobs:
                    st.markdown(f"""
**Title:** {job['title']}  
**Location:** {job['location']}  
**Salary:** {job['salary']}  
**Skills:** {job['skills']}  
**Eligibility:** {job['eligibility']}  
—""")

                    with st.expander("AI-Recommended Students for this Job"):
                        if not students:
                            st.info("No discoverable student profiles yet.")
                        else:
                            with st.spinner("Finding best-fit students..."):
                                rec_students = recommend_students_for_job(job, students, topn=5)
                            if not rec_students:
                                st.info("No suitable students found right now.")
                            else:
                                for s in rec_students:
                                    st.markdown(
                                        f"- **{s['name']}** ({s['email']}) — Fit: `{s['fit_score']:.2f}`  \n"
                                        f"  _Reason_: {s['reason']}"
                                    )

# ========== STUDENT TAB ==========
with tabs[1]:
    st.header("Student Portal")
    st.write("Get **AI-powered job recommendations** based on your skills (Gemini Flash + TF-IDF retrieval).")

    st.subheader("Your Profile")
    col1, col2 = st.columns(2)
    with col1:
        student_name = st.text_input("Name", key="s_name")
        student_email = st.text_input("Email", key="s_email")
    with col2:
        student_skills = st.text_input("Skills (comma separated)", key="student_skills")
        discoverable = st.checkbox("Make my profile discoverable to recruiters", value=True)

    if st.button("Save Profile & Get Recommendations"):
        if not student_name or not student_email or not student_skills:
            st.warning("Please fill in name, email, and skills.")
        else:
            # Upsert profile
            conn = get_db()
            c = conn.cursor()
            c.execute("SELECT id FROM students WHERE email=?", (student_email,))
            row = c.fetchone()
            now = datetime.datetime.utcnow().isoformat()
            if row:
                c.execute("""UPDATE students
                             SET name=?, skills=?, discoverable=?, updated_at=?
                             WHERE email=?""",
                          (student_name, student_skills, 1 if discoverable else 0, now, student_email))
            else:
                c.execute("""INSERT INTO students (name, email, skills, discoverable, updated_at)
                             VALUES (?, ?, ?, ?, ?)""",
                          (student_name, student_email, student_skills, 1 if discoverable else 0, now))
            conn.commit()
            conn.close()

            # AI recommendations from JobData
            if jobdata_df.empty:
                st.error("JobData.csv not found or empty.")
            else:
                with st.spinner("Thinking with Gemini…"):
                    recs = recommend_jobs_for_student(student_skills, jobdata_df, topn=5)

                if not recs:
                    st.info("No matching jobs found. Try adding more/different skills.")
                else:
                    st.success("Top Recommended Jobs")
                    for r in recs:
                        st.markdown(
                            f"**Job Role:** {r.get('job_role','')}  \n"
                            f"**Domain:** {r.get('domain','')}  \n"
                            f"**Skills Required:** {r.get('skills','')}  \n"
                            f"**Fit Score:** `{float(r.get('fit_score',0)):.2f}`  \n"
                            f"_Why:_ {r.get('reason','')}\n\n---"
                        )

    st.divider()
    st.subheader("Browse All Jobs (Recruiter Posted)")
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT * FROM jobs")
    jobs = c.fetchall()
    conn.close()
    if not jobs:
        st.info("No jobs posted by recruiters yet.")
    else:
        for job in jobs:
            st.markdown(f"""
**Title:** {job['title']}  
**Location:** {job['location']}  
**Salary:** {job['salary']}  
**Skills:** {job['skills']}  
**Eligibility:** {job['eligibility']}  
**Posted by:** {job['recruiter_email']}  
—""")
