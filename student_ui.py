import streamlit as st
import datetime
import pandas as pd
from database import get_db
from recommender import recommend_jobs_for_student
from config import JOBDATA_PATH

@st.cache_data(show_spinner=False)
def load_jobdata():
    try:
        df = pd.read_csv(JOBDATA_PATH)
        if 'Skills' not in df.columns:
            df['Skills'] = ''
        for col in df.columns:
            df[col] = df[col].fillna('')
        return df
    except Exception as e:
        st.sidebar.warning(f"JobData.csv failed to load: {e}")
        return pd.DataFrame(columns=['Domain','Job Role','Skills','Personality'])

jobdata_df = load_jobdata()

def student_tab():
    st.header("Student Portal")
    st.write("Get **AI-powered job recommendations** based on your skills (Gemini Flash + TF-IDF retrieval).")

    # --- Profile ---
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
            # Save profile
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

            # AI recommendations
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

    # --- Browse Jobs ---
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
