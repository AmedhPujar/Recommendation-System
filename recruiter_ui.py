import streamlit as st
import datetime
from database import get_db
from auth import hash_password, verify_password, create_jwt, decode_jwt
from recommender import recommend_students_for_job

def recruiter_tab():
    st.header("Recruiter Portal")
    menu = st.radio("Select Option", ["Signup", "Login", "Post Job", "My Jobs"], horizontal=True)

    # --- Signup ---
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

    # --- Login ---
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

    # --- Post Job ---
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

    # --- My Jobs ---
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
