from langchain.prompts import ChatPromptTemplate

student_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are an AI career advisor. Given a student's skills and candidate jobs, "
     "rank the best matches. Return STRICT JSON: "
     "[{job_role, domain, skills, reason, fit_score}]."),
    ("human",
     "Student skills: {skills}\n\nCandidate jobs: {jobs}\n\n"
     "Top {topn} results, JSON only.")
])

recruiter_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are an AI recruiter assistant. Given one job and student profiles, "
     "rank top candidates. Return STRICT JSON: "
     "[{name, email, reason, fit_score}]."),
    ("human",
     "Job: {job}\n\nStudents: {students}\n\nTop {topn} results, JSON only.")
])
