import streamlit as st
from database import init_db
from recruiter_ui import recruiter_tab
from student_ui import student_tab

st.set_page_config(page_title="Enter your title", layout="wide")
st.title("Enter your title")

tabs = st.tabs(["Recruiter", "Student"])

with tabs[0]:
    recruiter_tab()

with tabs[1]:
    student_tab()

if __name__ == "__main__":
    init_db()

