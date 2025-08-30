import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_google_genai import ChatGoogleGenerativeAI
from .prompts import student_prompt, recruiter_prompt
from .config import GOOGLE_API_KEY
from .utils import extract_json

def get_llm():
    if not GOOGLE_API_KEY:
        return None
    return ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3, max_output_tokens=1024)

def tfidf_retrieve_top_k(student_skills, corpus, k=15):
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
    return out.sort_values('Score', ascending=False).head(k)

# recommend_jobs_for_student() and recommend_students_for_job() stay here
