import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import fitz  # PyMuPDF for PDF resumes

# ----------------------------
# 1. Load Job Dataset
# ----------------------------
@st.cache_data
def load_jobs():
    jobs = pd.read_csv("all_job_post.csv")  # Replace with your Kaggle dataset
    jobs = jobs[['job_id','category','job_title','job_description','job_skill_set']].dropna()
    return jobs

# ----------------------------
# 2. Extract text from resume PDF
# ----------------------------
def extract_text_from_pdf(uploaded_file):
    text = ""
    with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text

# ----------------------------
# 3. Recommend jobs based on skills and description
# ----------------------------
def recommend_jobs(resume_text, jobs, top_n=5):
    # Combine job description + skill set for better matching
    jobs['combined'] = jobs['job_description'].astype(str) + " " + jobs['job_skill_set'].astype(str)

    # TF-IDF
    vectorizer = TfidfVectorizer(stop_words="english")
    job_tfidf = vectorizer.fit_transform(jobs['combined'])
    resume_tfidf = vectorizer.transform([resume_text])

    # Similarity
    similarity_scores = cosine_similarity(resume_tfidf, job_tfidf)
    top_indices = similarity_scores[0].argsort()[-top_n:][::-1]
    return jobs.iloc[top_indices]

def main():
    st.title("💼 Job Recommendation System")
    st.write("Upload your resume and get job recommendations based on your skills!")

    # Load job data
    jobs = load_jobs()

    # Resume upload
    uploaded_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
    
    if uploaded_file:
        resume_text = extract_text_from_pdf(uploaded_file)

        st.subheader("📄 Extracted Resume Text")
        st.write(resume_text[:1000] + " ...")  # show preview

        # Recommend jobs
        recommended = recommend_jobs(resume_text, jobs)

        st.subheader("Recommended Jobs for You")
        st.table(recommended[['job_id','job_title','category']])

        # Show detailed job descriptions
        for idx, row in recommended.iterrows():
            with st.expander(f"{row['job_title']} ({row['category']})"):
                st.write("Job ID:", row['job_id'])
                st.write("Description:", row['job_description'])
                st.write("Required Skills:", row['job_skill_set'])

if __name__ == "__main__":
    main()

