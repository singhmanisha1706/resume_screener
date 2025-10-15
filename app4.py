import streamlit as st
import pandas as pd
import tempfile
import os
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from pdfminer.high_level import extract_text as extract_pdf_text
from docx import Document

# Download NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except:
    pass

# Page configuration
st.set_page_config(
    page_title="HR Resume Parser - Candidate Ranking System",
    page_icon="👔",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1e3a8a;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #3b82f6;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    .candidate-card {
        border: 1px solid #e5e7eb;
        border-radius: 0.5rem;
        padding: 1.5rem;
        margin: 1rem 0;
        background-color: #f9fafb;
    }
    .match-score {
        font-size: 1.2rem;
        font-weight: bold;
        color: #059669;
    }
    .skill-pill {
        background-color: #3b82f6;
        color: white;
        padding: 0.3rem 0.7rem;
        border-radius: 1rem;
        margin: 0.2rem;
        display: inline-block;
        font-size: 0.8rem;
    }
    .contact-info {
        background-color: #e0f2fe;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class HRResumeProcessor:
    def __init__(self):
        self.skills_db = {
            'python', 'java', 'javascript', 'sql', 'html', 'css', 'react', 'angular', 
            'vue', 'node.js', 'express', 'django', 'flask', 'aws', 'azure', 'docker', 
            'kubernetes', 'machine learning', 'deep learning', 'tensorflow', 'pytorch', 
            'data analysis', 'tableau', 'power bi', 'excel', 'project management', 
            'agile', 'scrum', 'risk management', 'insurance', 'consulting', 'finance',
            'accounting', 'hr', 'recruitment', 'compensation', 'benefits', 'analytics',
            'communication', 'leadership', 'teamwork', 'problem solving', 'critical thinking'
        }
    
    def extract_text_from_file(self, file):
        """Extract text from PDF, DOCX, or TXT files"""
        text = ""
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.name)[1]) as tmp_file:
                tmp_file.write(file.getvalue())
                tmp_path = tmp_file.name
            
            # Extract text based on file type
            if file.name.lower().endswith('.pdf'):
                text = extract_pdf_text(tmp_path)
            elif file.name.lower().endswith('.docx'):
                doc = Document(tmp_path)
                text = '\n'.join([para.text for para in doc.paragraphs])
            else:
                # For text files
                with open(tmp_path, 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.read()
            
            # Clean up temporary file
            os.unlink(tmp_path)
            
        except Exception as e:
            st.error(f"Error reading file {file.name}: {str(e)}")
        
        return text
    
    def extract_skills(self, text):
        """Extract skills from text using skills database"""
        text_lower = text.lower()
        found_skills = []
        
        for skill in self.skills_db:
            if skill.lower() in text_lower:
                found_skills.append(skill)
        
        return found_skills
    
    def extract_emails(self, text):
        """Extract email addresses from text"""
        email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
        return re.findall(email_pattern, text)
    
    def extract_phones(self, text):
        """Extract phone numbers from text"""
        phone_pattern = r'(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'
        return re.findall(phone_pattern, text)
    
    def extract_name(self, text):
        """Basic name extraction from text"""
        lines = text.split('\n')
        for line in lines[:10]:  # Check first 10 lines
            line = line.strip()
            if re.match(r'^[A-Z][a-z]+\s+[A-Z][a-z]+$', line):
                return line
        return "Unknown Candidate"
    
    def parse_resume(self, file):
        """Main function to parse resume"""
        try:
            text = self.extract_text_from_file(file)
            if not text:
                return None
            
            return {
                'name': self.extract_name(text),
                'skills': self.extract_skills(text),
                'emails': self.extract_emails(text),
                'phones': self.extract_phones(text),
                'raw_text': text,
                'filename': file.name
            }
        except Exception as e:
            st.error(f"Error parsing resume {file.name}: {str(e)}")
            return None

class CandidateMatcher:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    
    def calculate_similarity(self, jd_text, resume_texts):
        """Calculate similarity between JD and resumes using TF-IDF"""
        if not jd_text or not resume_texts:
            return []
        
        all_texts = [jd_text] + resume_texts
        try:
            tfidf_matrix = self.vectorizer.fit_transform(all_texts)
            cosine_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
            return cosine_similarities
        except:
            return [0.0] * len(resume_texts)
    
    def calculate_skill_match(self, jd_text, resume_skills):
        """Calculate skill-based matching percentage"""
        if not jd_text or not resume_skills:
            return 0.0
        
        jd_lower = jd_text.lower()
        jd_skills = set()
        
        # Extract skills from JD
        for skill in resume_processor.skills_db:
            if skill.lower() in jd_lower:
                jd_skills.add(skill.lower())
        
        if not jd_skills:
            return 0.0
        
        resume_skills_lower = set(skill.lower() for skill in resume_skills)
        common_skills = jd_skills.intersection(resume_skills_lower)
        
        return len(common_skills) / len(jd_skills)
    
    def rank_candidates(self, jd_text, resumes_data):
        """Rank candidates based on multiple factors"""
        ranked_candidates = []
        
        # Extract resume texts for TF-IDF comparison
        resume_texts = [resume.get('raw_text', '') for resume in resumes_data]
        
        # Calculate text similarity
        text_similarities = self.calculate_similarity(jd_text, resume_texts)
        
        for i, resume in enumerate(resumes_data):
            # Calculate skill matching
            skill_match = self.calculate_skill_match(jd_text, resume.get('skills', []))
            
            # Combined score (weighted average)
            final_score = (
                text_similarities[i] * 0.6 +  # Text similarity weight
                skill_match * 0.4            # Skill match weight
            )
            
            ranked_candidates.append({
                'resume_data': resume,
                'similarity_score': final_score,
                'text_similarity': text_similarities[i],
                'skill_match': skill_match,
                'rank': 0
            })
        
        # Sort by final score (descending)
        ranked_candidates.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        # Assign ranks
        for i, candidate in enumerate(ranked_candidates):
            candidate['rank'] = i + 1
        
        return ranked_candidates

# Initialize processors
resume_processor = HRResumeProcessor()
candidate_matcher = CandidateMatcher()

def upload_section():
    st.markdown('<div class="sub-header">📁 Upload Resumes & Job Description</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("1. Job Description")
        jd_text = st.text_area(
            "Paste the job description here:",
            height=200,
            placeholder="Enter the complete job description including required skills, qualifications, and experience...",
            value=st.session_state.get('jd_text', '')
        )
        
        if jd_text:
            st.session_state.jd_text = jd_text
            st.success("Job description saved!")
    
    with col2:
        st.subheader("2. Upload Candidate Resumes")
        uploaded_files = st.file_uploader(
            "Select resume files (PDF, DOCX, or TXT):",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True,
            help="Upload multiple resumes for batch processing"
        )
        
        if uploaded_files:
            if st.button("🚀 Process Resumes", type="primary"):
                process_resumes(uploaded_files)

def process_resumes(uploaded_files):
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    processed_resumes = []
    for i, file in enumerate(uploaded_files):
        status_text.text(f"Processing {file.name}... ({i+1}/{len(uploaded_files)})")
        resume_data = resume_processor.parse_resume(file)
        if resume_data:
            resume_data['filename'] = file.name
            processed_resumes.append(resume_data)
        progress_bar.progress((i + 1) / len(uploaded_files))
    
    st.session_state.resumes_data = processed_resumes
    status_text.text(f"✅ Processed {len(processed_resumes)} resumes successfully!")
    progress_bar.empty()

def ranking_section():
    if not st.session_state.get('resumes_data') or not st.session_state.get('jd_text'):
        st.warning("Please upload resumes and a job description first!")
        return
    
    st.markdown('<div class="sub-header">📊 Candidate Ranking Results</div>', unsafe_allow_html=True)
    
    # Rank candidates
    ranked_candidates = candidate_matcher.rank_candidates(
        st.session_state.jd_text, st.session_state.resumes_data
    )
    
    # Display summary
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Candidates", len(ranked_candidates))
    with col2:
        avg_score = sum(c['similarity_score'] for c in ranked_candidates) / len(ranked_candidates)
        st.metric("Average Match Score", f"{avg_score:.1%}")
    with col3:
        top_score = ranked_candidates[0]['similarity_score'] if ranked_candidates else 0
        st.metric("Top Match Score", f"{top_score:.1%}")
    
    # Display ranked candidates
    for candidate in ranked_candidates:
        display_candidate_card(candidate)

def display_candidate_card(candidate):
    data = candidate['resume_data']
    
    with st.expander(
        f"#{candidate['rank']}: {data.get('name', 'Unknown Candidate')} - {candidate['similarity_score']:.1%} Match",
        expanded=candidate['rank'] <= 3
    ):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**👤 Candidate Information**")
            if data.get('name'):
                st.write(f"**Name:** {data['name']}")
            if data.get('emails'):
                st.write(f"**Email:** {data['emails'][0]}")
            if data.get('phones'):
                st.write(f"**Phone:** {data['phones'][0]}")
            st.write(f"**File:** {data.get('filename', 'N/A')}")
            
            st.markdown("**📊 Match Analysis**")
            st.write(f"**Overall Score:** {candidate['similarity_score']:.1%}")
            st.progress(float(candidate['similarity_score']))
            st.write(f"Text Similarity: {candidate['text_similarity']:.1%}")
            st.write(f"Skill Match: {candidate['skill_match']:.1%}")
        
        with col2:
            st.markdown("**🎯 Skills**")
            if data.get('skills'):
                skills_html = "".join([f'<span class="skill-pill">{skill}</span>' for skill in data['skills'][:15]])
                st.markdown(skills_html, unsafe_allow_html=True)
            else:
                st.write("No skills detected")

def export_section():
    if not st.session_state.get('resumes_data'):
        st.warning("No resumes to export. Please process resumes first.")
        return
    
    st.markdown('<div class="sub-header">💾 Export Results</div>', unsafe_allow_html=True)
    
    # Prepare data for export
    export_data = []
    for resume in st.session_state.resumes_data:
        export_data.append({
            "Name": resume.get('name', 'Unknown'),
            "Email": resume.get('emails', ['N/A'])[0],
            # "Phone": resume.get('phones', ['N/A'])[0],
            "Skills": ", ".join(resume.get('skills', [])),
            "Filename": resume.get('filename', '')
        })
    
    df = pd.DataFrame(export_data)
    
    # Display preview
    st.subheader("Candidate Data Preview")
    st.dataframe(df.head())
    
    # Export options
    col1, col2 = st.columns(2)
    
    with col1:
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download as CSV",
            data=csv,
            file_name="candidate_data.csv",
            mime="text/csv"
        )
    
    with col2:
        st.download_button(
            label="📥 Download All Resumes",
            data="\n".join([f"{r.get('name', 'Unknown')}: {r.get('filename', '')}" for r in st.session_state.resumes_data]),
            file_name="resume_list.txt",
            mime="text/plain"
        )

def main():
    st.markdown('<div class="main-header">HR Resume Parser & Candidate Ranking System</div>', unsafe_allow_html=True)
    
    # Initialize session state
    if 'resumes_data' not in st.session_state:
        st.session_state.resumes_data = []
    if 'jd_text' not in st.session_state:
        st.session_state.jd_text = ""
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    section = st.sidebar.radio("Go to", ["Upload Resumes", "Candidate Ranking", "Export Results"])
    
    if section == "Upload Resumes":
        upload_section()
    elif section == "Candidate Ranking":
        ranking_section()
    elif section == "Export Results":
        export_section()

if __name__ == "__main__":
    main()