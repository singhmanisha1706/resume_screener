# resume_screener
Smart Resume Screener is an AI-powered tool that automatically parses resumes, extracts key information like skills, education, and experience, and compares them with a given job description. It generates a match score and provides a justification for each candidate using both TF-IDF similarity and LLM-based semantic scoring.
System overview
<img width="969" height="495" alt="Screenshot 2025-10-15 131223" src="https://github.com/user-attachments/assets/ae32333f-d1d0-4399-85bb-253e5db8db90" />
Upload / Ingest
Frontend (Streamlit) accepts resume files (PDF/DOCX/TXT) and a Job Description (text or uploaded JD).

Files pushed to backend storage (local or S3).

Parsing & Extraction (Stage A — deterministic)

parser microservice:

PDF/DOCX → plain text (pdfminer, PyMuPDF, python-docx).

Field extractors: name, emails, phones, education, company/roles, dates, raw experience text.

Skills extraction: rule-based + dictionary lookup (skills_db.json).

Preprocess: lowercasing, punctuation removal, lemmatization for skill normalization.

Indexing & Fast Matching (Stage B — vector + TF-IDF)

Build TF-IDF vectorizer and/or embedding index (OpenAI/other embeddings + FAISS).

Precompute resume vectors and JD vector.

Compute fast similarity scores (cosine) and skill overlap ratios.

Use this to shortlist top-K candidates (K = 5–20) for LLM calls.

LLM Scoring & Explanation (Stage C — LLM)

For top-K only, call LLM with:

parsed fields + raw text + JD + short TF-IDF / skill match stats

Ask for JSON output: match_score (1–10), matched/missing skills, justification bullets, confidence.

Save LLM output and merge with TF-IDF signals to produce final score.

Storage & API

Database (Postgres/SQLite): parsed resume records, LLM responses, scores, file links.

API endpoints:

/parse_resume (upload)

/rank_candidates (jd + resume ids or files)

/recommend_jobs (resume id -> jobs)

/candidate/{id} (view detail)

Caching layer for LLM responses (cache by hash of JD+resume text).
<img width="986" height="638" alt="Screenshot 2025-10-15 131320" src="https://github.com/user-attachments/assets/5f3eb9c7-1517-45af-afec-62ce205a244e" />

Frontend

Streamlit dashboard: upload, view parsed fields, ranked list, expand for LLM justification, filter by skills, export shortlist.
