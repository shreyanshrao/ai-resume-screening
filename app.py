import streamlit as st
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# ---------------------------
# Extract text from PDF
# ---------------------------
def extract_text_from_pdf(file):
    pdf = PdfReader(file)
    text = ""
    for page in pdf.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text.lower().strip() if text else ""

# ---------------------------
# Keyword extraction
# ---------------------------
def extract_keywords(text):
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    return set(words)

# ---------------------------
# Rank resumes (0–100 score)
# ---------------------------
def rank_resumes(job_description, resumes):
    documents = [job_description] + resumes
    vectorizer = TfidfVectorizer().fit_transform(documents)
    vectors = vectorizer.toarray()

    job_vector = vectors[0]
    resume_vectors = vectors[1:]

    similarities = cosine_similarity([job_vector], resume_vectors).flatten()
    
    return similarities * 100  # convert to %

# ---------------------------
# Analyze candidate
# ---------------------------
def analyze_candidate(job_desc, resume_text, score):
    jd_keywords = extract_keywords(job_desc)
    resume_keywords = extract_keywords(resume_text)

    matched = list(jd_keywords.intersection(resume_keywords))
    missing = list(jd_keywords.difference(resume_keywords))

    strengths = matched[:3] if matched else ["No strong keyword match"]
    gaps = missing[:3] if missing else ["No major gaps detected"]

    # Recommendation logic
    if score >= 75:
        recommendation = "Strong Fit"
    elif score >= 50:
        recommendation = "Moderate Fit"
    else:
        recommendation = "Not Fit"

    return strengths, gaps, recommendation

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("🚀 AI Resume Screening & Ranking System")

job_description = st.text_area("📌 Enter the Job Description")
uploaded_files = st.file_uploader("📄 Upload PDF resumes", type=["pdf"], accept_multiple_files=True)

if uploaded_files and job_description:
    resumes = [extract_text_from_pdf(file) for file in uploaded_files]
    scores = rank_resumes(job_description, resumes)

    # Combine all results
    results = []
    for i in range(len(resumes)):
        strengths, gaps, recommendation = analyze_candidate(
            job_description, resumes[i], scores[i]
        )

        results.append({
            "file": uploaded_files[i],
            "score": scores[i],
            "strengths": strengths,
            "gaps": gaps,
            "recommendation": recommendation
        })

    # Sort by score
    ranked_resumes = sorted(results, key=lambda x: x["score"], reverse=True)

    st.subheader("🏆 Ranked Candidates")

    for i, data in enumerate(ranked_resumes, start=1):
        file = data["file"]
        score = data["score"]

        st.markdown(f"### 🥇 Rank {i}: {file.name}")
        st.write(f"**Match Score:** {score:.2f}/100")

        st.progress(int(score))

        st.write(f"**Key Strengths:** {', '.join(data['strengths'])}")
        st.write(f"**Key Gaps:** {', '.join(data['gaps'])}")
        st.write(f"**Final Recommendation:** {data['recommendation']}")

        st.divider()