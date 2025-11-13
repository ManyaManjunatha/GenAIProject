# app.py
import streamlit as st
import re
import time
from xgboost_predict import predict_xgb
import wikipedia
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------
# Streamlit Page Config
# ---------------------------
st.set_page_config(
    page_title="AI News Misinformation Detector",
    page_icon="🧠",
    layout="centered"
)

# ---------------------------
# Custom CSS for UI Polish
# ---------------------------
st.markdown("""
    <style>
    .title { font-size: 32px; font-weight: 700; text-align:center; margin-bottom:5px; }
    .subtitle { color: #6c757d; text-align:center; margin-bottom: 25px; }
    .result-card { border-radius: 15px; padding: 18px; box-shadow: 0 6px 18px rgba(0,0,0,0.08); margin-top:15px; }
    </style>
""", unsafe_allow_html=True)

# ---------------------------
# Helper: Wikipedia Verification
# ---------------------------
def wiki_fact_check(text):
    try:
        search_results = wikipedia.search(text)
        if not search_results:
            return {"verdict": "Not Found", "similarity": 0.0, "summary": ""}

        page_title = search_results[0]
        summary = wikipedia.summary(page_title, sentences=3)

        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform([text, summary])
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

        if similarity > 0.65:
            verdict = "Likely True"
        elif similarity > 0.35:
            verdict = "Partially Supported"
        else:
            verdict = "Likely False"

        return {"verdict": verdict, "similarity": similarity, "summary": summary, "source": page_title}

    except Exception:
        return {"verdict": "Error", "similarity": 0.0, "summary": ""}

# ---------------------------
# App Header
# ---------------------------
st.markdown('<div class="title">🧠 AI News Misinformation Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Detect and verify the authenticity of news articles using ML + Wikipedia verification</div>', unsafe_allow_html=True)
st.markdown("---")

# ---------------------------
# Input Section
# ---------------------------
st.subheader("📰 Enter a News Article")
title = st.text_input("News Title")
text = st.text_area("Article Text", height=180, placeholder="Paste or type a short news article...")

analyze_btn = st.button("🔍 Analyze")

# ---------------------------
# Prediction Logic
# ---------------------------
if analyze_btn:
    if not text.strip() and not title.strip():
        st.warning("Please enter a title or article text.")
    else:
        with st.spinner("Analyzing..."):
            time.sleep(0.3)
            content = title + " " + text

            # Model Prediction
            label, conf = predict_xgb(content)
            conf_percent = round(conf * 100, 2)

            # Wikipedia Fact Check
            wiki_data = wiki_fact_check(title or text[:80])

            # Adjust confidence based on Wiki verdict
            final_conf = conf
            if wiki_data['verdict'] == "Likely True":
                final_conf = min(conf + 0.1, 1.0)
            elif wiki_data['verdict'] == "Likely False":
                final_conf = max(conf - 0.1, 0.0)
            final_conf_percent = round(final_conf * 100, 2)

        # ---------------------------
        # Result Card
        # ---------------------------
        st.markdown('<div class="result-card">', unsafe_allow_html=True)

        if label == "Misinformation":
            st.markdown("### ❌ Misinformation Detected")
        else:
            st.markdown("### ✅ Likely True Information")

        st.metric(label="Confidence", value=f"{final_conf_percent}%")
        st.progress(int(final_conf_percent))

        st.write("---")

        st.markdown("**🧩 Wikipedia Verification**")
        if wiki_data["verdict"] == "Not Found":
            st.warning("No relevant Wikipedia article found.")
        elif wiki_data["verdict"] == "Error":
            st.error("Wikipedia check failed.")
        else:
            st.info(f"**Verdict:** {wiki_data['verdict']} (Similarity: {wiki_data['similarity']:.2f})")
            st.caption(f"📖 Source: {wiki_data['source']}")
            st.write(f"📝 {wiki_data['summary'][:400]}...")

        st.markdown("</div>", unsafe_allow_html=True)

        st.success("Analysis complete ✅")

# ---------------------------
# Sidebar Info
# ---------------------------
st.sidebar.header("ℹ️ About This App")
st.sidebar.write("""
**Purpose:**  
Detect misinformation in digital news articles using a machine learning model trained on the **WELFake dataset**, enhanced with **Wikipedia-based factual verification**.

**Pipeline:**  
1. TF-IDF feature extraction  
2. XGBoost classification  
3. Wikipedia semantic similarity  
4. Final confidence scoring
""")

st.sidebar.write("---")
st.sidebar.write("Developed by your team 💻")
