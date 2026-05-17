# 🎬 CineIQ: Explainable Hybrid Movie Recommendation Engine

## 📌 Problem Statement
Content discovery on modern streaming platforms is opaque, biased toward promoted titles, and traps users in recommendation loops. There is a need for an open, explainable movie recommendation engine that combines multiple ML strategies to deliver personalized, interpretable suggestions that evolve with user taste over time.

## 🚀 Deliverables
- **Hybrid Recommendation Engine:** Combines collaborative filtering, content-based filtering (TF-IDF + cosine similarity), and SVD-based matrix factorization via a weighted ensemble.
- **Sentiment-Aware Re-Ranker:** Uses VADER/DistilBERT on user reviews to re-rank recommendations based on real audience reception signals.
- **User Taste Dashboard:** Streamlit interface visualizing genre radar charts, decade preferences, and director/actor affinities from rating history.
- **Explainability Layer:** Every recommendation surfaces a human-readable reason using LIME or rule-based templates.

## 📊 Datasets
- **[MovieLens 25M](https://grouplens.org/datasets/movielens/25m/)** — User ratings and core metadata.
- **[TMDB Metadata (Kaggle)](https://www.kaggle.com/)** — Cast, genres, keywords for 45K movies.
- **[IMDB 50K Reviews (Kaggle)](https://www.kaggle.com/)** — Used for sentiment model training.

## 🛠️ Tech Stack
- **Machine Learning:** Python, scikit-learn, Surprise (SVD), Pandas, NumPy
- **NLP:** VADER / HuggingFace DistilBERT
- **Serving:** FastAPI (`/recommend` and `/similar` endpoints)
- **Dashboard:** Streamlit + Plotly
- **Tracking:** MLflow for experiment logging

## 🏃‍♂️ How to Run Locally
Just Run the servicelauncher.bat
**OR**

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Start the FastAPI Backend**
   ```bash
   python -m uvicorn api:app --reload --port 8000
   ```
   *The API documentation will be available at [http://localhost:8000/docs](http://localhost:8000/docs).*

3. **Start the Streamlit Dashboard**
   Open a new terminal window and run:
   ```bash
   python -m streamlit run app.py --server.port 8501
   ```
   *The dashboard will be available at [http://localhost:8501](http://localhost:8501).*
