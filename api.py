"""
============================================================================
  CINEIQ — Explainable Movie Recommendation Engine (FastAPI Backend)
============================================================================
  This module provides the REST API layer for CineIQ. It exposes two
  endpoints that return mock recommendation data derived from the MovieLens
  dataset, each enriched with explainability metadata so the frontend can
  show *why* a movie was recommended.

  Endpoints:
      GET  /recommend   → Top-N personalised movie recommendations
      GET  /similar     → Movies similar to a given title

  Run locally:
      uvicorn api:app --reload --port 8000
============================================================================
"""

import hashlib
import random
import pandas as pd
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os

# ---------------------------------------------------------------------------
#  Pydantic schemas
# ---------------------------------------------------------------------------

class MovieRecommendation(BaseModel):
    """Schema for a single movie recommendation returned by the API."""
    title: str
    final_score: float          # Weighted composite score  (0 – 100)
    content_score: float        # Content-based filtering   (0 – 100)
    collaborative_score: float  # Collaborative filtering   (0 – 100)
    sentiment_score: float      # Sentiment / review-based  (0 – 100)
    explainability_reason: str  # Human-readable explanation


class RecommendResponse(BaseModel):
    """Wrapper for the /recommend endpoint response."""
    recommendations: list[MovieRecommendation]


class SimilarResponse(BaseModel):
    """Wrapper for the /similar endpoint response."""
    query_title: str
    similar_movies: list[MovieRecommendation]


# ---------------------------------------------------------------------------
#  FastAPI application instance
# ---------------------------------------------------------------------------

app = FastAPI(
    title="CineIQ API",
    description="Explainable hybrid movie recommendation engine — mock API layer.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
#  Dataset Loading & Candidate Pool Generation
# ---------------------------------------------------------------------------

# Global cache for our candidate pool
CANDIDATE_POOL = []

@app.on_event("startup")
def load_dataset():
    """
    Load the MovieLens dataset into memory on startup.
    We take a sample of 1,000 movies to act as our candidate pool.
    """
    global CANDIDATE_POOL
    dataset_path = os.path.join("dataset", "movies.csv")
    
    if not os.path.exists(dataset_path):
        print(f"WARNING: Dataset not found at {dataset_path}. Using fallback mock data.")
        # Minimal fallback if the file isn't there for some reason
        CANDIDATE_POOL = [
            {"title": "Fallback Movie 1", "genres": "Action", "content_score": 80, "collaborative_score": 75, "sentiment_score": 90},
            {"title": "Fallback Movie 2", "genres": "Drama", "content_score": 60, "collaborative_score": 85, "sentiment_score": 70},
        ]
        return

    # Load dataset
    df = pd.read_csv(dataset_path)
    
    # We sample 1,000 movies. We use a fixed random_state to ensure
    # the same candidates are selected every time the server restarts.
    sample_size = min(1000, len(df))
    df_sample = df.sample(n=sample_size, random_state=42)
    
    candidates = []
    for _, row in df_sample.iterrows():
        movie_id = str(row['movieId'])
        title = str(row['title'])
        genres = str(row['genres'])
        
        # Deterministically generate base scores (0-100) using a hash of the movieId
        # This ensures the scores stay stable across API requests and server restarts.
        h = hashlib.md5(movie_id.encode('utf-8')).hexdigest()
        
        # Convert chunks of the hex hash to integer scores
        c_score = int(h[0:4], 16) % 100
        col_score = int(h[4:8], 16) % 100
        s_score = int(h[8:12], 16) % 100
        
        candidates.append({
            "title": title,
            "genres": genres,
            "content_score": float(c_score),
            "collaborative_score": float(col_score),
            "sentiment_score": float(s_score)
        })
        
    CANDIDATE_POOL = candidates
    print(f"Loaded {len(CANDIDATE_POOL)} candidate movies from dataset.")


def generate_explainability(genres: str, w_content: float, w_collab: float, w_sentiment: float) -> str:
    """Generate a dynamic reason based on the movie's genres and highest weight."""
    genre_list = genres.replace('|', ', ') if genres != "(no genres listed)" else "Unknown"
    
    if w_content >= w_collab and w_content >= w_sentiment:
        return f"Highly recommended based on your preference for {genre_list}."
    elif w_collab >= w_content and w_collab >= w_sentiment:
        return f"Trending! Other users who enjoy {genre_list} films rated this highly."
    else:
        return f"Critics and audiences are praising this {genre_list} film."

# ---------------------------------------------------------------------------
#  API endpoints
# ---------------------------------------------------------------------------

@app.get("/recommend", response_model=RecommendResponse)
async def get_recommendations(
    w_content: float = Query(0.33, description="Weight for content score (0-1)"),
    w_collab: float = Query(0.33, description="Weight for collaborative score (0-1)"),
    w_sentiment: float = Query(0.33, description="Weight for sentiment score (0-1)"),
    action_affinity: float = Query(82, description="Taste profile affinity for Action (0-100)"),
    scifi_affinity: float = Query(91, description="Taste profile affinity for Sci-Fi (0-100)"),
    romance_affinity: float = Query(58, description="Taste profile affinity for Romance (0-100)"),
    comedy_affinity: float = Query(65, description="Taste profile affinity for Comedy (0-100)"),
    horror_affinity: float = Query(40, description="Taste profile affinity for Horror (0-100)"),
):
    """
    Return the top-N personalised movie recommendations from the candidate pool,
    dynamically scored based on the provided weights and genre taste profile.
    """
    scored_candidates = []
    
    for c in CANDIDATE_POOL:
        # Boost/penalize content score based on taste profile affinities
        base_content = c['content_score']
        genre_bonus = 0.0
        
        movie_genres = c['genres'].split('|')
        
        # Calculate bonus: If affinity > 50, it's a positive boost. If < 50, it's a penalty.
        # We scale the modifier so 100 adds up to +20, and 0 adds up to -20 per matched genre.
        if "Action" in movie_genres: genre_bonus += (action_affinity - 50) * 0.4
        if "Sci-Fi" in movie_genres: genre_bonus += (scifi_affinity - 50) * 0.4
        if "Romance" in movie_genres: genre_bonus += (romance_affinity - 50) * 0.4
        if "Comedy" in movie_genres: genre_bonus += (comedy_affinity - 50) * 0.4
        if "Horror" in movie_genres: genre_bonus += (horror_affinity - 50) * 0.4
        
        # Ensure the content score stays bounded between 0 and 100
        adjusted_content = min(100.0, max(0.0, base_content + genre_bonus))

        # Calculate final score
        final = (adjusted_content * w_content) + \
                (c['collaborative_score'] * w_collab) + \
                (c['sentiment_score'] * w_sentiment)
                
        # Generate a reason
        reason = generate_explainability(c['genres'], w_content, w_collab, w_sentiment)
        
        scored_candidates.append(
            MovieRecommendation(
                title=c['title'],
                final_score=round(final, 2),
                content_score=round(adjusted_content, 1),
                collaborative_score=c['collaborative_score'],
                sentiment_score=c['sentiment_score'],
                explainability_reason=reason
            )
        )
        
    # Sort by final score descending
    scored_candidates.sort(key=lambda x: x.final_score, reverse=True)
    
    # Return Top 10
    return RecommendResponse(recommendations=scored_candidates[:10])


@app.get("/similar", response_model=SimilarResponse)
async def get_similar_movies(
    title: str = Query(
        default="Toy Story (1995)",
        description="Title of the movie to find similar entries for.",
    ),
):
    """
    Return movies similar to the supplied *title*.
    Mock implementation: returns 3 random highly-rated movies from the pool.
    """
    if not CANDIDATE_POOL:
        return SimilarResponse(query_title=title, similar_movies=[])
        
    # Just grab 3 pseudo-random movies for the similar endpoint
    sample = random.sample(CANDIDATE_POOL, min(3, len(CANDIDATE_POOL)))
    
    similars = []
    for c in sample:
        similars.append(
            MovieRecommendation(
                title=c['title'],
                final_score=85.0, # Fixed mock score
                content_score=c['content_score'],
                collaborative_score=c['collaborative_score'],
                sentiment_score=c['sentiment_score'],
                explainability_reason=f"Shares stylistic elements and genres ({c['genres'].replace('|', ', ')}) with the queried title."
            )
        )
        
    return SimilarResponse(query_title=title, similar_movies=similars)


# ---------------------------------------------------------------------------
#  Entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
