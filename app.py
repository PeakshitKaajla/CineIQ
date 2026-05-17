"""
============================================================================
  CINEIQ — Explainable Movie Recommendation Engine (Streamlit Frontend)
============================================================================
  Interactive dashboard that lets the user tune three recommendation-engine
  weights (Content-Based, Collaborative, Sentiment) via linked sliders that
  always sum to 100 %.  Movies are scored in real-time, sorted from highest
  to lowest, and displayed with progress-bar gauges and explainability text.

  Sections:
      Sidebar          → Weight sliders (auto-normalised to 100 %)
      Taste Profile     → Plotly radar chart of mock genre affinities
      Recommendations   → Dynamically ranked movie cards with gauges

  Run locally:
      streamlit run app.py
============================================================================
"""

import streamlit as st
import plotly.express as px
import pandas as pd
import requests

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  PAGE CONFIG — must be the first Streamlit command                      ║
# ╚══════════════════════════════════════════════════════════════════════════╝

st.set_page_config(
    page_title="CineIQ — Movie Recommendation Engine",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  CUSTOM CSS — dark cinematic theme overrides                            ║
# ╚══════════════════════════════════════════════════════════════════════════╝

st.markdown(
    """
    <style>
    /* ---------- global body ---------- */
    .stApp {
        background: linear-gradient(135deg, #0d0d0d 0%, #1a1a2e 50%, #16213e 100%);
        color: #e0e0e0;
    }

    /* ---------- sidebar ---------- */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0f23 0%, #1a1a2e 100%);
        border-right: 1px solid rgba(255, 255, 255, 0.06);
    }

    /* ---------- headings ---------- */
    h1, h2, h3 {
        background: linear-gradient(90deg, #e94560, #c23bf0, #5b86e5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }

    /* ---------- movie card ---------- */
    .movie-card {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 1.4rem 1.6rem;
        margin-bottom: 1rem;
        backdrop-filter: blur(6px);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .movie-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 30px rgba(233, 69, 96, 0.15);
    }

    .movie-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 0.3rem;
    }
    .movie-score {
        font-size: 2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #e94560, #5b86e5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .score-breakdown {
        font-size: 0.82rem;
        color: #8892b0;
        margin-top: 0.25rem;
    }
    .movie-rank {
        font-size: 1.6rem;
        font-weight: 900;
        color: rgba(233, 69, 96, 0.35);
    }

    /* ---------- progress bar override ---------- */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #e94560, #c23bf0, #5b86e5);
        border-radius: 8px;
    }
    .stProgress > div > div > div {
        background: rgba(255, 255, 255, 0.06);
        border-radius: 8px;
    }

    /* ---------- info box (explainability) ---------- */
    .stAlert {
        background: rgba(91, 134, 229, 0.08);
        border-left-color: #5b86e5;
        border-radius: 8px;
    }

    /* ---------- metric cards ---------- */
    [data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 12px;
        padding: 1rem;
    }

    /* ---------- weight badge (sidebar) ---------- */
    .weight-badge {
        display: inline-block;
        background: linear-gradient(135deg, #e94560, #5b86e5);
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-top: 0.25rem;
    }

    /* ---------- divider ---------- */
    hr {
        border-color: rgba(255, 255, 255, 0.06);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  MOCK DATA                                                              ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# Movies are now dynamically fetched from the FastAPI backend.

# Genre affinities are now interactive UI state variables

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  HELPER FUNCTIONS                                                       ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def normalise_weights(w_content: float, w_collab: float, w_sentiment: float):
    """
    Normalise three raw slider values so they sum to exactly 1.0.

    This guarantees the weighted score formula always operates on a valid
    probability distribution regardless of individual slider positions.

    Returns:
        Tuple of (normalised_content, normalised_collab, normalised_sentiment)
    """
    total = w_content + w_collab + w_sentiment
    if total == 0:
        # Edge case: all sliders at zero → equal weight
        return (1 / 3, 1 / 3, 1 / 3)
    return (w_content / total, w_collab / total, w_sentiment / total)


def compute_final_score():
    # Deprecated: The backend now computes the final score natively.
    pass


def get_ranked_movies(
    w_content: float,
    w_collab: float,
    w_sentiment: float,
    genre_affinities: dict,
) -> list[dict]:
    """
    Fetch ranked movies dynamically from the FastAPI backend.
    """
    try:
        response = requests.get(
            "http://localhost:8000/recommend",
            params={
                "w_content": w_content,
                "w_collab": w_collab,
                "w_sentiment": w_sentiment,
                "action_affinity": genre_affinities.get("Action", 82),
                "scifi_affinity": genre_affinities.get("Sci-Fi", 91),
                "romance_affinity": genre_affinities.get("Romance", 58),
                "comedy_affinity": genre_affinities.get("Comedy", 65),
                "horror_affinity": genre_affinities.get("Horror", 40),
            },
            timeout=5
        )
        response.raise_for_status()
        return response.json().get("recommendations", [])
    except Exception as e:
        st.error(f"Failed to fetch recommendations from backend: {e}")
        return []


def build_radar_chart(genre_affinities: dict):
    """
    Build a Plotly radar (polar) chart showing interactive genre affinities.

    The chart uses a dark transparent background so it blends with the
    cinematic theme.
    """
    genres = list(genre_affinities.keys())
    scores = list(genre_affinities.values())

    # Close the polygon by repeating the first point
    genres_closed = genres + [genres[0]]
    scores_closed = scores + [scores[0]]

    df = pd.DataFrame({"Genre": genres_closed, "Affinity": scores_closed})

    fig = px.line_polar(
        df,
        r="Affinity",
        theta="Genre",
        line_close=True,
        range_r=[0, 100],
        template="plotly_dark",
    )
    fig.update_traces(
        fill="toself",
        fillcolor="rgba(233, 69, 96, 0.18)",
        line=dict(color="#e94560", width=2.5),
    )
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0, 0, 0, 0)",
            radialaxis=dict(
                showticklabels=True,
                ticks="",
                gridcolor="rgba(255, 255, 255, 0.08)",
                linecolor="rgba(255, 255, 255, 0.06)",
            ),
            angularaxis=dict(
                gridcolor="rgba(255, 255, 255, 0.08)",
                linecolor="rgba(255, 255, 255, 0.06)",
            ),
        ),
        paper_bgcolor="rgba(0, 0, 0, 0)",
        plot_bgcolor="rgba(0, 0, 0, 0)",
        font=dict(color="#e0e0e0"),
        margin=dict(l=60, r=60, t=40, b=40),
        height=380,
    )
    return fig


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  SIDEBAR — Weight Sliders                                              ║
# ╚══════════════════════════════════════════════════════════════════════════╝

with st.sidebar:
    st.markdown("## 🎛️ Recommendation Weights")
    st.caption(
        "Adjust the three sliders below. They are automatically normalised "
        "so the effective weights always sum to **100 %**."
    )
    st.markdown("---")

    # Raw slider values (0.0 – 1.0)
    raw_content = st.slider(
        "📊 Content-Based Weight",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.01,
        help="How much to rely on movie metadata & genre similarity.",
    )
    raw_collab = st.slider(
        "👥 Collaborative Weight",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.01,
        help="How much to rely on similar users' preferences.",
    )
    raw_sentiment = st.slider(
        "💬 Sentiment Weight",
        min_value=0.0,
        max_value=1.0,
        value=0.2,
        step=0.01,
        help="How much to rely on audience & critic sentiment.",
    )

    # Normalise to sum = 1.0
    w_c, w_co, w_s = normalise_weights(raw_content, raw_collab, raw_sentiment)

    st.markdown("---")
    st.markdown("### Effective Weights")

    # Show effective (normalised) percentages
    col1, col2, col3 = st.columns(3)
    col1.metric("Content", f"{w_c * 100:.1f}%")
    col2.metric("Collab", f"{w_co * 100:.1f}%")
    col3.metric("Sentiment", f"{w_s * 100:.1f}%")

    # Visual confirmation that they sum to 100 %
    st.progress(1.0)
    st.caption("✅ Weights normalised to 100 %")

    st.markdown("---")
    st.markdown(
        "<div style='text-align:center;color:#8892b0;font-size:0.75rem;'>"
        "CineIQ v0.1 • MLOps Prototype</div>",
        unsafe_allow_html=True,
    )


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  MAIN HEADER                                                            ║
# ╚══════════════════════════════════════════════════════════════════════════╝

st.markdown(
    """
    <div style="text-align: center; padding: 1rem 0 0.5rem;">
        <span style="font-size: 3rem;">🎬</span>
        <h1 style="margin: 0.2rem 0 0;">CINEIQ</h1>
        <p style="color: #8892b0; font-size: 1.05rem; margin-top: 0.2rem;">
            Explainable Hybrid Movie Recommendation Engine
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("---")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 1 — Taste Profile (Radar Chart)                               ║
# ╚══════════════════════════════════════════════════════════════════════════╝

st.markdown("## 🎯 Your Taste Profile")
st.caption("Adjust your genre affinities below. These values directly influence the movies recommended to you in real-time!")

# Two-column layout: chart + stats
chart_col, stats_col = st.columns([3, 1])

with stats_col:
    st.markdown("#### Genre Affinities")
    action_val = st.slider("Action", 0, 100, 82)
    scifi_val = st.slider("Sci-Fi", 0, 100, 91)
    romance_val = st.slider("Romance", 0, 100, 58)
    comedy_val = st.slider("Comedy", 0, 100, 65)
    horror_val = st.slider("Horror", 0, 100, 40)
    
    current_affinities = {
        "Action": action_val,
        "Sci-Fi": scifi_val,
        "Romance": romance_val,
        "Comedy": comedy_val,
        "Horror": horror_val,
    }

with chart_col:
    st.plotly_chart(build_radar_chart(current_affinities), width="stretch")

st.markdown("---")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 2 — Recommendations (Dynamically Ranked)                      ║
# ╚══════════════════════════════════════════════════════════════════════════╝

st.markdown("## 🍿 Recommendations")
st.caption(
    "Movies are scored and ranked in real-time using the sidebar weights. "
    "Move the sliders to see the ranking change instantly."
)

# Get movies sorted by weighted final score, taking taste profile into account
ranked_movies = get_ranked_movies(w_c, w_co, w_s, current_affinities)

for rank, movie in enumerate(ranked_movies, start=1):
    # ---------- movie card container ----------
    st.markdown(
        f"""
        <div class="movie-card">
            <div style="display:flex; align-items:center; gap:1rem;">
                <span class="movie-rank">#{rank}</span>
                <div style="flex:1;">
                    <div class="movie-title">{movie['title']}</div>
                    <div class="score-breakdown">
                        Content: {movie['content_score']}  •  
                        Collaborative: {movie['collaborative_score']}  •  
                        Sentiment: {movie['sentiment_score']}
                    </div>
                </div>
                <span class="movie-score">{movie['final_score']:.1f}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Visual progress bar for the final score (0 – 100)
    st.progress(min(movie["final_score"] / 100, 1.0))

    # Explainability reason
    st.info(f"💡 {movie['explainability_reason']}")

    st.markdown("")  # spacing


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  FOOTER                                                                 ║
# ╚══════════════════════════════════════════════════════════════════════════╝

st.markdown("---")
st.markdown(
    """
    <div style="text-align:center; color:#8892b0; font-size:0.8rem; padding:1rem 0;">
        Built with ❤️ using <strong>Streamlit</strong> & <strong>FastAPI</strong>
        &nbsp;•&nbsp; CineIQ v0.1 — MLOps Prototype
    </div>
    """,
    unsafe_allow_html=True,
)
