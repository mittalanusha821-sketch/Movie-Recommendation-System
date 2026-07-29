"""
Movie Recommendation System — Streamlit App
-----------------------------------
Content-based recommender: converts movie descriptions to TF-IDF vectors,
then uses cosine similarity to find the most similar titles to whichever
movie you pick. From Project 4 of the
"AI Playground: 4 Real-World AI Projects" notebook.

Run locally:
    pip install -r requirements.txt
    streamlit run app.py
"""

import pandas as pd
import streamlit as st

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ------------------------------------------------------------------
# Page config
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Movie Recommender AI",
    page_icon="🎬",
    layout="centered",
)


# ------------------------------------------------------------------
# Dataset (identical to Project 4 in the notebook)
# ------------------------------------------------------------------
MOVIES = pd.DataFrame({
    "title": [
        "Interstellar", "Inception", "The Martian", "Arrival",
        "The Matrix", "Avatar", "Titanic", "The Notebook",
        "Avengers: Endgame", "Iron Man", "Jurassic Park", "The Dark Knight",
    ],
    "description": [
        "space science fiction astronauts future adventure",
        "science fiction dreams technology thriller mind bending",
        "space science fiction astronaut survival mars adventure",
        "science fiction aliens language space mystery",
        "science fiction technology artificial intelligence action",
        "science fiction space aliens adventure fantasy",
        "romance drama ship ocean historical tragedy",
        "romance relationship love drama emotional",
        "superhero action marvel time travel adventure",
        "superhero action technology marvel engineering",
        "dinosaurs science adventure action island",
        "superhero action crime batman thriller",
    ],
})


# ------------------------------------------------------------------
# Build the TF-IDF matrix + similarity matrix once, cache it
# ------------------------------------------------------------------
@st.cache_resource(show_spinner="Building recommendation engine...")
def build_engine():
    vectorizer = TfidfVectorizer(stop_words="english")
    movie_matrix = vectorizer.fit_transform(MOVIES["description"])
    similarity_matrix = cosine_similarity(movie_matrix)
    return vectorizer, movie_matrix, similarity_matrix


def recommend_movies(movie_title, similarity_matrix, number_of_recommendations=5):
    if movie_title not in MOVIES["title"].values:
        return pd.DataFrame(columns=["title", "similarity"])

    movie_index = MOVIES.index[MOVIES["title"] == movie_title][0]
    similarity_scores = list(enumerate(similarity_matrix[movie_index]))

    # Sort by similarity, skip the movie itself (always similarity 1.0 to itself)
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    similarity_scores = [s for s in similarity_scores if s[0] != movie_index]
    top_scores = similarity_scores[:number_of_recommendations]

    results = pd.DataFrame({
        "title": [MOVIES.iloc[i]["title"] for i, _ in top_scores],
        "description": [MOVIES.iloc[i]["description"] for i, _ in top_scores],
        "similarity": [score for _, score in top_scores],
    })
    return results


# ------------------------------------------------------------------
# UI
# ------------------------------------------------------------------
st.title("🎬 Movie Recommendation AI")
st.caption(
    "Content-based filtering: every movie's description is converted to a "
    "TF-IDF vector, then compared with **cosine similarity** to find the "
    "titles that share the most thematic overlap — no user ratings needed."
)

vectorizer, movie_matrix, similarity_matrix = build_engine()

tab_recommend, tab_explore = st.tabs(["🎯 Get Recommendations", "📊 Explore Catalog"])

# --- Tab 1: Recommendations ---
with tab_recommend:
    st.subheader("Pick a movie you like")
    selected_movie = st.selectbox("Movie", MOVIES["title"].tolist())

    num_recs = st.slider("Number of recommendations", min_value=1, max_value=8, value=5)

    if st.button("Recommend Similar Movies", type="primary", use_container_width=True):
        selected_description = MOVIES.loc[MOVIES["title"] == selected_movie, "description"].values[0]
        st.info(f"**{selected_movie}** — _{selected_description}_")

        results = recommend_movies(selected_movie, similarity_matrix, num_recs)

        st.subheader("Because you liked that, you might enjoy:")
        for _, row in results.iterrows():
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"**{row['title']}**")
                st.caption(row["description"])
            with col2:
                st.metric("Similarity", f"{row['similarity'] * 100:.0f}%")
            st.progress(float(row["similarity"]))

        with st.expander("What actually happened here?"):
            st.write(
                "Nothing was hard-coded like 'if you like Interstellar, recommend "
                "The Martian.' Instead, each movie's keyword description became a "
                "TF-IDF vector, and cosine similarity measured the angle between "
                "every pair of vectors — a smaller angle means more shared themes. "
                "The system recommended whichever movies had the smallest angle to "
                "the one you picked, purely from the words in their descriptions."
            )

# --- Tab 2: Catalog explorer ---
with tab_explore:
    st.subheader("Full movie catalog")
    st.dataframe(MOVIES, use_container_width=True, hide_index=True)

    st.subheader("Similarity matrix")
    st.caption("Every movie compared against every other movie (1.0 = identical, 0.0 = no shared themes).")
    sim_df = pd.DataFrame(similarity_matrix, index=MOVIES["title"], columns=MOVIES["title"])
    st.dataframe(sim_df.style.background_gradient(cmap="Blues").format("{:.2f}"), use_container_width=True)

st.divider()
st.caption(
    "Built from Project 4 of *AI Playground: 4 Real-World AI Projects*. "
    "This is content-based filtering — it works from item descriptions, unlike "
    "collaborative filtering (Netflix/Spotify-style), which learns from what "
    "similar users watched or listened to."
)
