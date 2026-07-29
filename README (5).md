# Movie Recommendation System — Streamlit App

Content-based movie recommender using **TF-IDF + cosine similarity**. This
is Project 4 from the notebook, wrapped in an interactive UI:

- Pick any movie from a dropdown, get the most similar titles ranked by
  similarity score, with a progress bar and short explanation
- Adjustable number of recommendations
- A full catalog explorer with the complete similarity matrix, heat-mapped

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the URL Streamlit prints (usually http://localhost:8501).

This app is lightweight — no heavy ML libraries (no TensorFlow/OpenCV), so
it installs and starts almost instantly.

## Deploy for free on Streamlit Community Cloud

1. Push `app.py` and `requirements.txt` to a GitHub repo (same folder, no
   duplicate/renamed files).
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click **New app**, pick your repo/branch, set the main file path to
   `app.py` exactly.
4. Click **Deploy**.

## Deploy elsewhere (Render, Railway, Hugging Face Spaces, etc.)

Start command:

```bash
streamlit run app.py --server.port $PORT --server.address 0.0.0.0
```

## Extend it

The 12-movie dataset is hard-coded at the top of `app.py` as a pandas
DataFrame with `title` and `description` columns. To add more movies, just
add rows with a short keyword-style description (genre, themes, setting) —
the more descriptive and specific the keywords, the better the
recommendations will be.
