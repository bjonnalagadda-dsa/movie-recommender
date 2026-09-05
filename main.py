"""
Movie Recommendation Engine API
---------------------------------
Item-based collaborative filtering over the real MovieLens (small) dataset
(100,836 ratings, 9,742 movies, 610 users), served as a REST API with a
small browser UI.

Run:
    python build_index.py      # once, precomputes model/neighbors.json.gz
    uvicorn main:app --reload

UI:    http://127.0.0.1:8000/
Docs:  http://127.0.0.1:8000/docs
"""
import os

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse

from recommender import MovieRecommender

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = FastAPI(title="Movie Recommendation Engine API")

print("Loading MovieLens data and neighbour index...")
engine = MovieRecommender()
print(f"Ready — {engine.movies_indexed} movies indexed ({engine.source}).")


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(os.path.join(BASE_DIR, "index.html"))


@app.get("/health")
def health():
    return {
        "message": "Movie Recommendation Engine API is running.",
        "movies_indexed": engine.movies_indexed,
        "index_source": engine.source,
    }


@app.get("/search")
def search(q: str = Query(..., description="Partial movie title to search for")):
    results = engine.search_title(q)
    if not results:
        raise HTTPException(status_code=404, detail="No movies matched that search.")
    return results


@app.get("/recommend")
def recommend(title: str = Query(..., description="Movie title (partial match ok)"), top_n: int = 10):
    movie, recs = engine.recommend_by_title(title, top_n=top_n)
    if movie is None:
        raise HTTPException(status_code=404, detail="No movie matched that title.")
    return {"matched_movie": movie, "recommendations": recs}


@app.get("/recommend/{movie_id}")
def recommend_by_id(movie_id: int, top_n: int = 10):
    recs = engine.recommend(movie_id, top_n=top_n)
    if recs is None:
        raise HTTPException(status_code=404, detail="Movie ID not found in dataset.")
    return {
        "movie_id": movie_id,
        "title": engine.movie_id_to_title.get(movie_id),
        "recommendations": recs,
    }
