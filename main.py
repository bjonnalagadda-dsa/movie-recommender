"""
Movie Recommendation Engine API
---------------------------------
Item-based collaborative filtering over the real MovieLens (small) dataset
(100,836 ratings, 9,742 movies, 610 users), served as a REST API.

Run:
    uvicorn main:app --reload

Docs:
    http://127.0.0.1:8000/docs

Note: model builds an in-memory similarity matrix on startup — this takes
a few seconds due to the dataset size (~9,700 x 9,700 similarity matrix).
"""
from fastapi import FastAPI, HTTPException, Query

from recommender import MovieRecommender

app = FastAPI(title="Movie Recommendation Engine API")

print("Loading MovieLens dataset and building similarity model...")
engine = MovieRecommender()
print("Model ready.")


@app.get("/")
def root():
    return {
        "message": "Movie Recommendation Engine API is running.",
        "movies_loaded": len(engine.movies),
        "ratings_loaded": len(engine.ratings),
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
    return {
        "matched_movie": movie,
        "recommendations": recs,
    }


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
