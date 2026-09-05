"""
Item-based Collaborative Filtering recommendation engine, trained on the
real MovieLens (small) dataset — 100,836 ratings across 9,742 movies from
610 users.

Approach: build a user-item rating matrix, compute cosine similarity
between movies based on how users rated them, and recommend movies most
similar to ones a user already rated highly. This is the classic
"users who liked X also liked Y" pattern used by real recommendation
systems (Amazon, Netflix use variants of this alongside deeper models).
"""
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

MOVIES_PATH = "movies.csv"
RATINGS_PATH = "ratings.csv"


class MovieRecommender:
    def __init__(self):
        self.movies = pd.read_csv(MOVIES_PATH)
        self.ratings = pd.read_csv(RATINGS_PATH)
        self._build_model()

    def _build_model(self):
        # user-item matrix: rows = movies, columns = users
        self.user_item = self.ratings.pivot_table(
            index="movieId", columns="userId", values="rating"
        ).fillna(0)

        # item-item cosine similarity matrix
        similarity = cosine_similarity(self.user_item.values)
        self.similarity_df = pd.DataFrame(
            similarity, index=self.user_item.index, columns=self.user_item.index
        )

        self.movie_id_to_title = dict(zip(self.movies.movieId, self.movies.title))
        self.title_to_movie_id = {v.lower(): k for k, v in self.movie_id_to_title.items()}

        # popularity (rating count) as a fallback / tie-breaker signal
        self.rating_counts = self.ratings.groupby("movieId").size()

    def search_title(self, query: str, limit: int = 5):
        query = query.lower()
        matches = self.movies[self.movies.title.str.lower().str.contains(query, na=False)]
        return matches.head(limit)[["movieId", "title", "genres"]].to_dict(orient="records")

    def recommend(self, movie_id: int, top_n: int = 10):
        if movie_id not in self.similarity_df.index:
            return None

        scores = self.similarity_df[movie_id].drop(index=movie_id)
        scores = scores.sort_values(ascending=False).head(top_n)

        results = []
        for mid, score in scores.items():
            results.append({
                "movieId": int(mid),
                "title": self.movie_id_to_title.get(mid, "Unknown"),
                "similarity_score": round(float(score), 4),
                "num_ratings": int(self.rating_counts.get(mid, 0)),
            })
        return results

    def recommend_by_title(self, title_query: str, top_n: int = 10):
        matches = self.search_title(title_query, limit=1)
        if not matches:
            return None, None
        movie = matches[0]
        recs = self.recommend(movie["movieId"], top_n=top_n)
        return movie, recs
