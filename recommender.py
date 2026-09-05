"""
Item-based Collaborative Filtering recommendation engine, trained on the
real MovieLens (small) dataset — 100,836 ratings across 9,742 movies from
610 users.

Approach: build a user-item rating matrix, compute cosine similarity
between movies based on how users rated them, and recommend movies most
similar to ones a user already rated highly. This is the classic
"users who liked X also liked Y" pattern used by real recommendation
systems (Amazon, Netflix use variants of this alongside deeper models).

Serving vs. training
--------------------
Building the full 9.7k x 9.7k similarity matrix needs well over a GB of
RAM, which does not fit a small free-tier container. So `build_index.py`
precomputes the top-K neighbours per movie once and writes them to
`model/neighbors.json.gz` (a few MB). At serve time we load only that
index — fast startup, tens of MB of memory. If the index file is absent
(e.g. local dev before running build_index.py), we fall back to building
the matrix in memory.
"""
import gzip
import json
import os
import re

import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MOVIES_PATH = os.path.join(BASE_DIR, "movies.csv")
RATINGS_PATH = os.path.join(BASE_DIR, "ratings.csv")
INDEX_PATH = os.path.join(BASE_DIR, "model", "neighbors.json.gz")

TOP_K = 50  # neighbours stored per movie in the precomputed index
_ARTICLE_RE = re.compile(r"^(the|a|an)\s+", re.IGNORECASE)


def build_neighbors(ratings: pd.DataFrame, top_k: int = TOP_K) -> dict:
    """Compute the top-`top_k` most similar movies for every movie.

    Returns {movie_id: [[neighbor_id, similarity, neighbor_num_ratings], ...]}
    sorted by similarity descending. This is the one memory-heavy step and
    is meant to run offline (build_index.py), not on the serving box, so
    sklearn is imported lazily here rather than at module load.
    """
    from sklearn.metrics.pairwise import cosine_similarity

    user_item = ratings.pivot_table(
        index="movieId", columns="userId", values="rating"
    ).fillna(0)

    sim = cosine_similarity(user_item.values.astype(np.float32))
    ids = [int(m) for m in user_item.index]
    counts = ratings.groupby("movieId").size().to_dict()

    neighbors = {}
    for i, mid in enumerate(ids):
        row = sim[i]
        order = np.argsort(row)[::-1]
        picks = []
        for j in order:
            nbr = ids[j]
            if nbr == mid:
                continue
            picks.append([nbr, round(float(row[j]), 4), int(counts.get(nbr, 0))])
            if len(picks) >= top_k:
                break
        neighbors[mid] = picks
    return neighbors


def _load_index(path: str) -> dict:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


class MovieRecommender:
    def __init__(self):
        self.movies = pd.read_csv(MOVIES_PATH)
        self.movie_id_to_title = dict(zip(self.movies.movieId, self.movies.title))
        self.movie_id_to_genres = dict(zip(self.movies.movieId, self.movies.genres))

        if os.path.exists(INDEX_PATH):
            self.neighbors = _load_index(INDEX_PATH)
            self.source = "precomputed index"
        else:
            # Local fallback: build it in memory (needs ~1.5 GB RAM).
            ratings = pd.read_csv(RATINGS_PATH)
            self.neighbors = build_neighbors(ratings)
            self.source = "in-memory matrix (no index file found)"

        self.movies_indexed = len(self.neighbors)

    # -- search -------------------------------------------------------------
    def search_title(self, query: str, limit: int = 5):
        q = query.strip().lower()
        titles = self.movies.title.str.lower()

        # 1) direct substring ("toy story" -> "Toy Story (1995)")
        mask = titles.str.contains(re.escape(q), na=False)

        # 2) fall back to "all query words appear in the title", after
        #    dropping a leading article, so "the matrix" finds "Matrix, The".
        if not mask.any():
            words = [w for w in _ARTICLE_RE.sub("", q).split() if w]
            if words:
                mask = titles.apply(lambda t: all(w in t for w in words))

        matches = self.movies[mask]
        return matches.head(limit)[["movieId", "title", "genres"]].to_dict(orient="records")

    # -- recommend --------------------------------------------------------
    def recommend(self, movie_id: int, top_n: int = 10):
        picks = self.neighbors.get(int(movie_id))
        if picks is None:
            return None
        results = []
        for nbr_id, score, n_ratings in picks[:top_n]:
            results.append({
                "movieId": int(nbr_id),
                "title": self.movie_id_to_title.get(nbr_id, "Unknown"),
                "genres": self.movie_id_to_genres.get(nbr_id, ""),
                "similarity_score": round(float(score), 4),
                "num_ratings": int(n_ratings),
            })
        return results

    def recommend_by_title(self, title_query: str, top_n: int = 10):
        matches = self.search_title(title_query, limit=1)
        if not matches:
            return None, None
        movie = matches[0]
        recs = self.recommend(movie["movieId"], top_n=top_n)
        return movie, recs
