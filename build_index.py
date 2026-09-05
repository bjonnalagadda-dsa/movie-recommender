"""
Offline step: precompute the top-K similar movies for every movie and
write them to model/neighbors.json.gz.

This is the one memory-heavy part of the pipeline (it builds the full
~9.7k x 9.7k cosine-similarity matrix). Running it here, once, keeps the
web service itself small enough to run on a free-tier container — at
serve time recommender.py just loads this file.

Run:
    python build_index.py
"""
import gzip
import json
import os

import pandas as pd

from recommender import build_neighbors, RATINGS_PATH, INDEX_PATH, TOP_K


def main():
    print(f"Loading ratings from {os.path.basename(RATINGS_PATH)} ...")
    ratings = pd.read_csv(RATINGS_PATH)
    print(f"Building top-{TOP_K} neighbour index for "
          f"{ratings.movieId.nunique()} movies (this needs ~1.5 GB RAM) ...")
    neighbors = build_neighbors(ratings, top_k=TOP_K)

    os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
    with gzip.open(INDEX_PATH, "wt", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in neighbors.items()}, f, separators=(",", ":"))

    size_mb = os.path.getsize(INDEX_PATH) / (1024 * 1024)
    print(f"Wrote {INDEX_PATH}  ({size_mb:.1f} MB, {len(neighbors)} movies)")


if __name__ == "__main__":
    main()
