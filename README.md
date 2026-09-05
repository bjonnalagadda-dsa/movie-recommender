# Movie Recommendation Engine (Item-Based Collaborative Filtering)

A recommendation engine trained on the real **MovieLens (small) dataset**
— 100,836 ratings across 9,742 movies from 610 users — served as a REST
API.

## Why this project

Demonstrates a different ML technique than NLP/text classification:
collaborative filtering, the foundation of real-world recommendation
systems (Netflix, Amazon, Spotify use variants of this alongside deeper
models). Shows comfort with matrix operations, similarity metrics, and
working with a genuine public dataset rather than synthetic data.

## Stack

- **FastAPI** — REST API
- **Pandas** — data loading and the user-item pivot table
- **scikit-learn (cosine similarity)** — item-item similarity computation
- **MovieLens (small) dataset** — real public data (`movies.csv`, `ratings.csv`)

## How it works

1. Build a user-item ratings matrix (movies × users)
2. Compute cosine similarity between every pair of movies based on how
   users rated them (movies rated similarly by the same users → high
   similarity)
3. Given a movie, return the most similar movies by that score —
   the classic "users who liked X also liked Y" pattern

## Run locally

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

Note: building the similarity matrix on startup takes ~10-15 seconds due
to dataset size — this is expected, not a bug.

Then open `http://127.0.0.1:8000/docs`.

## Example

```bash
curl "http://127.0.0.1:8000/recommend?title=toy%20story&top_n=5"
```

Returns movies most similar to Toy Story based on real user rating
patterns (verified output includes Toy Story 2, Jurassic Park, Star
Wars: A New Hope — sensible for a 1995 family/adventure film).

## Offline Evaluation (Precision@K)

Most collaborative-filtering demos stop at "recommendations look
reasonable." This includes an actual offline evaluation: an 80/20
train/test split on ratings, with the similarity matrix built only on
training data (no test leakage), then Precision@10 measured on held-out
"liked" movies (rating ≥ 4.0).

Run it:
```bash
python evaluate.py
```

**Real result on this dataset:**
```
Collaborative filtering Precision@10: 0.0414
Popularity baseline Precision@10:     0.0664
```

**The popularity baseline actually outperformed pure item-based CF here** —
and that's a genuine, worth-reporting finding, not a bug. Likely causes:
- The MovieLens-small dataset is sparse (610 users, ~9,700 movies) —
  item-item similarity is noisy when few users have rated any given pair
- Using a single "seed" movie per user (their top-rated film) throws away
  most of that user's signal
- Popular movies are popular partly *because* they're broadly liked, so a
  naive popularity baseline is a genuinely strong signal in this domain —
  a well-known challenge in recommender systems research, not a flaw
  specific to this implementation

**What I'd try next given this result:** blend multiple seed movies per
user instead of just one, or move to matrix factorization (SVD), which
handles sparsity better than raw item-item cosine similarity.

## Possible extensions (mentioned honestly — not yet built)

- Matrix factorization (SVD) to address the sparsity issue found above
- Hybrid filtering combining genres (content-based) with collaborative signal
- Cold-start handling for movies/users with very few ratings
