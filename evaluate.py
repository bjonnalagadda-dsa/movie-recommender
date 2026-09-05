"""
Evaluation harness for the recommendation engine.

Most tutorial collaborative-filtering projects stop at "here are some
recommendations that look reasonable." This adds an actual offline
evaluation methodology: a train/test split on ratings, then Precision@K
— of the top-K recommendations for a movie a user rated highly in the
training set, how many did that same user also rate highly in the
held-out test set?

This is a standard way recommendation systems are evaluated before ever
being shown to real users (offline evaluation), and it's the kind of
thing that separates "built a demo" from "understands how you'd validate
this in production."
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity

RATING_THRESHOLD = 4.0   # a rating >= this counts as "liked"
TOP_K = 10


def load_data():
    movies = pd.read_csv("movies.csv")
    ratings = pd.read_csv("ratings.csv")
    return movies, ratings


def build_similarity(train_ratings):
    user_item = train_ratings.pivot_table(index="movieId", columns="userId", values="rating").fillna(0)
    sim = cosine_similarity(user_item.values)
    sim_df = pd.DataFrame(sim, index=user_item.index, columns=user_item.index)
    return sim_df


def precision_at_k(sim_df, train_ratings, test_ratings, k=TOP_K):
    """
    For each user in the test set with at least one highly-rated movie in
    train AND at least one highly-rated movie in test, check what fraction
    of the top-K recommendations (based on their favorite train movie)
    appear among their liked test movies.
    """
    liked_train = train_ratings[train_ratings.rating >= RATING_THRESHOLD]
    liked_test = test_ratings[test_ratings.rating >= RATING_THRESHOLD]

    test_liked_by_user = liked_test.groupby("userId")["movieId"].apply(set).to_dict()

    precisions = []
    users_evaluated = 0

    for user_id, group in liked_train.groupby("userId"):
        if user_id not in test_liked_by_user:
            continue
        # use the user's single highest-rated train movie as the "seed"
        seed_movie = group.sort_values("rating", ascending=False).iloc[0]["movieId"]
        if seed_movie not in sim_df.index:
            continue

        top_k_recs = set(sim_df[seed_movie].drop(index=seed_movie).sort_values(ascending=False).head(k).index)
        relevant = test_liked_by_user[user_id]

        hits = len(top_k_recs & relevant)
        precisions.append(hits / k)
        users_evaluated += 1

    return {
        "precision_at_k": round(float(np.mean(precisions)), 4) if precisions else 0.0,
        "users_evaluated": users_evaluated,
        "k": k,
        "rating_threshold_used": RATING_THRESHOLD,
    }


def main():
    movies, ratings = load_data()
    train, test = train_test_split(ratings, test_size=0.2, random_state=42)

    print(f"Total ratings: {len(ratings)}  |  Train: {len(train)}  |  Test: {len(test)}")
    print("Building similarity matrix on training data only (no test leakage)...")
    sim_df = build_similarity(train)

    print("Evaluating Precision@K on held-out test set...")
    results = precision_at_k(sim_df, train, test, k=TOP_K)
    print(results)

    # baseline: what precision would a "recommend the most popular movies" strategy get?
    popularity_baseline = train.groupby("movieId").size().sort_values(ascending=False).head(TOP_K).index
    liked_test = test[test.rating >= RATING_THRESHOLD]
    test_liked_by_user = liked_test.groupby("userId")["movieId"].apply(set).to_dict()
    baseline_precisions = [
        len(set(popularity_baseline) & liked) / TOP_K
        for liked in test_liked_by_user.values()
    ]
    baseline_precision = round(float(np.mean(baseline_precisions)), 4) if baseline_precisions else 0.0
    print(f"Popularity baseline Precision@{TOP_K}: {baseline_precision}")
    print(f"Collaborative filtering lift over baseline: {round(results['precision_at_k'] - baseline_precision, 4)}")


if __name__ == "__main__":
    main()
