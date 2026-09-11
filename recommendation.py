import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------------------------
# PATH
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = BASE_DIR / "data" / "Netflix_Clean.csv"


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

df = pd.read_csv(DATA_FILE)

print("Dataset loaded:", df.shape)


# ---------------------------------------------------------
# CREATE RECOMMENDATION FEATURES
# ---------------------------------------------------------

feature_columns = [
    "type",
    "director",
    "country",
    "rating",
    "listed_in"
]


for column in feature_columns:
    df[column] = (
        df[column]
        .fillna("Unknown")
        .astype(str)
    )


# Combine important features

df["combined_features"] = (
    df["type"] + " " +
    df["director"] + " " +
    df["country"] + " " +
    df["rating"] + " " +
    df["listed_in"]
)


# ---------------------------------------------------------
# TF-IDF
# ---------------------------------------------------------

tfidf = TfidfVectorizer(
    stop_words="english",
    max_features=10000
)

tfidf_matrix = tfidf.fit_transform(
    df["combined_features"]
)

print(
    "TF-IDF matrix shape:",
    tfidf_matrix.shape
)


# ---------------------------------------------------------
# COSINE SIMILARITY
# ---------------------------------------------------------

cosine_sim = cosine_similarity(
    tfidf_matrix,
    tfidf_matrix
)

print("Similarity matrix created.")


# ---------------------------------------------------------
# TITLE INDEX
# ---------------------------------------------------------

indices = pd.Series(
    df.index,
    index=df["title"].str.lower()
).drop_duplicates()


# ---------------------------------------------------------
# RECOMMENDATION FUNCTION
# ---------------------------------------------------------

def recommend_titles(
    title,
    num_recommendations=10
):
    
    title_key = title.lower().strip()
    
    if title_key not in indices:
        return pd.DataFrame(
            columns=[
                "title",
                "type",
                "rating",
                "listed_in"
            ]
        )
    
    idx = indices[title_key]
    
    similarity_scores = list(
        enumerate(cosine_sim[idx])
    )
    
    similarity_scores = sorted(
        similarity_scores,
        key=lambda x: x[1],
        reverse=True
    )
    
    # Remove the selected title itself
    similarity_scores = similarity_scores[1:]
    
    top_scores = similarity_scores[
        :num_recommendations
    ]
    
    movie_indices = [
        item[0]
        for item in top_scores
    ]
    
    result = df.iloc[
        movie_indices
    ][
        [
            "title",
            "type",
            "rating",
            "listed_in"
        ]
    ].copy()
    
    result["similarity_score"] = [
        round(item[1], 3)
        for item in top_scores
    ]
    
    return result


# ---------------------------------------------------------
# TEST RECOMMENDATION
# ---------------------------------------------------------

if __name__ == "__main__":
    
    sample_title = df["title"].iloc[0]
    
    print("\nSelected title:")
    print(sample_title)
    
    recommendations = recommend_titles(
        sample_title,
        10
    )
    
    print("\nRecommended Titles:")
    print(recommendations.to_string(index=False))