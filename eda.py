import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = BASE_DIR / "data" / "cleaned_netflix.csv"

FIGURE_DIR = BASE_DIR / "outputs" / "figures"

FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

df = pd.read_csv(DATA_FILE)

print("Dataset Shape:", df.shape)

print("\nDataset Statistics:")
print(df.describe(include="all"))


# ---------------------------------------------------------
# 1. CONTENT TYPE DISTRIBUTION
# ---------------------------------------------------------

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="type"
)

plt.title("Netflix Content Type Distribution")
plt.xlabel("Content Type")
plt.ylabel("Number of Titles")

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "content_type_distribution.png",
    dpi=300
)

plt.show()


# ---------------------------------------------------------
# 2. RELEASE YEAR TREND
# ---------------------------------------------------------

year_counts = (
    df["release_year"]
    .value_counts()
    .sort_index()
)

plt.figure(figsize=(12, 6))

plt.plot(
    year_counts.index,
    year_counts.values
)

plt.title("Netflix Content Release Trend")
plt.xlabel("Release Year")
plt.ylabel("Number of Titles")

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "release_year_trend.png",
    dpi=300
)

plt.show()


# ---------------------------------------------------------
# 3. TOP COUNTRIES
# ---------------------------------------------------------

countries = (
    df["country"]
    .dropna()
    .str.split(",")
    .explode()
    .str.strip()
)

top_countries = (
    countries
    .value_counts()
    .head(10)
)

plt.figure(figsize=(10, 6))

sns.barplot(
    x=top_countries.values,
    y=top_countries.index
)

plt.title("Top 10 Countries by Netflix Content")
plt.xlabel("Number of Titles")
plt.ylabel("Country")

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "top_countries.png",
    dpi=300
)

plt.show()


# ---------------------------------------------------------
# 4. TOP GENRES
# ---------------------------------------------------------

genres = (
    df["listed_in"]
    .dropna()
    .str.split(",")
    .explode()
    .str.strip()
)

top_genres = (
    genres
    .value_counts()
    .head(15)
)

plt.figure(figsize=(10, 7))

sns.barplot(
    x=top_genres.values,
    y=top_genres.index
)

plt.title("Top 15 Netflix Genres")
plt.xlabel("Number of Titles")
plt.ylabel("Genre")

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "top_genres.png",
    dpi=300
)

plt.show()


# ---------------------------------------------------------
# 5. RATING DISTRIBUTION
# ---------------------------------------------------------

plt.figure(figsize=(10, 6))

sns.countplot(
    data=df,
    y="rating",
    order=df["rating"].value_counts().index
)

plt.title("Netflix Rating Distribution")
plt.xlabel("Number of Titles")
plt.ylabel("Rating")

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "rating_distribution.png",
    dpi=300
)

plt.show()


# ---------------------------------------------------------
# 6. MOVIE vs TV SHOW BY YEAR
# ---------------------------------------------------------

type_year = (
    df.groupby(
        ["release_year", "type"]
    )
    .size()
    .reset_index(name="count")
)

plt.figure(figsize=(12, 6))

sns.lineplot(
    data=type_year,
    x="release_year",
    y="count",
    hue="type"
)

plt.title("Movie vs TV Show Release Trend")
plt.xlabel("Release Year")
plt.ylabel("Number of Titles")

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "movie_vs_tv_trend.png",
    dpi=300
)

plt.show()


# ---------------------------------------------------------
# 7. SUMMARY
# ---------------------------------------------------------

print("\n========== EDA SUMMARY ==========")

print(
    "\nContent Type:\n",
    df["type"].value_counts()
)

print(
    "\nTop Ratings:\n",
    df["rating"].value_counts().head()
)

print(
    "\nTop Countries:\n",
    top_countries.head()
)

print(
    "\nTop Genres:\n",
    top_genres.head()
)