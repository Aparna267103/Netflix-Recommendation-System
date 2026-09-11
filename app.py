import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

import sys

# =========================================================
# PAGE CONFIGURATION AND PATHS
# =========================================================

st.set_page_config(
    page_title="Netflix Data Science Dashboard",
    page_icon="🎬",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent.parent

sys.path.append(
    str(BASE_DIR / "src")
)

from src.recommendation import recommend_titles

# =========================================================
# PATH
# =========================================================

DATA_FILE = (
    BASE_DIR /
    "data" /
    "Netflix_Clean.csv"
)


# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_data():
    return pd.read_csv(DATA_FILE)


df = load_data()


# =========================================================
# CONTENT RECOMMENDATIONS
# =========================================================

st.divider()

st.header("🎯 Content Recommendation System")

selected_title = st.selectbox(
    "Select a Netflix title",
    sorted(df["title"].dropna().unique())
)

if st.button("Get Recommendations"):

    recommendations = recommend_titles(
        selected_title,
        10
    )

    if recommendations.empty:

        st.warning(
            "No recommendations found."
        )

    else:

        st.success(
            f"Recommendations for: {selected_title}"
        )

        st.dataframe(
            recommendations,
            use_container_width=True
        )


# =========================================================
# TITLE
# =========================================================

st.title(
    "🎬 Netflix Data Science Dashboard"
)

st.markdown(
    """
    This dashboard provides insights into Netflix content,
    genres, countries, ratings, release trends and
    recommendation-oriented analysis.
    """
)


st.divider()


# =========================================================
# SIDEBAR FILTERS
# =========================================================

st.sidebar.header(
    "🔎 Filters"
)


content_types = st.sidebar.multiselect(
    "Content Type",
    options=sorted(df["type"].dropna().unique()),
    default=sorted(df["type"].dropna().unique())
)


ratings = st.sidebar.multiselect(
    "Rating",
    options=sorted(df["rating"].dropna().unique()),
    default=sorted(df["rating"].dropna().unique())
)


filtered_df = df[
    (df["type"].isin(content_types)) &
    (df["rating"].isin(ratings))
]


# =========================================================
# KPI SECTION
# =========================================================

col1, col2, col3, col4 = st.columns(4)


with col1:
    st.metric(
        "Total Titles",
        f"{len(filtered_df):,}"
    )


with col2:
    st.metric(
        "Movies",
        f"{(filtered_df['type'] == 'Movie').sum():,}"
    )


with col3:
    st.metric(
        "TV Shows",
        f"{(filtered_df['type'] == 'TV Show').sum():,}"
    )


with col4:
    st.metric(
        "Countries",
        filtered_df["country"].nunique()
    )


st.divider()


# =========================================================
# CONTENT TYPE CHART
# =========================================================

col1, col2 = st.columns(2)


with col1:

    type_counts = (
        filtered_df["type"]
        .value_counts()
        .reset_index()
    )

    type_counts.columns = [
        "type",
        "count"
    ]

    fig = px.pie(
        type_counts,
        names="type",
        values="count",
        title="Content Type Distribution",
        hole=0.4
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# =========================================================
# RATING CHART
# =========================================================

with col2:

    rating_counts = (
        filtered_df["rating"]
        .value_counts()
        .reset_index()
    )

    rating_counts.columns = [
        "rating",
        "count"
    ]

    fig = px.bar(
        rating_counts,
        x="rating",
        y="count",
        title="Content Rating Distribution"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# =========================================================
# RELEASE YEAR TREND
# =========================================================

yearly = (
    filtered_df
    .groupby(
        ["release_year", "type"]
    )
    .size()
    .reset_index(
        name="count"
    )
)


fig = px.line(
    yearly,
    x="release_year",
    y="count",
    color="type",
    markers=True,
    title="Netflix Content Release Trend"
)

st.plotly_chart(
    fig,
    use_container_width=True
)


# =========================================================
# TOP GENRES
# =========================================================

genres = (
    filtered_df["listed_in"]
    .dropna()
    .str.split(",")
    .explode()
    .str.strip()
)

top_genres = (
    genres
    .value_counts()
    .head(15)
    .reset_index()
)

top_genres.columns = [
    "genre",
    "count"
]


fig = px.bar(
    top_genres,
    x="count",
    y="genre",
    orientation="h",
    title="Top 15 Genres"
)

st.plotly_chart(
    fig,
    use_container_width=True
)


# =========================================================
# TOP COUNTRIES
# =========================================================

countries = (
    filtered_df["country"]
    .dropna()
    .str.split(",")
    .explode()
    .str.strip()
)

top_countries = (
    countries
    .value_counts()
    .head(15)
    .reset_index()
)

top_countries.columns = [
    "country",
    "count"
]


fig = px.bar(
    top_countries,
    x="count",
    y="country",
    orientation="h",
    title="Top 15 Countries"
)

st.plotly_chart(
    fig,
    use_container_width=True
)


# =========================================================
# DATA TABLE
# =========================================================

st.subheader(
    "📋 Filtered Netflix Titles"
)

st.dataframe(
    filtered_df[
        [
            "show_id",
            "type",
            "title",
            "country",
            "release_year",
            "rating",
            "duration",
            "listed_in"
        ]
    ],
    use_container_width=True
)


# =========================================================
# BUSINESS INSIGHTS
# =========================================================

st.divider()

st.header(
    "💡 Business Insights"
)


total_titles = len(filtered_df)

movie_count = (
    filtered_df["type"] == "Movie"
).sum()

tv_count = (
    filtered_df["type"] == "TV Show"
).sum()


st.markdown(
    f"""
    ### Key Findings

    - The filtered dataset contains **{total_titles:,} titles**.
    - Movies account for **{movie_count:,} titles**.
    - TV Shows account for **{tv_count:,} titles**.
    - The dashboard highlights the most represented genres,
      countries and content ratings.
    - Release-year analysis helps identify content growth
      patterns over time.
    """
)


st.info(
    """
    Business Recommendation:

    Netflix can use content distribution, genre popularity,
    geographic availability and release trends to support
    content acquisition, regional strategy and catalogue
    planning decisions.
    """
)
