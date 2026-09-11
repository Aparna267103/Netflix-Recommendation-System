import pandas as pd
import numpy as np
from pathlib import Path


# ---------------------------------------------------------
# 1. PATH CONFIGURATION
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = BASE_DIR / "data" / "Dataset.csv"
OUTPUT_FILE = BASE_DIR / "data" / "cleaned_netflix.csv"


# ---------------------------------------------------------
# 2. LOAD DATA
# ---------------------------------------------------------

def load_data(file_path):
    """Load Netflix dataset."""
    
    df = pd.read_csv(file_path)
    
    print("Dataset loaded successfully.")
    print(f"Rows    : {df.shape[0]}")
    print(f"Columns : {df.shape[1]}")
    
    return df


# ---------------------------------------------------------
# 3. BASIC DATA INSPECTION
# ---------------------------------------------------------

def inspect_data(df):
    """Display basic information about dataset."""
    
    print("\n========== DATASET INFORMATION ==========")
    print(df.info())
    
    print("\n========== FIRST 5 ROWS ==========")
    print(df.head())
    
    print("\n========== MISSING VALUES ==========")
    print(df.isnull().sum())
    
    print("\n========== DUPLICATE ROWS ==========")
    print(df.duplicated().sum())


# ---------------------------------------------------------
# 4. HANDLE MISSING VALUES
# ---------------------------------------------------------

def handle_missing_values(df):
    """Handle missing values column-wise."""
    
    # Text columns
    text_columns = [
        "director",
        "country",
        "date_added",
        "rating",
        "duration"
    ]
    
    for column in text_columns:
        if column in df.columns:
            df[column] = df[column].fillna("Unknown")
    
    return df


# ---------------------------------------------------------
# 5. REMOVE DUPLICATES
# ---------------------------------------------------------

def remove_duplicates(df):
    """Remove duplicate records."""
    
    before = len(df)
    
    df = df.drop_duplicates()
    
    after = len(df)
    
    print(f"\nDuplicates removed: {before - after}")
    
    return df


# ---------------------------------------------------------
# 6. CLEAN TEXT COLUMNS
# ---------------------------------------------------------

def clean_text_columns(df):
    """Standardize text values."""
    
    text_columns = [
        "type",
        "title",
        "director",
        "country",
        "rating",
        "duration",
        "listed_in"
    ]
    
    for column in text_columns:
        if column in df.columns:
            df[column] = (
                df[column]
                .astype(str)
                .str.strip()
                .str.replace(r"\s+", " ", regex=True)
            )
    
    return df


# ---------------------------------------------------------
# 7. DATE TRANSFORMATION
# ---------------------------------------------------------

def transform_date(df):
    """Convert date_added into datetime format."""
    
    if "date_added" in df.columns:
        
        df["date_added"] = pd.to_datetime(
            df["date_added"],
            errors="coerce"
        )
        
        # Additional date features
        df["date_added_year"] = df["date_added"].dt.year
        df["date_added_month"] = df["date_added"].dt.month
        df["date_added_month_name"] = (
            df["date_added"].dt.month_name()
        )
    
    return df


# ---------------------------------------------------------
# 8. FEATURE ENGINEERING
# ---------------------------------------------------------

def create_features(df):
    """Create useful analytical features."""
    
    # Number of genres
    if "listed_in" in df.columns:
        df["genre_count"] = (
            df["listed_in"]
            .fillna("")
            .apply(
                lambda x: len(
                    [item for item in x.split(",") if item.strip()]
                )
            )
        )
    
    # Number of countries
    if "country" in df.columns:
        df["country_count"] = (
            df["country"]
            .fillna("")
            .apply(
                lambda x: len(
                    [item for item in x.split(",") if item.strip()]
                )
            )
        )
    
    # Number of directors
    if "director" in df.columns:
        df["director_count"] = (
            df["director"]
            .fillna("")
            .apply(
                lambda x: len(
                    [item for item in x.split(",") if item.strip()]
                )
            )
        )
    
    # Duration numeric value
    if "duration" in df.columns:
        df["duration_value"] = pd.to_numeric(
            df["duration"].str.extract(r"(\d+)")[0],
            errors="coerce"
        )
    
    return df


# ---------------------------------------------------------
# 9. FINAL CLEANING
# ---------------------------------------------------------

def final_cleaning(df):
    """Final validation before saving."""
    
    # Standardize column names
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )
    
    # Sort by release year
    if "release_year" in df.columns:
        df = df.sort_values(
            by="release_year",
            ascending=True
        )
    
    # Reset index
    df = df.reset_index(drop=True)
    
    return df


# ---------------------------------------------------------
# 10. MAIN PIPELINE
# ---------------------------------------------------------

def main():
    
    df = load_data(INPUT_FILE)
    
    inspect_data(df)
    
    df = handle_missing_values(df)
    
    df = remove_duplicates(df)
    
    df = clean_text_columns(df)
    
    df = transform_date(df)
    
    df = create_features(df)
    
    df = final_cleaning(df)
    
    df.to_csv(
        OUTPUT_FILE,
        index=False
    )
    
    print("\n======================================")
    print("DATA CLEANING COMPLETED")
    print("======================================")
    print(f"Cleaned dataset saved to:")
    print(OUTPUT_FILE)
    print(f"Final shape: {df.shape}")


if __name__ == "__main__":
    main()