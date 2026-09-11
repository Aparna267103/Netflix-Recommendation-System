import pandas as pd
import numpy as np
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = BASE_DIR / "data" / "cleaned_netflix.csv"

MODEL_DIR = BASE_DIR / "outputs" / "models"

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

df = pd.read_csv(DATA_FILE)


# ---------------------------------------------------------
# FEATURE ENGINEERING
# ---------------------------------------------------------

# Duration numeric value

df["duration_numeric"] = pd.to_numeric(
    df["duration"]
    .astype(str)
    .str.extract(r"(\d+)")[0],
    errors="coerce"
)


# First genre

df["primary_genre"] = (
    df["listed_in"]
    .fillna("Unknown")
    .str.split(",")
    .str[0]
    .str.strip()
)


# First country

df["primary_country"] = (
    df["country"]
    .fillna("Unknown")
    .str.split(",")
    .str[0]
    .str.strip()
)


# ---------------------------------------------------------
# SELECT FEATURES
# ---------------------------------------------------------

features = [
    "release_year",
    "rating",
    "duration_numeric",
    "primary_genre",
    "primary_country"
]

target = "type"


X = df[features]

y = df[target]


# ---------------------------------------------------------
# TRAIN TEST SPLIT
# ---------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ---------------------------------------------------------
# PREPROCESSING
# ---------------------------------------------------------

numeric_features = [
    "release_year",
    "duration_numeric"
]

categorical_features = [
    "rating",
    "primary_genre",
    "primary_country"
]


numeric_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        )
    ]
)


categorical_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            numeric_transformer,
            numeric_features
        ),
        (
            "cat",
            categorical_transformer,
            categorical_features
        )
    ]
)


# ---------------------------------------------------------
# MODELS
# ---------------------------------------------------------

models = {

    "Logistic Regression":
        LogisticRegression(
            max_iter=1000
        ),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        )
}


# ---------------------------------------------------------
# TRAIN & EVALUATE
# ---------------------------------------------------------

results = []

best_model = None
best_score = 0
best_model_name = ""


for name, classifier in models.items():
    
    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "classifier",
                classifier
            )
        ]
    )
    
    pipeline.fit(
        X_train,
        y_train
    )
    
    predictions = pipeline.predict(
        X_test
    )
    
    accuracy = accuracy_score(
        y_test,
        predictions
    )
    
    precision = precision_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )
    
    recall = recall_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )
    
    f1 = f1_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )
    
    results.append({
        "Model": name,
        "Accuracy": round(accuracy, 4),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1 Score": round(f1, 4)
    })
    
    print("\n================================")
    print(name)
    print("================================")
    
    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )
    
    if accuracy > best_score:
        best_score = accuracy
        best_model = pipeline
        best_model_name = name


# ---------------------------------------------------------
# MODEL COMPARISON
# ---------------------------------------------------------

results_df = pd.DataFrame(results)

print("\n========== MODEL COMPARISON ==========")

print(
    results_df.to_string(index=False)
)


# ---------------------------------------------------------
# SAVE BEST MODEL
# ---------------------------------------------------------

model_path = (
    MODEL_DIR /
    "netflix_type_classifier.pkl"
)

joblib.dump(
    best_model,
    model_path
)

print("\nBest Model:", best_model_name)

print(
    "Best Accuracy:",
    round(best_score, 4)
)

print(
    "\nModel saved to:",
    model_path
)