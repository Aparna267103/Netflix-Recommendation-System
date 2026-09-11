import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error


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


# ---------------------------------------------------------
# YEARLY CONTENT COUNT
# ---------------------------------------------------------

yearly_content = (
    df.groupby("release_year")
    .size()
    .reset_index(name="content_count")
)

yearly_content = yearly_content.sort_values(
    "release_year"
)


# ---------------------------------------------------------
# REMOVE EXTREME EARLY YEARS
# ---------------------------------------------------------

yearly_content = yearly_content[
    yearly_content["release_year"] >= 2000
].copy()


print("\nYearly content data:")
print(yearly_content.head())


# ---------------------------------------------------------
# FEATURES AND TARGET
# ---------------------------------------------------------

X = yearly_content[
    ["release_year"]
]

y = yearly_content[
    "content_count"
]


# ---------------------------------------------------------
# TRAIN / TEST SPLIT
# ---------------------------------------------------------

split_index = int(
    len(yearly_content) * 0.8
)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]


# ---------------------------------------------------------
# MODEL
# ---------------------------------------------------------

model = LinearRegression()

model.fit(
    X_train,
    y_train
)


# ---------------------------------------------------------
# TEST PREDICTION
# ---------------------------------------------------------

y_pred = model.predict(
    X_test
)


# ---------------------------------------------------------
# EVALUATION
# ---------------------------------------------------------

mae = mean_absolute_error(
    y_test,
    y_pred
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        y_pred
    )
)

print("\n========== MODEL EVALUATION ==========")

print("MAE :", round(mae, 2))
print("RMSE:", round(rmse, 2))


# ---------------------------------------------------------
# FUTURE FORECAST
# ---------------------------------------------------------

last_year = int(
    yearly_content["release_year"].max()
)

future_years = np.arange(
    last_year + 1,
    last_year + 6
).reshape(-1, 1)


future_predictions = model.predict(
    future_years
)


forecast_df = pd.DataFrame({
    "year": future_years.flatten(),
    "predicted_content": future_predictions
})


forecast_df["predicted_content"] = (
    forecast_df["predicted_content"]
    .clip(lower=0)
    .round()
    .astype(int)
)


print("\n========== FUTURE FORECAST ==========")

print(
    forecast_df
)


# ---------------------------------------------------------
# VISUALIZATION
# ---------------------------------------------------------

plt.figure(figsize=(12, 6))

plt.plot(
    yearly_content["release_year"],
    yearly_content["content_count"],
    label="Historical"
)

plt.plot(
    future_years.flatten(),
    future_predictions,
    linestyle="--",
    label="Forecast"
)

plt.title(
    "Netflix Content Trend and Future Forecast"
)

plt.xlabel("Year")

plt.ylabel(
    "Number of Titles"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "content_forecast.png",
    dpi=300
)

plt.show()