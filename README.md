# 🎬 Netflix Recommendation System

## 📌 Project Overview

This project analyzes Netflix content and builds a **content-based recommendation system** using Machine Learning and Natural Language Processing techniques.

The project includes data cleaning, exploratory data analysis (EDA), content trend prediction, content type classification, and a Streamlit dashboard with a recommendation system.

## 🎯 Project Objectives

* Clean and preprocess the Netflix dataset
* Perform exploratory data analysis
* Identify Netflix content trends and patterns
* Predict future content trends
* Classify Netflix content as Movie or TV Show
* Build a content-based recommendation system
* Create an interactive Streamlit dashboard

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* TF-IDF
* Cosine Similarity
* Plotly
* Streamlit
* Joblib

## 📂 Project Structure

```text
Netflix-Recommendation-System/
│
├── src/
│   ├── data_cleaning.py
│   ├── eda.py
│   ├── recommendation.py
│   ├── trend_prediction.py
│   └── classification.py
│
├── app/
│   └── app.py
│
├── README.md
└── requirements.txt
```

## 🔄 Project Workflow

1. **Data Cleaning**

   * Handle missing values
   * Remove duplicate records
   * Clean text columns
   * Transform date information
   * Create additional features

2. **Exploratory Data Analysis**

   * Analyze content types
   * Analyze release-year trends
   * Identify top countries
   * Identify popular genres
   * Analyze content ratings

3. **Recommendation System**

   * Combine relevant content features
   * Apply TF-IDF Vectorization
   * Calculate Cosine Similarity
   * Recommend similar Netflix titles

4. **Trend Prediction**

   * Analyze yearly content counts
   * Apply Linear Regression
   * Evaluate the model using MAE and RMSE
   * Forecast future content trends

5. **Classification**

   * Classify content as Movie or TV Show
   * Compare Logistic Regression and Random Forest
   * Evaluate using Accuracy, Precision, Recall and F1 Score
   * Save the best-performing model

6. **Streamlit Dashboard**

   * Interactive Netflix dashboard
   * Content filters
   * KPIs
   * Charts and visualizations
   * Content recommendations
   * Business insights

## 🚀 How to Run

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Streamlit Dashboard

```bash
streamlit run app/app.py
```

## 💡 Key Business Insights

The project provides insights into:

* Netflix content distribution
* Popular genres and countries
* Movie vs TV Show trends
* Content ratings
* Release-year patterns
* Similar-content recommendations
* Future content trends

## 📌 Conclusion

This project demonstrates how Python, Machine Learning, NLP, data analysis, and visualization techniques can be combined to analyze Netflix content and build a content recommendation system.
