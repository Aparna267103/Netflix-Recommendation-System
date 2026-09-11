# Netflix Recommendation System

## 📌 Project Overview

The Netflix Recommendation System is a machine learning-based application designed to recommend movies and TV shows based on their content characteristics.

The project analyzes Netflix titles and uses content-based recommendation techniques to identify titles that are similar to the user's selected movie or TV show.

## 🎯 Objectives

* Analyze Netflix movie and TV show data.
* Perform data cleaning and preprocessing.
* Explore important features in the dataset.
* Build a content-based recommendation system.
* Recommend similar movies or TV shows based on the selected title.
* Deploy the recommendation system using Streamlit.

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Streamlit
* Matplotlib / Seaborn
* Jupyter Notebook
* VS Code

## 🔄 Project Workflow

```text
Netflix Dataset
      ↓
Data Cleaning
      ↓
Data Preprocessing
      ↓
Feature Selection
      ↓
Feature Transformation
      ↓
Similarity Calculation
      ↓
Recommendation System
      ↓
Streamlit Application
```

## 🤖 Recommendation Process

The system uses a content-based recommendation approach.

When a user selects a movie or TV show, the system compares its available content-related features with other titles in the dataset.

Based on the similarity between titles, the system returns a list of recommended movies or TV shows.

## 🌐 Streamlit Application

The project is deployed as an interactive Streamlit application.

The application allows users to:

* Select a Netflix title.
* Generate recommendations.
* View similar movies or TV shows.
* Interact with the recommendation system through a simple web interface.

## 📂 Project Structure

```text
Netflix-Recommendation-System/
│
├── app.py
├── netflix_recommendation.ipynb
├── requirements.txt
├── README.md
│
├── data/
│   └── netflix_titles.csv
│
└── screenshots/
```

## ▶️ How to Run

### 1. Clone the repository

```bash
git clone <your-github-repository-link>
```

### 2. Install required libraries

```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit application

```bash
streamlit run app.py
```

The application will open in the browser.

## 📊 Results

The system provides movie and TV show recommendations based on similarity between Netflix titles.

## 🔮 Future Improvements

* Add user-based recommendation.
* Combine content-based and collaborative filtering.
* Improve recommendation accuracy.
* Add user ratings and preferences.
* Deploy the application online.

## 👩‍💻 Project Author

Aparna V

This project was developed as part of an internship/project learning experience.

