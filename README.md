# World Happiness – ML Project (Team 4)

This repository contains the solution for **ML Homework 2 (Team 4)** based on the World Happiness dataset (2015–2024).  
We predict the **Happiness score** of countries and build a simple **time-series forecast** with an interactive dashboard.

## Repository structure

- `Homework_2_team4_with_output.ipynb` – Jupyter/Colab notebook:
  - full project code
  - exploratory data analysis (EDA) with plots
  - model training and evaluation
  - all outputs (tables, metrics, charts) preserved for review
- `Homework_2_team4.py` – main ML script:
  - script version of the project without stored outputs
  - can be run from terminal or VS Code
  - convenient for reproducible runs and automated checks
  - contains data loading and cleaning, EDA, Linear Regression, Random Forest, XGBoost and cross-validation
- `MARIMO2.py` – marimo app with an interactive dashboard:
  - country dropdown + forecast horizon slider
  - line chart with historical happiness and XGBoost forecast
  - scenario sliders for key drivers (GDP, social support, etc.)
- `requirements.txt` – Python dependencies for this project
- `world_happiness_combined.csv` – combined World Happiness dataset (panel data 2015–2024)

## How to run

### 1. Install dependencies

    python3 -m pip install -r requirements.txt

### 2. Run the ML script

    python3 Homework_2_team4.py

This will:

- load the dataset,
- run EDA,
- train and evaluate Linear Regression and Random Forest,
- build the XGBoost time-series model,
- print evaluation metrics and cross-validation results.

### 3. Run the marimo dashboard

If marimo is not installed yet:

    python3 -m pip install marimo

Then start the app:

    marimo edit MARIMO2.py

Open the URL from the terminal (e.g. `http://localhost:7000`), run the cells and use:

- the **Country** dropdown to choose a country,
- the **Forecast horizon** slider to select how many years ahead to show.

The line chart will display both historical `Happiness score` and the XGBoost forecast.

## Methods

- Linear Regression (OLS)
- Random Forest Regressor
- XGBoost Regressor for time-series forecasting
- K-fold cross-validation for tabular models
- TimeSeriesSplit cross-validation for temporal data

---

**Authors:** Zolotukhina Eugeniia, Murchich Natalia, Akhundova Fidan
