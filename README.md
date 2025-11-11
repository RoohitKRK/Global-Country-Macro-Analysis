# Global Country Macro and Health Indicators Regression Analysis

This project is a comprehensive regression analysis utilizing various machine learning and deep learning models to predict a key public health metric: **Infant mortality**. The models are trained on a global dataset of macroeconomic and health indicators.

## Dataset

The primary dataset used is `world-data-2023.csv`. It contains 35 columns of global country data, including metrics like:
* `Density (P/Km2)`
* `GDP`
* `Birth Rate`
* `Life expectancy`
* `Maternal mortality ratio`
* `Total tax rate`

The goal of the analysis is to predict the `Infant mortality` rate based on the other features.

## Methodology

The analysis follows a standard machine learning workflow:

1.  **Data Preprocessing and Cleaning**: Initial steps included cleaning the raw data, handling missing values, and transforming object types into numerical data suitable for modeling. The final dataset used 22 cleaned features.
2.  **Exploratory Data Analysis (EDA)**: The data distributions for all numerical features were visualized using histograms.
3.  **Feature Engineering**: Features were scaled using `StandardScaler`.
4.  **Model Training**: The dataset was split into training and testing sets (80-20 split) for training and evaluating a variety of regression models:
    * **Machine Learning Models**
        * Linear Regression
        * Decision Tree Regressor
        * Random Forest Regressor
        * K-Nearest Neighbors (KNeighbors Regressor)
        * AdaBoost Regressor
        * Gradient Boosting Regressor
    * **Deep Learning Model**
        * A Sequential Keras Model (built with `tensorflow` and `keras`).

## Prerequisites

To run the notebook locally, you need a Python environment with the required packages.

