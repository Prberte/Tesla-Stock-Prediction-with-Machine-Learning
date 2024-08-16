# Tesla Stock Prediction with Machine Learning

![Tesla Logo](https://upload.wikimedia.org/wikipedia/commons/thumb/e/e8/Tesla_logo.png/1024px-Tesla_logo.png)

## Overview

This project aims to predict Tesla's stock prices using various machine learning techniques. It leverages historical stock data and implements different machine learning models to analyze trends and predict future stock prices. The goal is to provide insights into how well these models can forecast stock price movements and to understand the underlying factors influencing Tesla's stock.

## Features

- **Data Collection & Preprocessing:** 
  - Utilizes historical stock data for Tesla.
  - Data cleaning and feature engineering processes are applied.
  
- **Machine Learning Models Implemented:**
  - Linear Regression
  - Decision Trees
  - Random Forest
  - Support Vector Machine (SVM)
  - Long Short-Term Memory (LSTM) Networks

- **Evaluation Metrics:**
  - Mean Absolute Error (MAE)
  - Root Mean Squared Error (RMSE)
  - R-Squared (R²)
  
- **Visualization:**
  - Stock price trends
  - Model predictions vs. actual prices

## Requirements

To run the code in this repository, you need the following Python libraries:

- Python 3.x
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- keras (for LSTM implementation)
- yfinance (for data collection)

You can install the required packages using the following command:

```bash
pip install -r requirements.txt
```

## Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/Prberte/Tesla-Stock-Prediction-with-Machine-Learning.git
   cd Tesla-Stock-Prediction-with-Machine-Learning
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Jupyter Notebook:**
   Open `Tesla_Stock_Prediction.ipynb` in Jupyter Notebook or any compatible environment to execute the code and view the results.

## Data Collection

The stock data is sourced using the `yfinance` library, which fetches historical stock prices for Tesla (TSLA). The dataset includes:

- **Date:** The trading date.
- **Open:** The opening price of the stock.
- **High:** The highest price of the stock during the trading day.
- **Low:** The lowest price of the stock during the trading day.
- **Close:** The closing price of the stock.
- **Volume:** The volume of shares traded.

## Model Implementation

### 1. Linear Regression
A basic model that fits a linear equation to observed data.

### 2. Decision Trees
A non-linear model that makes decisions based on the data features.

### 3. Random Forest
An ensemble method that uses multiple decision trees to improve the prediction accuracy.

### 4. Support Vector Machine (SVM)
A model that finds the hyperplane that best separates the data into classes.

### 5. Long Short-Term Memory (LSTM) Networks
A type of recurrent neural network (RNN) suitable for time-series prediction.

## Evaluation

The models are evaluated based on several metrics to understand their performance:

- **Mean Absolute Error (MAE):** Measures the average magnitude of errors in a set of predictions.
- **Root Mean Squared Error (RMSE):** Penalizes larger errors more than smaller ones.
- **R-Squared (R²):** Indicates the proportion of the variance in the dependent variable that is predictable from the independent variable(s).

## Results

The project compares the performance of different models and visualizes the predictions versus actual stock prices. The results indicate how well each model can forecast Tesla's stock prices based on historical data.

## Contributing

Contributions to this project are welcome! If you would like to add new models, improve the existing ones, or suggest any enhancements, feel free to submit a pull request or open an issue.

## Acknowledgments

- [Yahoo Finance](https://finance.yahoo.com) for providing the stock data.
- All contributors who helped develop and maintain this project.
