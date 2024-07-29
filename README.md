# Cannabis Sales Analysis and Prediction

This repository contains code for analyzing and predicting cannabis sales using various machine learning techniques, primarily focusing on XGBoost regression and ARIMA time series forecasting.

## Key Features

- Data preprocessing and feature engineering
- XGBoost regression model with hyperparameter optimization
- ARIMA time series forecasting
- MLflow integration for experiment tracking
- Model versioning and serialization

## File Structure

- `exploratory_data_analysis.ipynb`: Main Jupyter notebook for Exploratory Data Analysis
- `preprocessing.py`: Contains functions for data preparation and feature engineering
- `training.py`: Implements the MLTechniques class for model training and evaluation
-  `prediction.py`: Load the best model and new data to make new predictions.
- `README.md`: This file, providing an overview of the project

## Usage

1. Prepare your data using functions from `preprocessing.py`
2. Train and evaluate models using the MLTechniques class in `training.py`
3. Use MLflow to track experiments and compare model performances
4. Save and version your models for production use

## Dependencies

- pandas
- numpy
- scikit-learn
- xgboost
- statsmodels
- mlflow
- hyperopt

## Getting Started

1. Clone this repository
2. Install the required dependencies
3. Run the Jupyter notebook `EAD.ipynb` for initial data exploration
4. Use the provided scripts for data preprocessing and model training

