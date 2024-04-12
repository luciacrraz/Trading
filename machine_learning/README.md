# Machine Learning Project

## Introduction

This project focuses on leveraging machine learning (ML) models to optimize and evaluate trading strategies using provided datasets. Below, we outline the project's structure and objectives.

## Project Structure

The project follows a Python project structure, consisting of the following components:

- **_init_.py**: Initialization script for the machine learning module.
- **backtest.py**: Conducts backtesting of trading strategies using ML models.
- **main.py**: Orchestrates the execution of machine learning workflows.
- **optimization.py**: Handles optimization of backtest parameters and strategies.
- **signals.py**: Defines buy/sell signals based on ML model predictions.

## Objectives

1. Utilize provided training and validation datasets for strategy optimization.
2. Utilize Logistic Regression, Support Vector Machine, and XGBoost ML models for strategy evaluation.
3. Define independent and dependent variables for model training, incorporating additional technical indicators if necessary.
4. Split train datasets into train/test sets and construct appropriate dependent variables (e.g., "Buy" and "Not buy", "Sell" and "Not sell").
5. Fine-tune hyperparameters for each ML model to generate accurate buy/sell signals for backtesting.
6. Conduct backtesting for each dataset timeframe (1d, 1h, 5m, 1m), tracking operations and portfolio value.
7. Optimize backtest parameters (TPE, Grid Search, PSO, Genetic Algorithms, etc.), stop-loss/take-profit, and trade volume to maximize strategy profitability.
8. Thoroughly describe the selected optimal strategy, including variables used and a brief description of ML models and results.
9. Implement the optimal strategy with the test dataset and compare against a passive strategy.
10. Present results and conclusions in a Jupyter notebook, focusing on essential visualizations such as candlestick charts, indicators, trading signals, cash/portfolio value over time, and any other relevant insights.