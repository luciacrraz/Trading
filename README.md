# Trading Strategy Optimization Project

## Introduction

This project aims to develop and evaluate algorithmic trading strategies using two complementary approaches: technical analysis (TA) and machine learning (ML). By leveraging these methodologies, we seek to devise profitable strategies across various timeframes (1-day, 1-hour, 5-minute, and 1-minute) utilizing the provided AAPL datasets.

## Project Structure

The project follows a structured Python directory layout:

- **LICENSE**: Defines the licensing terms for the project.
- **README.md**: Offers an overview of the project, its structure, and instructions for running it.
- **Report_ML.ipynb**: Jupyter notebook presenting results and conclusions from machine learning experiments.
- **Report_TA.ipynb**: Jupyter notebook presenting results and conclusions from technical analysis experiments.
- **data**: Directory containing training and testing datasets for AAPL in different timeframes (CSV format).
  - **machine_learning**:
    - **ML_Analysis.ipynb**: Jupyter notebook for machine learning analysis, including model training and hyperparameter tuning.
    - **main.py**: Main script for machine learning workflows.
  - **technical_analysis**:
    - **backtest.py**: Script for backtesting trading strategies.
    - **main.py**: Main script for technical analysis workflows.
    - **optimization.py**: Script for optimizing technical indicator parameters and backtest configurations.
    - **signals.py**: Script for defining buy/sell signals based on technical indicators.
  - **utils**:
    - **utils.py**: Contains utility functions used throughout the project.

## Deliverables

### Delivery 1: Technical Analysis (TA)

#### Project Structure

Follows the aforementioned structure.

#### Data Preparation

- Utilizes provided training and validation datasets (AAPL_1d, AAPL_1h, etc.).

#### Technical Indicator Selection

- Team members select various technical indicators (e.g., MACD, RSI) for analysis.

#### Strategy Backtesting

For each dataset timeframe:
- Defines buy/sell signals based on selected technical indicators.
- Generates all possible indicator combinations (2^n - 1).
- Backtests these strategies, tracking operations, cash flow, and portfolio value.
- Optimizes indicator parameters, stop-loss/take-profit levels, and trade volumes using algorithms like TPE and Grid Search.
- Documents the optimal strategy thoroughly.
- Compares the optimal strategy with a passive buy-and-hold strategy on the testing dataset.

#### Results Presentation

- Utilizes Jupyter notebook Report_TA.ipynb to present results and conclusions.
- Includes:
  - List of operations
  - Candlestick charts with indicators
  - Trading signals
  - Cash/portfolio value over time
  - Any other relevant charts or insights

### Delivery 2: Machine Learning (ML)

#### Project Structure

Follows the aforementioned structure.

#### Data Preparation

- Utilizes provided training and validation datasets (AAPL_1d, AAPL_1h, etc.).
- May incorporate additional technical indicators into the data.

#### Model Selection

- Conducts experiments with Logistic Regression, Support Vector Machine (SVM), and XGBoost models.

#### Model Training and Evaluation

- Splits the training data into training and validation sets.
- Defines independent (features) and dependent (target) variables for model training.
  - Constructs target variable based on next k-period price changes.
- Fine-tunes hyperparameters for each model.
- Evaluates model performance using appropriate metrics.

#### Strategy Backtesting

For each dataset timeframe:
- Utilizes model predictions to generate buy/sell signals.
- Generates all possible combinations of ML models (2^n - 1).
- Backtests these strategies, tracking operations, cash flow, and portfolio value.
