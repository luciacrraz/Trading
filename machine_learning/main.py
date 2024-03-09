import pandas as pd
import ta
import optuna
import time
import numpy as np
from multiprocessing import Pool
from itertools import combinations, chain
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")

data_1m_test = pd.read_csv("../data/aapl_1m_test.csv")
data_1m_test = data_1m_test.dropna()

def powerset(s):
    return chain.from_iterable(combinations(s,r) for r in range(1,len(s)+1))


def file_features(data, ds_type: str):
    data1 = pd.DataFrame()
    # Calcular indicadores tecnicos
    cmf_data = ta.volume.ChaikinMoneyFlowIndicator(data.High, data.Low, data.Close, data.Volume, window=14)
    rsi_data = ta.momentum.RSIIndicator(data.Close, window=14)

    data1["CMF"] = cmf_data.chaikin_money_flow()
    data1["RSI"] = rsi_data.rsi()
    # Calcular la volatilidad
    data1['Volatility'] = data['High'] - data['Low']
    data1['Close_Lag0'] = data['Close']
    # Calcular las tendencias
    for i in range(1, 5 + 1):
        data1[f'Close_Lag{i}'] = data['Close'].shift(i)
    # Variable ded respuesta
    if ds_type == "buy":
        data1['Response'] = (data['Close'] < data['Close'].shift(-10))
    else:
        data1['Response'] = (data['Close'] > data['Close'].shift(-10))

    data1 = data1.drop(data1.index[:30])
    data1 = data1.drop(data1.index[-30:])
    data1.reset_index(drop=True, inplace=True)

    return data1

dataresult_long_1m_test = file_features(data_1m_test, ds_type="buy")
dataresult_short_1m_test = file_features(data_1m_test, ds_type="sell")

def buy_signals(data):
    buy_signals = pd.DataFrame()
    # Selecciona las características
    X = data.iloc[:, :-1]
    # Selecciona la variable objetivo
    y = data.iloc[:, -1]

    # Crear modelos con los mejores parámetros encontrados para cada algoritmo
    best_logistic_model = LogisticRegression(penalty='l2', C=0.010405629107406833, solver='liblinear')
    best_svm_model = SVC(C=0.02891729617292502, kernel='linear', gamma='scale')
    best_xgboost_model = XGBClassifier(n_estimators=300, max_depth=9, learning_rate=0.02572836520302602, subsample=0.8,
                                       colsample_bytree=1.0)

    # Entrenar los modelos con todo el conjunto de datos original
    best_logistic_model.fit(X, y)
    best_svm_model.fit(X, y)
    best_xgboost_model.fit(X, y)

    # Realizar predicciones en el conjunto de datos original
    predictions_lr = best_logistic_model.predict(X)
    predictions_svm = best_svm_model.predict(X)
    predictions_xgboost = best_xgboost_model.predict(X)

    # Agregar las predicciones como nuevas columnas al conjunto de datos original
    buy_signals['predicciones_lr'] = predictions_lr
    buy_signals['predicciones_svm'] = predictions_svm
    buy_signals['predicciones_xgboost'] = predictions_xgboost

    return buy_signals

def sell_signals(data):
    sell_signals = pd.DataFrame()
    # Selecciona las características
    X = data.iloc[:, :-1]
    # Selecciona la variable objetivo
    y = data.iloc[:, -1]

    # Crear modelos con los mejores parámetros encontrados para cada algoritmo
    best_logistic_model = LogisticRegression(penalty='l1', C=0.002778605059659017, solver='saga')
    best_svm_model = SVC(C=0.28120622947860485, kernel='rbf', gamma='auto')
    best_xgboost_model = XGBClassifier(n_estimators=400, max_depth=3, learning_rate=0.11719371112820326, subsample=0.7,
                                       colsample_bytree=0.8)

    # Entrenar los modelos con todo el conjunto de datos original
    best_logistic_model.fit(X, y)
    best_svm_model.fit(X, y)
    best_xgboost_model.fit(X, y)

    # Realizar predicciones en el conjunto de datos original
    predictions_lr = best_logistic_model.predict(X)
    predictions_svm = best_svm_model.predict(X)
    predictions_xgboost = best_xgboost_model.predict(X)
    predictions_xgboost_bool = predictions_xgboost.astype(bool)

    # Agregar las predicciones como nuevas columnas al conjunto de datos original
    sell_signals['predicciones_lr'] = predictions_lr
    sell_signals['predicciones_svm'] = predictions_svm
    sell_signals['predicciones_xgboost'] = predictions_xgboost_bool

    return sell_signals

global_buy_signals = buy_signals(dataresult_long_1m_test)
global_sell_signals = sell_signals(dataresult_short_1m_test)


def backtest(data, buy_signals, sell_signals, stop_loss, take_profit, n_shares):
    history = []
    active_operations = []
    cash = 1_000_000
    com = 1.25 / 100

    for i, row in data.iterrows():
        # close active operation
        active_op_temp = []
        for operation in active_operations:
            if operation["stop_loss"] > row.Close:
                cash += (row.Close * operation["n_shares"]) * (1 - com)
            elif operation["take_profit"] < row.Close:
                cash += (row.Close * operation["n_shares"]) * (1 - com)
            else:
                active_op_temp.append(operation)
        active_operations = active_op_temp

        # check if we have enough cash
        if cash < (row.Close * (1 + com)):
            asset_vals = sum([operation["n_shares"] * row.Close for operation in active_operations])
            portfolio_value = cash + asset_vals
            continue

        # Apply buy signals
        if buy_signals.loc[i].any():
            active_operations.append({
                "bought": row.Close,
                "n_shares": n_shares,
                "stop_loss": row.Close * stop_loss,
                "take_profit": row.Close * take_profit
            })

            cash -= row.Close * (1 + com) * n_shares

        # Apply sell signals
        if sell_signals.loc[i].any():
            active_op_temp = []
            for operation in active_operations:
                if operation["take_profit"] < row.Close or operation["stop_loss"] > row.Close:
                    cash += (row.Close * operation["n_shares"]) * (1 - com)
                else:
                    active_op_temp.append(operation)
            active_operations = active_op_temp

        asset_vals = sum([operation["n_shares"] * row.Close for operation in active_operations])
        portfolio_value = cash + asset_vals

    return portfolio_value


def optimize(trial, strategy, data):
    portfolio_value = 0

    stop_loss = trial.suggest_float("stop_loss", 0.00250, 0.05)
    take_profit = trial.suggest_float("take_profit", 0.00250, 0.05)
    n_shares = trial.suggest_int("n_shares", 5, 200)

    strat_params = {}

    buy_signals = pd.DataFrame()
    sell_signals = pd.DataFrame()

    if "logistic" in strategy:
        buy_signals["logistic"] = global_buy_signals["predicciones_lr"]
        sell_signals["logistic"] = global_sell_signals["predicciones_lr"]

    if "svm" in strategy:
        buy_signals["svm"] = global_buy_signals["predicciones_svm"]
        sell_signals["svm"] = global_sell_signals["predicciones_svm"]

    if "xg" in strategy:
        buy_signals["xg"] = global_buy_signals["predicciones_xgboost"]
        sell_signals["xg"] = global_sell_signals["predicciones_xgboost"]

    return backtest(data, buy_signals, sell_signals, stop_loss, take_profit, n_shares)

def optimize_file(data):
    data = data.drop(data.index[:30])
    data = data.drop(data.index[-30:])
    data.reset_index(drop=True, inplace=True)
    strategies = list(powerset(["logistic", "svm", "xg"]))
    best_strat = None
    best_val = -1
    best_params = None

    for strat in strategies:
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda x: optimize(x, strat, data), n_trials=30)
        value = study.best_value
        if value > best_val:
            best_val = value
            best_strat = strat
            best_params = study.best_params
    print(study.best_value)
    print(best_strat)
    print(best_params)

    return {"file": data,
            "strat": best_strat,
            "value": best_val,
            "params": best_params}

file_1m_test = optimize_file(data_1m_test)
print(file_1m_test)

