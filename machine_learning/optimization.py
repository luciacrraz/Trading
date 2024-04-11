import pandas as pd
import ta
import optuna
from itertools import combinations, chain
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

from signals import buy_signals, sell_signals
from backtest import backtest
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


def objective_log_regresor(trial, data):
    # Dividir los datos en conjuntos de entrenamiento y prueba
    X = data.iloc[:, :-1]
    # Selecciona la variable objetivo
    y = data.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Definir los parámetros a optimizar
    penalty = trial.suggest_categorical('penalty', ['l1', 'l2'])
    C = trial.suggest_loguniform('C', 0.001, 1000)
    solver = trial.suggest_categorical('solver', ['liblinear', 'saga'])

    # Crear el modelo de regresión logística con los parámetros sugeridos
    model = LogisticRegression(penalty=penalty, C=C, solver=solver, max_iter=10_000, random_state=123)
    # Entrenar el modelo
    model.fit(X_train, y_train)
    # Calcular la precisión en el conjunto de prueba
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

def objective_svm(trial, data):
    # Dividir los datos en conjuntos de entrenamiento y prueba
    X = data.iloc[:, :-1]
    # Selecciona la variable objetivo
    y = data.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Definir los parámetros a optimizar
    C = trial.suggest_loguniform('C', 0.001, 1000)
    kernel = trial.suggest_categorical('kernel', ['linear', 'poly', 'rbf', 'sigmoid'])
    if kernel == 'poly':
        degree = trial.suggest_int('degree', 2, 5)
    else:
        degree = 3  # Valor predeterminado si el kernel no es 'poly'
    gamma = trial.suggest_categorical('gamma', ['scale', 'auto']) if kernel in ['rbf', 'poly', 'sigmoid'] else 'scale'
    # Crear el modelo SVM con los parámetros sugeridos
    model = SVC(C=C, kernel=kernel, degree=degree, gamma=gamma, max_iter=10_000, random_state=123)
    # Entrenar el modelo
    model.fit(X_train, y_train)
    # Calcular la precisión en el conjunto de prueba
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

def objective_xgboost(trial, data):
    data = data.copy()
    # Dividir los datos en conjuntos de entrenamiento y prueba
    X = data.iloc[:, :-1]
    # Selecciona la variable objetivo
    y = data.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    # Definir los parámetros a optimizar
    n_estimators = trial.suggest_int('n_estimators', 100, 1000, step=100)
    max_depth = trial.suggest_int('max_depth', 3, 10)
    learning_rate = trial.suggest_loguniform('learning_rate', 0.01, 0.5)
    subsample = trial.suggest_discrete_uniform('subsample', 0.5, 1.0, 0.1)
    colsample_bytree = trial.suggest_discrete_uniform('colsample_bytree', 0.5, 1.0, 0.1)
    # Crear el modelo XGBoost con los parámetros sugeridos
    model = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        random_state=123,
        max_iter=10_000
    )
    # Entrenar el modelo
    model.fit(X_train, y_train)
    # Calcular la precisión en el conjunto de prueba
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy


def optimize_params_log_regresor(data):
    # Crear un estudio Optuna para la optimización
    study = optuna.create_study(direction='maximize')

    # Función objetivo con el dataset como parámetro fijo
    objective_fn = lambda trial: objective_log_regresor(trial, data)

    # Ejecutar la optimización
    study.optimize(objective_fn, n_trials=20)

    # Obtener los mejores parámetros
    best_params = study.best_params
    best_accuracy = study.best_value

    return best_params, best_accuracy


def optimize_params_svm(data):
    # Crear un estudio Optuna para la optimización
    study = optuna.create_study(direction='maximize')

    # Función objetivo con el dataset como parámetro fijo
    objective_fn = lambda trial: objective_svm(trial, data)

    # Ejecutar la optimización
    study.optimize(objective_fn, n_trials=20)

    # Obtener los mejores parámetros
    best_params = study.best_params
    best_accuracy = study.best_value

    return best_params, best_accuracy


def optimize_params_xgboost(data):
    # Crear un estudio Optuna para la optimización
    study = optuna.create_study(direction='maximize')

    # Función objetivo con el dataset como parámetro fijo
    objective_fn = lambda trial: objective_xgboost(trial, data)

    # Ejecutar la optimización
    study.optimize(objective_fn, n_trials=20)

    # Obtener los mejores parámetros
    best_params = study.best_params
    best_accuracy = study.best_value

    return best_params, best_accuracy

def optimize_params(data):
    # Optimización de regresión logística
    best_params_lr, best_accuracy_lr = optimize_params_log_regresor(data)
    print("Mejores parámetros de regresión logística:", best_params_lr)
    # Optimización de SVM
    best_params_svm, best_accuracy_svm = optimize_params_svm(data)
    print("Mejores parámetros de SVM:", best_params_svm)
    # Optimización de XGBoost
    best_params_xgb, best_accuracy_xgb = optimize_params_xgboost(data)
    print("Mejores parámetros de XGBoost:", best_params_xgb)
    return best_params_svm, best_params_lr, best_params_xgb


def params(data: str):
    data_1d_train = pd.read_csv(data)
    data_1d_train = data_1d_train.dropna()
    dataresult_long_1d_train = file_features(data_1d_train, ds_type="buy")
    dataresult_short_1d_train = file_features(data_1d_train, ds_type="sell")
    best_params_svm_long, best_params_lr_long, best_params_xgb_long = optimize_params(dataresult_long_1d_train)
    best_params_svm_short, best_params_lr_short, best_params_xgb_short = optimize_params(dataresult_short_1d_train)

    return best_params_lr_long, best_params_svm_long, best_params_xgb_long, best_params_lr_short, best_params_svm_short, best_params_xgb_short

logistic_params_1d_long, svm_params_1d_long, xgboost_params_1d_long, logistic_params_1d_short, svm_params_1d_short, xgboost_params_1d_short = params("../data/aapl_1d_train.csv")
logistic_params_1h_long, svm_params_1h_long, xgboost_params_1h_long, logistic_params_1h_short, svm_params_1h_short, xgboost_params_1h_short = params("../data/aapl_1h_train.csv")
logistic_params_1m_long, svm_params_1m_long, xgboost_params_1m_long, logistic_params_1m_short, svm_params_1m_short, xgboost_params_1m_short = params("../data/aapl_1m_train.csv")
logistic_params_5m_long, svm_params_5m_long, xgboost_params_5m_long, logistic_params_5m_short, svm_params_5m_short, xgboost_params_5m_short = params("../data/aapl_5m_train.csv")

def buy_signals(data, logistic_params, svm_params, xgboost_params):
    buy_signals = pd.DataFrame()
    # Selecciona las características
    X = data.iloc[:, :-1]
    # Selecciona la variable objetivo
    y = data.iloc[:, -1]

    # Crear modelos con los parámetros proporcionados
    logistic_model = LogisticRegression(**logistic_params)
    svm_model = SVC(**svm_params)
    xgboost_model = XGBClassifier(**xgboost_params)

    # Entrenar los modelos con todo el conjunto de datos original
    logistic_model.fit(X, y)
    svm_model.fit(X, y)
    xgboost_model.fit(X, y)

    # Realizar predicciones en el conjunto de datos original
    predictions_lr = logistic_model.predict(X)
    predictions_svm = svm_model.predict(X)
    predictions_xgboost = xgboost_model.predict(X)
    predictions_xgboost_bool = predictions_xgboost.astype(bool)

    # Agregar las predicciones como nuevas columnas al conjunto de datos original
    buy_signals['predicciones_lr'] = predictions_lr
    buy_signals['predicciones_svm'] = predictions_svm
    buy_signals['predicciones_xgboost'] = predictions_xgboost_bool

    return buy_signals

def sell_signals(data, logistic_params, svm_params, xgboost_params):
    sell_signals = pd.DataFrame()
    # Selecciona las características
    X = data.iloc[:, :-1]
    # Selecciona la variable objetivo
    y = data.iloc[:, -1]

    # Crear modelos con los parámetros proporcionados
    logistic_model = LogisticRegression(**logistic_params)
    svm_model = SVC(**svm_params)
    xgboost_model = XGBClassifier(**xgboost_params)

    # Entrenar los modelos con todo el conjunto de datos original
    logistic_model.fit(X, y)
    svm_model.fit(X, y)
    xgboost_model.fit(X, y)

    # Realizar predicciones en el conjunto de datos original
    predictions_lr = logistic_model.predict(X)
    predictions_svm = svm_model.predict(X)
    predictions_xgboost = xgboost_model.predict(X)
    predictions_xgboost_bool = predictions_xgboost.astype(bool)

    # Agregar las predicciones como nuevas columnas al conjunto de datos original
    sell_signals['predicciones_lr'] = predictions_lr
    sell_signals['predicciones_svm'] = predictions_svm
    sell_signals['predicciones_xgboost'] = predictions_xgboost_bool

    return sell_signals




def optimize(trial, strategy, data):
    portfolio_value = 0

    stop_loss = trial.suggest_float("stop_loss", 0.80, 0.90)
    take_profit = trial.suggest_float("take_profit", 1.01, 1.10)
    n_shares = trial.suggest_int("n_shares", 20, 50)

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
        study.optimize(lambda x: optimize(x, strat, data), n_trials=20)
        value = study.best_value
        print(f"Best value found so far: {value}")
        if value > best_val:
            best_val = value
            best_strat = strat
            best_params = study.best_params
            print(best_strat)
            print(best_params)

    return {"file": data,
            "strat": best_strat,
            "value": best_val,
            "params": best_params}