import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier

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

