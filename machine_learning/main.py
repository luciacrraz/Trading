import pandas as pd

from optimization import *


if __name__ == '__main__':
    data_1d_test = pd.read_csv("../data/aapl_1d_test.csv")
    data_1d_test = data_1d_test.dropna()
    dataresult_long_1d_test = file_features(data_1d_test, ds_type="buy")
    dataresult_short_1d_test = file_features(data_1d_test, ds_type="sell")
    global global_buy_signals
    global global_sell_signals
    global_buy_signals = buy_signals(dataresult_long_1d_test, logistic_params_1d_long, svm_params_1d_long, xgboost_params_1d_long)
    global_sell_signals = sell_signals(dataresult_short_1d_test, logistic_params_1d_short, svm_params_1d_short, xgboost_params_1d_short)
    file_1d_test = optimize_file(data_1d_test)

    data_1m_test = pd.read_csv("../data/aapl_1m_train.csv")
    data_1m_test = data_1m_test.dropna()
    dataresult_long_1m_test = file_features(data_1m_test, ds_type="buy")
    dataresult_short_1m_test = file_features(data_1m_test, ds_type="sell")
    global_buy_signals = buy_signals(dataresult_long_1m_test, logistic_params_1m_long, svm_params_1m_long, xgboost_params_1m_long)
    global_sell_signals = sell_signals(dataresult_short_1m_test, logistic_params_1m_short, svm_params_1m_short, xgboost_params_1m_short)
    file_1m_test = optimize_file(data_1m_test)

    data_1h_test = pd.read_csv("../data/aapl_1h_test.csv")
    data_1h_test = data_1h_test.dropna()
    dataresult_long_1h_test = file_features(data_1h_test, ds_type="buy")
    dataresult_short_1h_test = file_features(data_1h_test, ds_type="sell")
    global_buy_signals = buy_signals(dataresult_long_1h_test, logistic_params_1h_long, svm_params_1h_long, xgboost_params_1h_long)
    global_sell_signals = sell_signals(dataresult_short_1h_test, logistic_params_1h_short, svm_params_1h_short, xgboost_params_1h_short)
    file_1h_test = optimize_file(data_1h_test)

    data_5m_test = pd.read_csv("../data/aapl_5m_train.csv")
    data_5m_test = data_5m_test.dropna()
    dataresult_long_5m_test = file_features(data_5m_test, ds_type="buy")
    dataresult_short_5m_test = file_features(data_5m_test, ds_type="sell")
    global_buy_signals = buy_signals(dataresult_long_5m_test, logistic_params_5m_long, svm_params_5m_long, xgboost_params_5m_long)
    global_sell_signals = sell_signals(dataresult_short_5m_test, logistic_params_5m_short, svm_params_5m_short, xgboost_params_5m_short)
    file_5m_test = optimize_file(data_5m_test)