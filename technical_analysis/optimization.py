import pandas as pd
import optuna
from itertools import combinations, chain

from technical_analysis.signals import stoch_signals, tsi_signals, roc_signals, rsi_signals


class Position:
    def __init__(self, timestamp, order_type, n_shares, stop_at, take_at, bought_at):
        self.timestamp = timestamp
        self.order_type = order_type
        self.n_shares = n_shares
        self.stop_at = stop_at
        self.take_at = take_at
        self.bought_at = bought_at


def powerset(s):
    return chain.from_iterable(combinations(s, r) for r in range(1, len(s) + 1))


def optimize(trial, strategy, data):
    portfolio_value = 0

    stop_loss = trial.suggest_float("stop_loss", 0.00250, 0.05)
    take_profit = trial.suggest_float("take_profit", 0.00250, 0.05)
    n_shares = trial.suggest_int("n_shares", 5, 200)

    strat_params = {}

    buy_signals = pd.DataFrame()
    sell_signals = pd.DataFrame()

    if "rsi" in strategy:
        strat_params['rsi'] = {
            "rsi_window": trial.suggest_int("rsi_window", 5, 100),
            "rsi_upper": trial.suggest_float("rsi_upper", 65, 95),
            "rsi_lower": trial.suggest_float("rsi_lower", 5, 35)

        }
        rsi_buy, rsi_sell = rsi_signals(data, **strat_params["rsi"])
        buy_signals["rsi"] = rsi_buy
        sell_signals["rsi"] = rsi_sell

    if "roc" in strategy:
        strat_params['roc'] = {
            "roc_window": trial.suggest_int("roc_window", 5, 100),
            "roc_upper": trial.suggest_float("roc_upper", 0.8, 1.5),
            "roc_lower": trial.suggest_float("roc_lower", -2, -1)

        }
        roc_buy, roc_sell = roc_signals(data, **strat_params["roc"])
        buy_signals["roc"] = roc_buy
        sell_signals["roc"] = roc_sell

    if "tsi" in strategy:
        strat_params['tsi'] = {
            "tsi_window_slow": trial.suggest_int("tsi_window_slow", 5, 20),
            "tsi_window_fast": trial.suggest_int("tsi_window_fast", 20, 40),

            "tsi_upper": trial.suggest_float("tsi_upper", 25, 45),
            "tsi_lower": trial.suggest_float("tsi_lower", -40, -20)

        }
        tsi_buy, tsi_sell = tsi_signals(data, **strat_params["tsi"])
        buy_signals["tsi"] = tsi_buy
        sell_signals["tsi"] = tsi_sell

    if "stoch" in strategy:
        strat_params['stoch'] = {
            "stoch_window": trial.suggest_int("stoch_window", 5, 21),
            "stoch_smooth_window": trial.suggest_int("stoch_smooth_window", 3, 10),
            "stoch_upper": trial.suggest_float("stoch_upper", 70, 90),
            "stoch_lower": trial.suggest_float("stoch_lower", 10, 30)

        }
        stoch_buy, stoch_sell = stoch_signals(data, **strat_params["stoch"])
        buy_signals["stoch"] = stoch_buy
        sell_signals["stoch"] = stoch_sell
    # print(buy_signals)
    # print(sell_signals)

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


def optimize_file(file_path: str):
    data = pd.read_csv(file_path)
    data = data.dropna()
    strategies = list(powerset(["rsi", "roc", "tsi", "stoch"]))
    best_strat = None
    best_val = -1
    best_params = None

    for strat in strategies:
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda x: optimize(x, strat, data), n_trials=50)
        value = study.best_value
        if value > best_val:
            best_val = value
            best_strat = strat
            best_params = study.best_params
    print(study.best_value)
    print(best_strat)
    print(best_params)

    return {"file": file_path,
            "strat": best_strat,
            "value": best_val,
            "params": best_params}
