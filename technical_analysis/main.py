from multiprocessing import Pool
import optuna

from technical_analysis.optimization import optimize_file

optuna.logging.set_verbosity(optuna.logging.WARNING)


if __name__ == '__main__':
    with Pool(4) as p:
        res = p.map(optimize_file, ["../data/aapl_1m_train.csv",
                                    "../data/aapl_5m_train.csv",
                                    "../data/aapl_1h_train.csv",
                                    "../data/aapl_1d_train.csv"])
        print(res)