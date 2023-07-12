import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime


def cholesky_decomp(matrix):
    """
    Helper function to perform the cholesky decomposition of a positive definite, symmetric matrix
    :param matrix: Input Matrix
    :return: Decomposed Lower Triangular Matrix (Note L * L.transpose() = A)
    """

    cholesky = np.zeros(shape=matrix.shape)

    n = matrix.shape[0]
    cholesky[0, 0] = matrix[0, 0] ** (1/2)

    for j in range(1, n):
        cholesky[j, 0] = matrix[j, 0] / cholesky[0, 0]

    for i in range(1, n):
        for j in range(i, n):
            if i == j:
                cholesky[i, i] = (matrix[i, i] - sum(cholesky[i, :] ** 2)) ** (1 / 2)
            else:
                sigma = 0
                for p in range(0, i):
                    sigma += cholesky[i, p] * cholesky[j, p]
                cholesky[j, i] = (matrix[j, i] - sigma) / cholesky[i, i]

    return cholesky


class Position:
    def __init__(self, portfolio_df):
        """
        :param portfolio_df: Pandas DataFrame of Asset weights indexed by stock ticker
        """

        self._port_df = portfolio_df.sort_index()
        self._tickers = list(portfolio_df.index)

    def historical_var(self, start, end, thresh):
        """
        :param start: String Date (%Y-%m-%d) for start of validation period
        :param end: String Date (%Y-%m-%d) for end of validation period
        :param thresh: Int (0-100), confidence threshold for VaR. (99 = 99% prob that losses do not exceed result)
        :return: Value at Risk for the portfolio, by the testing period and threshold
        """
        prices = yf.download(self._tickers, datetime.strptime(start, '%Y-%m-%d'),
                             datetime.strptime(end, '%Y-%m-%d'))['Adj Close']

        returns = prices.pct_change().dropna()

        port_returns = returns.multiply(self._port_df, axis=1).sum(axis=1)
        sorted_rets = port_returns.sort_values()
        VaR = sorted_rets.iloc[round(len(sorted_rets.index) * (100-thresh)/100)-1]

        return round(-VaR, 4)

    def monte_carlo_var(self, thresh, n_sims, start=None, end=None, cov_mtrx=None):
        """
        :param thresh: Int (0-100), confidence threshold for VaR. (99 = 99% prob that losses do not exceed result)
        :param n_sims: Int, number of MonteCarlo simulations to run
        :param start: Optional String Date (%Y-%m-%d) for start of period to use historical covariance matrix
        :param end: Optional String Date (%Y-%m-%d) for end of period to use historical covariance matrix
        :param cov_mtrx: Optional covariance matrix input if not estimated from historical returns from Yahoo
        :return:
        """

        if cov_mtrx is None:
            if start is None or end is None:
                raise ValueError('Must supply either a covariance matrix or Start/End dates to estimate one from')

        if cov_mtrx:
            pass
        else:
            prices = yf.download(self._tickers, start=datetime.strptime(start, '%Y-%m-%d'),
                                 end=datetime.strptime(end, '%Y-%m-%d'))['Adj Close']

            returns = prices.pct_change().dropna()

            means = returns.mean().to_numpy()
            cov_mtrx = returns.cov().to_numpy()

            rands = np.random.normal(size=(n_sims, len(self._tickers)))
            log = []
            for trial in rands:
                sim_asset_returns = np.matmul(cov_mtrx, trial) + means
                sim_port_return = (self._port_df.to_numpy() * sim_asset_returns).sum()
                log.append(sim_port_return)

        sorted_rets = pd.DataFrame(log)[0].sort_values()
        print(sorted_rets)
        VaR = sorted_rets.iloc[round(len(sorted_rets.index) * (100 - thresh) / 100) - 1]

        return VaR


if __name__ == '__main__':
    test = pd.DataFrame([0.1, 0.3, 0.35, 0.05, 0.2], index=['AAPL', 'XOM', 'VZ', 'TSLA', 'MMM'])[0]

    test_pos = Position(test)
    #print(test_pos.historical_var('2021-01-01', '2022-01-01', 95))

    print(test_pos.monte_carlo_var(95, 10000, start='2021-01-01', end='2022-01-01'))



