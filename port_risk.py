import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime


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


if __name__ == '__main__':
    test = pd.DataFrame([0.1, 0.3, 0.35, 0.05, 0.2], index=['AAPL', 'XOM', 'VZ', 'TSLA', 'MMM'])[0]

    test_pos = Position(test)
    print(test_pos.historical_var('2021-01-01', '2022-01-01', 95))



