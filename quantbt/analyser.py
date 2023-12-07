import random
import pandas as pd
import numpy as np # noqa: F401
import matplotlib.pyplot as plt # noqa: F401
import concurrent.futures # noqa: F401
from typing import Dict, List # noqa: F401

from backtester import Backtester
from utils import Logger 
from utils import clear_terminal, debug, sorted_index  # noqa: F401
from reporters import Reporter # noqa: F401
import os

from engine import Engine
from alpha import BaseAlpha
from copy import deepcopy



class Analyser:
    def __init__(self, backtester:Backtester) -> None:
        '''
        This class embodies the analyst performing series of backtests.
        '''
        # Assert the backtester.alphas is not empty
        # TODO : Assert backtester is ready for backtesting
        assert backtester.alphas, ValueError('Backtester must have at least one alpha to be analyzed.')

        self.backtester = backtester
        self.logger = Logger('analyzer')


    def analyse_robustness_price(self, iterations:int, synthesis_mode : float | str='perturb', noise_factor : float = 1):
        '''
        For this analysis, synthetic OHLCV data is created by modifying the actual backtest data. 
        Then, the backtest is run `n` times, and the performance metrics are recalculated and compared.
        '''

        # Pick the synthesizer
        synthesis_mode = str(synthesis_mode)

        if str(synthesis_mode) in ['noise', 'perturb', '0']:
            synthesizer = self.synthesize_price_perturb
        elif str(synthesis_mode) in ['permute', '1']:
            synthesizer = self.synthesize_price_permute
        else:
            self.logger.warn(f'Invalid `synthesis_mode` value passed ({synthesis_mode}). It would be randomly selected for each iteration.')
            synthesizer = None

        # Store backtest results, contains trade history list for each backtest iteration
        analysis_history = []

        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Submit tasks for each iteration
            futures = [executor.submit(self._iter_robustness_price, synthesizer, noise_factor) for _ in range(iterations)]

            # Wait for all tasks to complete
            concurrent.futures.wait(futures)

            # Retrieve results from completed tasks
            analysis_history = [future.result() for future in futures]
        
        # Sequential execution without concurrency
        # analysis_history = [self._iter_robustness_price(synthesizer, noise_factor) for _ in range(iterations)]

        return analysis_history
    

    def synthesize_price_permute(self, _data:pd.DataFrame, scale_factor : float):
        """
        Generate synthetic price data based on permutations of original data.

        Parameters:
        _data (pd.DataFrame): Input dataframe containing financial data.

        Returns:
        pd.DataFrame: DataFrame with synthetic price data.
        """ 

        # Check if 'close' column is present in the dataframe
        assert 'close' in _data.columns, ValueError("Missing required `close` column in dataframe.")
        
        # Create a copy of the input data
        data = _data.copy()

        # Compute Percentage Changes
        change_close = data['close'].pct_change()
        change_close = change_close.fillna(change_close.mean())

        # Compute the gap between open and previous close
        open_gap = data['open'] - data['close'].shift(1).fillna(data['close'])

        # Compute the difference between high and maximum of open and close
        high_bodyhigh_diff = data['high'] - data[['open', 'close']].max(axis=1)
        
        # Compute the difference between minimum of open and close and low
        low_bodylow_diff = data[['open', 'close']].min(axis=1) - data['low']

        # Permute Percentage Changes and Gaps
        change_close = np.random.permutation(change_close) * scale_factor
        open_gap = np.random.permutation(open_gap) * scale_factor
        high_bodyhigh_diff = np.random.permutation(high_bodyhigh_diff) * scale_factor
        low_bodylow_diff = np.random.permutation(low_bodylow_diff) * scale_factor

        # Create a synthetic dataframe
        synth = pd.DataFrame()

        # Generate synthetic 'close' prices
        synth['close'] = data['close'].shift(1).fillna(data['close']) * (1 + change_close)
        
        # Generate synthetic 'open' prices
        synth['open'] = data['close'].shift(1).fillna(data['close']) + open_gap
        synth['close'] = synth['close'].fillna(synth['open'] * 1.01)
        synth['open'] = synth['open'].fillna(data['close'] + open_gap.mean())

        # Generate synthetic 'high' and 'low' prices
        synth['high'] = synth[['open', 'close']].max(axis=1) + high_bodyhigh_diff
        synth['low'] = synth[['open', 'close']].min(axis=1) - low_bodylow_diff
        

        data[['open', 'high', 'low', 'close']] = synth[['open', 'high', 'low', 'close']]

        # Return selected columns of the synthetic dataframe
        return data
        

    def synthesize_price_perturb(self, _data:pd.DataFrame, noise_factor:float):

        """
        Generate synthetic price data with added noise based on the original data.

        Parameters:
        _data (pd.DataFrame): Input dataframe containing financial data.
        noise_factor (float): Factor to control the amount of noise added to the data.

        Returns:
        pd.DataFrame: DataFrame with synthetic price data including noise.
        """

        # Keep the original columns
        columns = _data.columns

        # Create a copy of the input data
        data = _data.copy()

        # Compute Percentage Changes and Gaps
        change_close = data['close'].pct_change()
        open_gap = data['open'] - data['close'].shift(1).fillna(0)

        # Fill missing values and add normal noise based on the noise factor
        change_close = change_close.fillna(change_close.mean())
        change_close = np.random.normal(
            change_close.mean(),
            change_close.std(),
            len(change_close)
        )
        change_close = self.add_noise(change_close, noise_factor)

        # Generate synthetic 'close' and 'open' prices with added noise
        data['close'] = data['close'] * (1 + change_close)
        data['open'] = data['close'] + open_gap

        # Return the dataframe with synthetic prices including noise
        return data[columns]
    

    def add_noise(self, series, percentage):
        noise = (np.random.random(size=series.shape) - 0.5) * percentage * 2
        return series + series * noise


    def _iter_robustness_price(self, synthesizer, noise_factor):
        # Randomily pick synthesizer if it is not specified
        if not synthesizer:
            synthesizer =  random.choice([self.synthesize_price_permute, self.synthesize_price_perturb])


        # Modify backtester.engine.dataframes 
        dataframes = deepcopy(self.backtester.original_dataframes)
        
        # Synthesize new data
        for ticker, ticker_data in dataframes.items():
            new_data = synthesizer(ticker_data, noise_factor)
            dataframes[ticker] = new_data

        # Reset backtester.engine with new dataframes
        self.backtester.reset_backtester(dataframes)
        
        print(f"Starting Capital : {self.backtester.engine.CAPITAL}")

        # Run backtest
        trade_history = self.backtester.backtest()

        return trade_history
    

if __name__ == '__main__':
    import yfinance as yf
    start_date = '2020-01-02'
    end_date = '2023-12-31'

    clear_terminal()
    with open('logs.log', 'w'):
        pass

    tickers = ['AAPL', 'GOOG', 'TSLA']
    ticker_path = [f'data/prices/{ticker}.csv' for ticker in tickers]

    dfs = []

    for ticker in tickers:
        file_name = f'data/prices/{ticker}.csv'

        if os.path.exists(file_name):
            df = pd.read_csv(file_name, index_col='Date', parse_dates=True)
        else:
            df = yf.download(ticker, start=start_date, end=end_date)
            df.to_csv(file_name)
            
        dfs.append(df)

    dataframes = dict(zip(tickers, dfs))

    engine = Engine(tickers, dataframes, '1d', start_date, end_date)
    # alpha = BaseAlpha(engine, 1, .1, .03)
    # alpha2 = BaseAlpha(engine, 1, .05, .2)
    alpha2 = BaseAlpha(engine, 1, .1, .05)

    backtester = Backtester(engine, alpha2)
    # backtester.add_alpha(alpha)
    # backtester.add_alpha(alpha2)


    # From Here Analyse
    anal = Analyser(backtester)
    anal.analyse_robustness_price(10, 'permute', noise_factor=0.01)

    # dataframes = backtester.engine.dataframes
        
    # for ticker, ticker_data in dataframes.items():
    #     # Synthesize new data
    #     new_data = anal.synthesize_price_perturb(ticker_data, 0.01)
    #     dataframes[ticker] = new_data

    # # Reset backtester.engine.dataframes 
    # backtester.engine.reset(dataframes)

    # trade_history = backtester.backtest()

    # # Use Reported
    # reporter = Reporter(trade_history)

    # metric :pd.DataFrame = reporter.report()['full'][1]

    # metric.to_csv('data/test.csv', index=True)