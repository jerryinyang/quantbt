import pandas as pd
import numpy as np 
from concurrent.futures import ProcessPoolExecutor, as_completed

from backtester import Backtester
from reporters import AutoReporter

from engine import Engine
from copy import deepcopy, copy # noqa
from tqdm import tqdm


class Analyser:

    def __init__(self, backtester:Backtester) -> None:
        '''
        This class embodies the analyst performing series of backtests.
        '''
        # Assert the backtester.alphas is not empty
        # TODO : Assert backtester is ready for backtesting
        assert backtester.alphas, ValueError('Backtester must have at least one alpha to be analyzed.')

        self.backtester = backtester


    def analyse_robustness_price(self, reporter : AutoReporter, iterations:int, synthesis_mode : float | str='perturb', noise_factor : float = 1):
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
            synthesizer = self.synthesize_price_perturb

        # Create multiple processes for the backtests
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(self._iter_robustness_price, synthesizer, noise_factor) for _ in range(iterations+1)]
            kwargs = {
                'ascii' : "░▒█" ,
                'total': len(futures),
                'unit': 'it',
                'unit_scale': True,
                'leave': True # Leave the Bar in the terminal when completed
            }
            # using tqdm to track progress
            [reporter.compute_report(future.result()) for future in tqdm(as_completed(futures), **kwargs)]

            # Run strategy with Original Backtester
            self.backtester.id = 'original'
            self.backtester.backtest(analysis_mode=True)
            reporter.compute_report(self.backtester)

        print('Analysis Completed.')
    

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

        # Modify backtester.engine.dataframes 
        dataframes = deepcopy(self.backtester.original_dataframes)
        
        # Synthesize new data
        for ticker, ticker_data in dataframes.items():
            new_data = synthesizer(ticker_data, noise_factor)
            dataframes[ticker] = new_data

        # Get a copy of self.backtester, unique for each iteration
        backtester = copy(self.backtester)

        # Reset backtester.engine with new dataframes
        backtester.reset_backtester(dataframes)

        # Run backtest
        backtester.backtest(analysis_mode=True)

        return backtester
    

    # PICKLE-COMPATIBILITY
    def __getstate__(self):
        state = self.__dict__.copy()
        return state


    def __setstate__(self, state):
        # Customize the object reconstruction
        self.__dict__.update(state)



if __name__ == '__main__':
    import yfinance as yf
    import pandas as pd
    import os
    import pickle
    
    from alpha import BaseAlpha
    from dataloader import DataLoader
    from utils import clear_terminal

    start_date = '2020-01-02'
    end_date = '2023-12-31'

    clear_terminal()
    with open('logs.log', 'w'):
        pass

    tickers = ['AAPL'] # , 'GOOG', 'TSLA']
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

    # Create DataHandler
    dataloader = DataLoader(dataframes, '1d', start_date, end_date)
    engine = Engine(dataloader)
    alpha = BaseAlpha('base_alpha', engine, .1, .05)

    backtester = Backtester(dataloader, engine, alpha, 1 )

    reporter = AutoReporter(returns_mode='full')

    analyser = Analyser(backtester)
    data = analyser.analyse_robustness_price(reporter, 50, 'perturb', 0.01)

    # Pickle the instance
    with open('1000.pkl', 'wb') as file:
        pickle.dump(reporter, file)