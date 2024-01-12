import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import copy, deepcopy  # noqa
from typing import Union

import pandas as pd
import synthesizers as synth
from backtester import Backtester
from engine import Engine
from reporters import AutoReporter
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


    # ROBUSTNESS TO PRICE DATA
    def analyse_robustness_price(self, reporter : AutoReporter, iterations:int, synthesis_mode : float | str='perturb', **args):
        '''
        For this analysis, synthetic OHLCV data is created by modifying the actual backtest data. 
        Then, the backtest is run `n` times, and the performance metrics are recalculated and compared.
        '''

        # Pick the synthesizer
        synthesis_mode = str(synthesis_mode)

        if str(synthesis_mode) in ['noise', 'perturb']:
            synthesizer = synth.Noise(**args)

        elif str(synthesis_mode) in ['new', 'forecast', 'gbm']:
            synthesizer = synth.GBM(**args)
        else:
            synthesizer = synth.Noise(**args)

        # Create multiple processes for the backtests
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(self._iter_price_data, synthesizer) for _ in range(iterations+1)]
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

        # Return Dataframe of all backtests and their reports
        return reporter.report()
    

    def _iter_price_data(self, synthesizer : Union[synth.GBM, synth.Noise]):

        # Modify backtester.engine.dataframes 
        dataframes = deepcopy(self.backtester.original_dataframes)
        
        # Synthesize new data
        for ticker, ticker_data in dataframes.items():
            new_data = synthesizer.synthesize(ticker_data)
            dataframes[ticker] = new_data

        # Get a copy of self.backtester, unique for each iteration
        backtester = self.backtester.copy()

        # Reset backtester.engine with new dataframes
        backtester.reset_backtester(dataframes)

        # Run backtest
        backtester.backtest(analysis_mode=True)

        return backtester
    

    # ROUBUSTNESS TO TRADE ORDER
    def analyse_robustness_trade_order(self, reporter : AutoReporter, iterations:int):
        '''
        For this analysis, synthetic OHLCV data is created by modifying the actual backtest data. 
        Then, the backtest is run `n` times, and the performance metrics are recalculated and compared.
        '''

        # Run Initial Backtest
        self.backtester.id = 'original'
        self.backtester.backtest(analysis_mode=True)

        # Create multiple processes for the backtests
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(self._iter_trade_order, self.backtester) for _ in range(iterations+1)]
            kwargs = {
                'ascii' : "░▒█" ,
                'total': len(futures),
                'unit': 'it',
                'unit_scale': True,
                'leave': True # Leave the Bar in the terminal when completed
            }

            # using tqdm to track progress
            [reporter.compute_report(future.result(), analysis_shuffle=True) for future in tqdm(as_completed(futures), **kwargs)]

        print('Analysis Completed.')
        return reporter.report()


    def _iter_trade_order(self, _backtester : Backtester):
        # Get a copy of the passed backtester, unique for each iteration
        backtester = _backtester.copy()

        # Make a copy of self.backtester.engine.history
        history = (backtester.engine.history)

        # Shuffle/Permute the List 
        random.shuffle(history) # Shuffle the list

        # Reassign the history
        backtester.engine.history = history

        return backtester


    # PICKLE-COMPATIBILITY
    def __getstate__(self):
        state = self.__dict__.copy()
        return state


    def __setstate__(self, state):
        # Customize the object reconstruction
        self.__dict__.update(state)



if __name__ == '__main__':
    import os
    import pickle

    import yfinance as yf
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

    backtester = Backtester(dataloader, engine, alpha, 1)
    reporter = AutoReporter(earnings_mode='trades')
    analyser = Analyser(backtester)
    
    # Get reports
    earnings, metrics = analyser.analyse_robustness_trade_order(reporter, 20)
    

    # Pickle the instance
    with open('earnings.pkl', 'wb') as file:
        pickle.dump(earnings, file)