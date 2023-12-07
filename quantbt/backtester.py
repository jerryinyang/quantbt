import logging
import pandas as pd
from typing import List

from engine import Engine
from alpha import Alpha
from orders import Order
from portfolio import Portfolio # noqa: F401
from dataloader import DataLoader
from sizers import Sizer
from utils import Bar, debug, Logger # noqa: F401
from copy import deepcopy

logging.basicConfig(filename='logs.log', level=logging.INFO)

exectypes = Order.ExecType

class Backtester:
    _original_dfs : dict[pd.DataFrame] = None

    logger = Logger('logger_backtester')

    def __init__(self, 
                 dataloader : DataLoader, 
                 engine : Engine, 
                 alphas : list[Alpha], 
                 max_allocation : float,
                 analysis_mode : bool = False) -> None:
        
        self.datas = dataloader
        self.engine = engine

        # Add Alphas into backtester alphas list
        self.alphas : List[Alpha] = self._init_alpha(alphas)
        self.max_allocation = max_allocation

        # Add a Sizer
        self.sizer = Sizer(self.engine, self.alphas, max_allocation)

        # If Backtester is being initialized within and Analyzer, don't modify the initial_engine
        if not analysis_mode:
            self._original_dfs = deepcopy(dataloader.dataframes)


    def _init_alpha(self, alphas: Alpha|list[Alpha]):
        if not alphas:
            return []
        
        if isinstance(alphas, list):
            for alpha in alphas:
                alpha.reset_alpha(self.engine)
            return alphas

        else:
            alphas.reset_alpha(self.engine)
            return [alphas]
        
    
    def add_alpha(self, alpha:Alpha):
        alpha_list = self._init_alpha(self.engine)

        for alpha in alpha_list:
            self.alphas.append(alpha)

        # Add alpha to engine observers
        self.engine.add_observer(alpha)

        # Reset Sizer
        self.sizer = Sizer(self.engine, self.alphas)


    def backtest(self, analysis_mode=False):
        if not analysis_mode:
            print('Initiating Backtest')
        
        # Iterate through each bar/index (timestamp) in the backtest range
        for bar_index in self.engine.portfolio.dataframe.index:
            date = self.engine.portfolio.dataframe.loc[bar_index, 'timestamp']

            # Create the bar objects for easier access to data
            bars = {}
            for ticker in self.engine.tickers:

                bar = Bar(
                    open=self.engine.dataframes[ticker].loc[date, 'open'],
                    high=self.engine.dataframes[ticker].loc[date, 'high'],
                    low=self.engine.dataframes[ticker].loc[date, 'low'],
                    close=self.engine.dataframes[ticker].loc[date, 'close'],
                    volume=self.engine.dataframes[ticker].loc[date, 'volume'],
                    index=bar_index,
                    timestamp=date,
                    resolution=self.datas.resolution,
                    ticker=ticker
                )
                bars[ticker] = bar

            # If Date is not the first date
            if bar_index > 0:
                # Update Portfolio
                self.engine.compute_portfolio_stats(bar_index)

                # Process Orders
                self.engine.compute_orders(bars)

                # Process Active Trades
                self.engine.compute_trades_stats(bars)

            # Filter Universe for Assets Eligible for Trading in a specific date
            eligible_assets, non_eligible_assets = self.engine.filter_eligible_assets(date)

            # Executing Signals
            for ticker in non_eligible_assets:
                # Units of asset in holding (Set to zero)
                self.engine.portfolio.dataframe.loc[bar_index, f'{ticker} units'] += 0
                self.engine.portfolio.dataframe.loc[bar_index, f'{ticker} open_pnl'] += 0

            # Calculate Asset Exposure Allocation
            allocation_matrix = self.sizer.calculate_risk()

            # Run All Alphas in the backtester
            for alpha in self.alphas:
                # Get Alpha Allocation for each ticker
                allocation = {}
                
                for asset_name in allocation_matrix.keys():
                    allocation[asset_name] = allocation_matrix[asset_name][alpha.name]

                alpha_long, alpha_short = alpha.next(eligibles=eligible_assets, datas=bars, allocation_per_ticker=allocation)

                non_eligible_assets = list(set(eligible_assets) - set(alpha_long + alpha_short))
                for ticker in non_eligible_assets:
                    # Units of asset in holding (Set to zero)
                    self.engine.portfolio.dataframe.loc[bar_index, f'{ticker} units'] += 0
                    self.engine.portfolio.dataframe.loc[bar_index, f'{ticker} open_pnl'] += 0
        
        if not analysis_mode:
            print(f"Backtest Complete. Final Equity : {self.engine.portfolio.dataframe.iloc[-1, self.engine.portfolio.dataframe.columns.get_loc('balance')]}")

        return self.engine.portfolio


    def reset_backtester(self, dataframes:pd.DataFrame):
        '''
        Resets the backtester for another run. Resets the engine with new data.
        '''
        # Reset DataLoader
        self.datas.reset_dataloader(dataframes)

        # Use DataLoader to Reset Engine
        self.engine.reset_engine(self.datas)

        # Reset Backtester; Reset Alphas with new engine
        self.__init__(
                 self.datas, 
                 self.engine, 
                 self.alphas, 
                 self.max_allocation,
                 True)


    @property
    def original_dataframes(self):
        return self._original_dfs
    
    @original_dataframes.setter
    def original_dataframes(self, dataframes):
        self._original_dfs = dataframes


if __name__ == '__main__':
    import yfinance as yf
    import pandas as pd
    import os
    
    from alpha import BaseAlpha
    from dataloader import DataLoader
    from reporters import Reporter  # noqa: F401
    from utils import clear_terminal
    

    start_date = '2020-01-02'
    end_date = '2023-12-31'

    clear_terminal()
    with open('logs.log', 'w'):
        pass

    tickers = ['AAPL'] #, 'GOOG', 'TSLA']
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

    # # backtester.add_alpha(alpha)

    trade_history = backtester.backtest()

    # # Use Reported
    # reporter = Reporter(trade_history)
    # metric :pd.DataFrame = reporter.report()['full'][1]
    # metric.to_csv('data/test.csv', index=True)