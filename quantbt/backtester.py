import uuid
import pickle # noqa
import pandas as pd

from typing import List, Dict, Literal, Union
from copy import deepcopy, copy
from pathlib import Path

from engine import Engine
from alpha import Alpha
from orders import Order
from dataloader import DataLoader
from sizers import Sizer
from utils import Bar, Logger, Resolution # noqa: F401

exectypes = Order.ExecType

class Backtester:
    _original_dfs : dict[pd.DataFrame] = None
    logger = Logger('logger_backtester')
    backtest_storage_path = Path('data/backtests')

    params = {
        'max_exposure' : 1
    }

    def __init__(self, start_date:str, end_date:str, max_exposure:float) -> None:
        
        self.id = str(uuid.uuid4())
        self.start_date = start_date
        self.end_date = end_date
        self.params.update(max_exposure=max_exposure)

        self.alphas : List[Alpha] = None
        self.datas : DataLoader = None
        self.engine : Engine = None
        self.sizer : Sizer = None
    
        self._datas_uninit : Dict[Literal['tickers', 'dataframes', 'resolution']] = {
            'tickers' : [], # Stores each ticker added
            'dataframes' : [], # Stores each dataframe added
            'resolution' : Resolution('D') # Stores the general (minimum) resolution for the backtest
        } # Stores Uninitialized Data

        self._alphas_uninit : List[Alpha] = [] # Stores Uninitialized Alphas


    def add_alpha(self, alpha:Alpha):
        assert (alpha is not None) and (isinstance(alpha, Alpha)), '`alpha` passed is not an Alpha object'
        self._alphas_uninit.append(alpha)

        self.logger.info(f"Alpha `{alpha.name}` added.")


    def add_data(self, ticker:str, dataframe:pd.DataFrame, resolution:Union[str,int]):
        '''Add Dataframes to self._datas'''

        # Assert Data Types and Content
        assert ticker is not None, '`ticker` cannot be None.'
        assert (dataframe is not None) and (not dataframe.empty), '`dataframe` must contain some data.'
        assert set(['open', 'high', 'low', 'close', 'volume']).issubset(dataframe.columns.str.lower())

        resolution = Resolution(resolution) # Create a resolution instance

        # Append New Data
        self._datas_uninit['tickers'].append(ticker.upper())
        self._datas_uninit['dataframes'].append(dataframes)
        self._datas_uninit['resolution'] = min(self._datas_uninit['resolution'], resolution)

        # Feedback
        self.logger.info(f"{ticker} data added.")


    def backtest(self, analysis_mode:bool=False):
        if not analysis_mode:
            # Initialize the backtester components
            self._initiate_backtest()

            # Store the initialized data
            self._original_dfs = deepcopy(self.datas.dataframes)
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


        # Test :
        for alpha in self.alphas:
            if '_signals' in alpha.__dict__:
                df = pd.DataFrame(alpha._signals)
                df.to_parquet('alpha_signals.parquet')

        return self.engine


    def _initiate_backtest(self):
        '''
        Initializes the components of the backtester, before running a backtest
        '''
        # Initialize the Dataloader
        self._init_dataloader()

        # Initialize the Engine
        self.engine = Engine(self.datas)
        
        # Initialize the Alphas
        self._init_alphas

        # Initialize the Sizer
        self.sizer = Sizer(self.engine, self.alphas, self.params.get('max_exposure', 1))


    def _init_dataloader(self):
        '''Create Dataloader for Backtester Engine'''
        
        # Assert Available Data
        if (len(self._datas_uninit['resolution']) == 0) and (len(self._datas_uninit['dataframes']) == 0):
            self.logger.warning('No data available for backtest.')

        dataframes = { self._datas_uninit['tickers'][index] : self._datas_uninit['tickers'][index]
                      for index in len(self._datas_uninit['tickers'])}
        resolution = self._datas_uninit['resolution']

        # Set self.datas
        self.datas = DataLoader(dataframes=dataframes, resolution=resolution, start_date=self.start_date, end_date=end_date)


    def _init_alphas(self):
        '''
        Initializes all added alphas in the backtester
        '''

        if len(self._alphas_uninit) < 0: 
            self.logger.warning('No alphas have been added for backtesting')

        # Assert Engine has been created
        assert self.engine is not None, 'Backtester `engine` has not been initialized.'
        
        self.alphas = []

        # Reset all alphas
        for alpha in self._alphas_uninit:
            alpha.reset_alpha(self.engine)
        
            # Add alpha to engine observers
            self.engine.add_observer(alpha)

            # Add Initialized Alpha
            self.alphas.append(alpha)
        
    
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


    def copy(self, deep:bool=False):
        # Create a copy 
        backtester = deepcopy(self) if deep else copy(self)

        # Reset Backtester ID
        backtester.id = str(uuid.uuid4())

        return backtester


    @property
    def original_dataframes(self):
        return self._original_dfs
    

    @original_dataframes.setter
    def original_dataframes(self, dataframes):
        self._original_dfs = dataframes


    # PICKLE-COMPATIBILITY
    def __getstate__(self):
        state = self.__dict__.copy()
        return state


    def __setstate__(self, state):
        # Customize the object reconstruction
        self.__dict__.update(state)


if __name__ == '__main__':
    import yfinance as yf # noqa
    import pandas as pd
    import os # noqa
    
    from alpha import BaseAlpha, EmaCrossover # noqa
    from strategies import PipMinerStrategy
    from dataloader import DataLoader
    from reporters import AutoReporter  # noqa: F401
    from utils import clear_terminal

    start_date = '2018-01-01'
    end_date = '2020-12-29'

    clear_terminal()
    with open('logs.log', 'w'):
        pass

    # tickers = ['GOOGL'] # 'GOOG', 'TSLA', 'MSFT', 'META', 'GOOGL', 'NVDA', 'AMZN', 'UNH']
    # ticker_path = [f'data/prices/{ticker}.csv' for ticker in tickers]

    # dfs = []

    # for ticker in tickers:
    #     file_name = f'data/prices/{ticker}.csv'

    #     if os.path.exists(file_name):
    #         df = pd.read_csv(file_name, index_col='Date', parse_dates=True)
    #     else:
    #         df = yf.download(ticker, start=start_date, end=end_date)
    #         df.to_csv(file_name)
            
    #     dfs.append(df)
    
    # FOR CRYPTO
    tickers = ['BTCUSDT']# 'DOGEUSDT', 'ETHUSDT', 'GMTUSDT', 'SOLUSDT']

    dfs = []

    for ticker in tickers:
        file_name = f'/Users/jerryinyang/Code/quantbt/data/prices/{ticker}_1D.parquet'
        df = pd.read_parquet(file_name)
        dfs.append(df)

    dataframes = dict(zip(tickers, dfs))

    # Create DataHandler
    dataloader = DataLoader(dataframes, '1d', start_date, end_date)
    engine = Engine(dataloader)

    # alpha = BaseAlpha('base_alpha', engine, .1, .05)
    alpha = PipMinerStrategy('pip_miner', engine, 5, 24, 6, 85)

    backtester = Backtester(dataloader, engine, alpha, 1)

    trades = backtester.backtest()

    # Use Reporter
    reporter = AutoReporter('full', 'full')
    reporter.compute_report(backtester)
    reporter.compute_report(backtester)

    # Pickle the instance
    with open('reporter.pkl', 'wb') as file:
        pickle.dump(reporter, file)

