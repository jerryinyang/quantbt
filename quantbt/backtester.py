import inspect
import uuid
import pickle
import pandas as pd
import numpy as np # noqa

from typing import List, Dict, Literal, Tuple
from copy import deepcopy, copy
from pathlib import Path

from engine import Engine
from alpha import Alpha
from orders import Order
from dataloader import DataLoader
from sizers import Sizer
from utils import Bar, Logger, debug # noqa: F401


exectypes = Order.ExecType

class Backtester:
    _original_dfs : Dict[str, pd.DataFrame] = None
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

        self.alphas : List[Alpha] = []
        self.datas : DataLoader = None
        self.engine : Engine = None
        self.sizer : Sizer = None
    
        self._datas_uninit : Dict[Literal['tickers', 'dataframes']] = {
            'tickers' : [], # Stores each ticker added
            'dataframes' : [], # Stores each dataframe added
        } # Stores Uninitialized Data

        self._alphas_uninit : List[Tuple[Alpha, Dict]] = [] # Stores Uninitialized Alphas


    def add_alpha(self, alpha:Alpha, **kwargs):
        """
        This method adds an Alpha object to the current instance.

        The Alpha object is instantiated with the provided keyword arguments.
        An AssertionError is raised if any required argument for the Alpha
        constructor is missing from kwargs.

        Args:
            alpha (Alpha): The Alpha class to be instantiated. Must be passed as a class, not an instance.
            **kwargs: Keyword arguments for instantiating the Alpha object.

        Raises:
            AssertionError: If alpha is None or not a class, or if any required argument for the Alpha constructor is missing from kwargs.
        """

        assert (alpha is not None) and (isinstance(alpha, object)), '`alpha` passed is not an Alpha object (pass as an object, not an instance)'

        # Add the engine to the arguments dictionary
        kwargs.update(engine=self.engine)

        # Get the list of argument names that alpha's constructor expects
        sig = inspect.signature(alpha.__init__)

        # Select arguments without default values
        alpha_args = [name for name, param in sig.parameters.items() if (param.default == inspect.Parameter.empty) and (name != 'self') and (name != 'kwargs')]

        # Check if all required arguments are present in kwargs
        for arg in alpha_args:
            assert arg in kwargs.keys(), f'Argument {arg} not found in kwargs'

        # Add Alpha and its arguments to self._alphas_uninit
        self._alphas_uninit.append((alpha, kwargs))

        if kwargs.get('name'):
            self.logger.info(f"Alpha `{kwargs['name']}` added.")


    def add_data(self, ticker:str, dataframe:pd.DataFrame, date_column_index=-1):
        '''Add Dataframes to self._datas'''

        # Assert Data Types and Content
        assert ticker is not None, '`ticker` cannot be None.'
        assert (dataframe is not None) and (not dataframe.empty), '`dataframe` must contain some data.'
        assert set(['open', 'high', 'low', 'close', 'volume']).issubset(dataframe.columns.str.lower())

        if date_column_index >= 0:
            dataframe.rename(columns={dataframe.columns[date_column_index]: 'date'}, inplace=True)

        # Append New Data
        self._datas_uninit['tickers'].append(ticker.upper())
        self._datas_uninit['dataframes'].append(dataframe)

        # Feedback
        self.logger.info(f"{ticker} data added.")


    def backtest(self, analysis_mode:bool=False):
        # Initialize the backtester components
        self._initiate_backtest()

        if not analysis_mode:
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
            allocation_matrix = self.sizer.get_allocation()

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
            print(f"Backtest Complete. NET RETURN : {(self.engine.portfolio.dataframe.iloc[-1, self.engine.portfolio.dataframe.columns.get_loc('balance')]) - self.engine.CAPITAL}")

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
        self._init_alphas()

        # Initialize the Sizer
        self.sizer = Sizer(self.engine, self.alphas, self.params.get('max_exposure', 1))


    def _init_dataloader(self):
        '''Create Dataloader for Backtester Engine'''
        
        # Assert Available Data
        if (len(self._datas_uninit['tickers']) == 0) and (len(self._datas_uninit['dataframes']) == 0):
            self.logger.warning('No data available for backtest.')


        dataframes = { self._datas_uninit['tickers'][index] : self._datas_uninit['dataframes'][index]
                      for index in range(len(self._datas_uninit['tickers']))}

        # Set self.datas
        self.datas = DataLoader(dataframes=dataframes, start_date=self.start_date, end_date=self.end_date)


    def _init_alphas(self):
        '''
        Initializes all added alphas in the backtester
        '''

        if len(self._alphas_uninit) < 0: 
            self.logger.warning('No alphas have been added for backtesting')

        # Assert Engine has been created
        assert self.engine is not None, 'Backtester `engine` has not been initialized.'

        # Reset all alphas
        for alpha, args in self._alphas_uninit:
            # Update arguments with self.engine
            args.update(engine=self.engine)

            # Instantiate the alpha object
            alpha = alpha(**args)
        
            # Add alpha to engine observers
            self.engine.add_observer(alpha)

            # Add Initialized Alpha
            self.alphas.append(alpha)
        
    
    def reset_backtester(self, dataframes:Dict[str, pd.DataFrame]):
        '''
        Resets the backtester for another run. Resets the engine with new data.
        '''
        # Store uninitialized alphas
        _alphas_uninit = deepcopy(self._alphas_uninit)

        # Reset Backtester; Reset Alphas with new engine
        self.__init__(
                 self.start_date,
                 self.end_date,
                 self.params['max_exposure'])
        
        # Add new datas
        for ticker, data in dataframes:
            self.add_data(ticker, data)
        
        # Add uninstialized data and alphas
        self._alphas_uninit = _alphas_uninit


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
    from strategies import PipMinerStrategy # noqa
    from reporters import AutoReporter  # noqa: F401
    from utils import clear_terminal

    start_date = '2022-01-01'
    end_date = '2023-12-31'

    clear_terminal()
    with open('logs.log', 'w'):
        pass

    with open('log_trades.log', 'w'):
        pass

    with open('log_signals.log', 'w'):
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
    tickers = ['BTCUSDT_1D'] # 'DOGEUSDT', 'ETHUSDT', 'GMTUSDT', 'SOLUSDT']

    dfs = []   

    # Create DataHandler
    backtester = Backtester(start_date=start_date, end_date=end_date, max_exposure=1)

    # backtester.add_alpha(PipMinerStrategy, name='pip_miner', n_pivots=5, lookback=24, hold_period=6, n_clusters=85, train_split_percent=.6)
    # backtester.add_alpha(BaseAlpha, name='base_alpha', profit_perc=.1, loss_perc=.05)
    backtester.add_alpha(EmaCrossover, source='close', fast_length=10, slow_length=20, profit_perc=.05, loss_perc=.01)

    for ticker in tickers:
        try:
            file_name = f'/Users/jerryinyang/Code/quantbt/data/prices/{ticker}.parquet'
            df = pd.read_parquet(file_name)
        except Exception:
            file_name = f'/Users/jerryinyang/Code/quantbt/data/prices/{ticker}.csv'
            df = pd.read_csv(file_name)
        
        backtester.add_data(ticker=ticker, dataframe=df) #, date_column_index=0)
    

    trades = backtester.backtest()

    # Use Reporter
    reporter = AutoReporter('full', 'full')
    reporter.compute_report(backtester)

    # Pickle the instance
    with open('reporter.pkl', 'wb') as file:
        pickle.dump(reporter, file)
