import logging
from typing import List

from engine import Engine, Alpha
from orders import Order
from utils import Bar, debug # noqa: F401
from copy import deepcopy



logging.basicConfig(filename='logs.log', level=logging.INFO)

exectypes = Order.ExecType

class Backtester:
    _original_dfs : Engine= None

    def __init__(self, engine:Engine, alphas:list[Alpha], in_analyzer:bool=False) -> None:
        self.engine = engine
        self.alphas : List[Alpha] = self._init_alpha(alphas)

        # If Backtester is being initialized within and Analyzer, don't modify the initial_engine
        if not in_analyzer:
            self._original_dfs = deepcopy(engine.dataframes)

    @property
    def original_dataframes(self):
        return self._original_dfs
    
    @original_dataframes.setter
    def original_dataframes(self, dataframes):
        self._original_dfs = dataframes


    def _init_alpha(self, alphas: list[Alpha]):
        if isinstance(alphas, list):
            for alpha in alphas:
                alpha.reset_alpha(self.engine)
            return alphas

        else:
            alphas.reset_alpha(self.engine)
            return [alphas]
        
    
    def add_alpha(self, alpha:Alpha):
        alpha = self._init_alpha(self.engine)
        self.alphas.append(alpha)


    def backtest(self):
        print('Initiating Backtest')

        # Set Backtest Range for Engine
        self.engine.portfolio= self.engine.init_portfolio(self.engine.date_range)
        
        # Iterate through each bar/index (timestamp) in the backtest range
        for bar_index in self.engine.portfolio.index:
            date = self.engine.portfolio.loc[bar_index, 'timestamp']

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
                    resolution=self.engine.resolution,
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
                self.engine.portfolio.loc[bar_index, f'{ticker} units'] += 0
                self.engine.portfolio.loc[bar_index, f'{ticker} open_pnl'] += 0

            # Run All Alphas in the backtester
            for alpha in self.alphas:
                alpha_long, alpha_short = alpha.next(eligibles=eligible_assets, datas=bars)

                non_eligible_assets = list(set(eligible_assets) - set(alpha_long + alpha_short))
                for ticker in non_eligible_assets:
                    # Units of asset in holding (Set to zero)
                    self.engine.portfolio.loc[bar_index, f'{ticker} units'] += 0
                    self.engine.portfolio.loc[bar_index, f'{ticker} open_pnl'] += 0
            
        print(f"Backtest Complete. Final Equity : {self.engine.portfolio.iloc[-1, self.engine.portfolio.columns.get_loc('balance')]}")
        return self.engine.history['trades']


    def reset_backtester(self, dataframes:dict):
        '''
        Resets the backtester for another run. Resets the engine with new data.
        '''
        # Reset Engines
        self.engine.reset_engine(dataframes)

        # Reset Backtester; Reset Alphas with new engine
        self.__init__(self.engine, self.alphas, True)