'''
CUSTOM STRATEGIES (ALPHAS) AND THEIR DEPENDENCIES
'''

from collections import deque
from copy import deepcopy
from typing import Dict, List, Literal

import numpy as np
from alpha import Alpha
from engine import Engine
from models.pip_miner import PipMiner, PipMinerMulti  # noqa
from orders import Order
from utils import Bar, DotDict, debug  # noqa: F401
from utils_tv import na

exectypes = Order.ExecType


class PipMinerAlpha(Alpha):

    parameters = DotDict({
        'hold_period' : 6,
    })
    params = parameters

    def __init__(self, engine:Engine, miner : PipMinerMulti, **kwargs) -> None:
        '''
        This class handles the creating and sending of orders to the engine
        Arguments: 
            engine : The broker emulator
        '''
        hold_period = miner.hold_period
        lookback = miner.lookback
        n_pivots = miner.n_pivots

        # Update params to self.params
        kwargs.update(hold_period=hold_period,)

        super().__init__(engine, **kwargs)

        # Keep the miner
        self._miner = miner

        # Store Prices, for each ticker
        self._windows : Dict[str, deque] = {ticker : deque([np.nan] * lookback, maxlen=lookback) 
                                            for ticker in engine.tickers}

        # Store Last Pivots, for comparison
        self._last_pivots : Dict[str, List] = {ticker : [0] * n_pivots
                                               for ticker in engine.tickers}

        # Track Signals and Holding Periods
        self._signals : Dict[str, Literal[-1, 0, -1]] = {ticker : 0
                                                         for ticker in engine.tickers}
        self._holdings : Dict[str, int] = {ticker : 0
                                          for ticker in engine.tickers}
        
        # # Keep Signals
        # self._signals = {
        #     'Signals' : [],
        #     'Data' : [],
        # }


    def next(self, eligibles: List[str], datas: Dict[str, Bar], allocation_per_ticker: Dict[str, float]):
        # Check if alpha is warmed
        if not super().next(eligibles, datas, allocation_per_ticker):
            # Returns, if alpha is still warming up
            return [], []

        # Decision-making / Signal-generating Algorithm
        alpha_long, alpha_short = self._signal_generator(eligibles)

        # Get the list of tickers with pending orders
        pending_orders = [order.ticker for order in self._orders]
        
        # Manage All Tickers
        for ticker in self.engine.tickers:
            bar = datas[ticker]

            # Check for pending orders for that ticker
            if ticker in pending_orders:
                # Ignore new signals
                continue
            
            # If ticker has an open trade
            # Manage the open trades, ignore new signals
            trades = self._trades[ticker]

            if len(trades) > 0:
                
                # Confirm that only one trade is active
                assert len(trades) == 1, 'Somehow, more than one trades are open.'

                # Get the first item in the dictionary
                trade_name = next(iter(trades))
                trade = trades[trade_name]

                # Holding period is still valid
                if self._holdings[ticker] >= 0:
                    self._holdings[ticker] -= 1

                # Holding period has been exhausted    
                if self._holdings[ticker] < 0:
                    # Close the trade
                    self.close_trade(bar, trade)
                    self._signals[ticker] = 0                       
                
                # Trade is still active
                else:
                    continue

            # No open positions
            risk_dollars =  self._calculate_asset_allocation(bar)
            position_size = risk_dollars / bar.close

            if ticker in alpha_long:
                self.buy(bar, bar.close, position_size, exectypes.Market)
                self._holdings[ticker] = self.hold_period
                self._signals[ticker] = 1

            elif ticker in alpha_short:
                self.sell(bar, bar.close, position_size, exectypes.Market)
                self._holdings[ticker] = self.hold_period - 1
                self._signals[ticker] = -1


        return alpha_long, alpha_short 

        
    def reset_alpha(self, engine:Engine):
        '''
        Resets the alpha with a new engine.
        '''
        self.__init__(engine, self._miner)
        self.logger.info(f'Alpha {self.name} successfully reset.')


    def _signal_generator(self, eligibles:List[str]) -> tuple:        
        alpha_long = [] 
        alpha_short = []

        for ticker in eligibles:
            # Get the data window
            window = deepcopy(self._windows[ticker])
            window.reverse()

            # Check the signal value and holding period
            signal = self._signals[ticker]
            holding = self._holdings[ticker]

            # Previous signal is still active
            # Skip Signal Generation 
            if (signal != 0) and (holding > 0):
                continue
                
            # No previous signal is active
            # Get the last pivot for that ticker
            last_pivots = self._last_pivots[ticker]

            # Generate the signal
            signal, pivot_prices = self._miner.generate_signal(np.array(window))
            self._last_pivots[ticker] = pivot_prices

            # Check internal points to make sure they are not the same as the previous
            current_internal_points = pivot_prices[1:-1]
            previous_internal_points = last_pivots[1:-1]

            if current_internal_points == previous_internal_points:
                # return 0, pivot_prices
                continue

            # For new unique patterns
            # Append tickers based on signal
            if signal > 0:
                alpha_long.append(ticker)
        
            elif signal < 0:
                alpha_short.append(ticker)

        return alpha_long, alpha_short  


    def _warmup(self, datas:Dict[str, Bar]) -> bool:
        """
        Warms up the data windows in self._windows
        
        Returns:
            bool : All windows in self._windows are filled.
        """

        # Check all EMA.is_ready
        warmed = True

        for ticker in self.engine.tickers:
            bar = datas[ticker]

            # Append new data to self._windows
            self._windows[ticker].appendleft(bar.close)

            # Check if all windows are warmed
            warmed = warmed and all([not na(data) 
                                     for data in self._windows[ticker]])
        
        return warmed
          
    
    def _calculate_asset_allocation(self, bar: Bar) -> float:
        """
        Calculates the allocation size for a given asset at a specific time.

        Parameters:
        - self: The instance of the class containing this method.
        - bar (Bar): An instance of the Bar class representing the current market data.

        Returns:
        float: The calculated allocation size based on the current balance and predefined allocations.

        Raises:
        None

        Notes:
        - The function retrieves the current balance from the portfolio's dataframe.
        - Uses the ticker information from the provided Bar object to identify the asset.
        - Calculates the risk amount based on the available balance and predefined allocations.
        """
        # Get the current balance
        balance = self.engine.portfolio.dataframe.loc[bar.index, 'balance']
        ticker = bar.ticker

        # Calculate the risk amount, based on available balance and predefined allocations
        return balance * self._allocations[ticker]


    def stop(self):
        pass
        # import pandas as pd

        # # debug([len(item) for item in self._signals.values()])
        # data = pd.DataFrame(self._signals)
        # data.to_csv('/Users/jerryinyang/Code/quantbt/research/backtesting_signals.csv', index=False)