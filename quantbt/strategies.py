'''
CUSTOM STRATEGIES (ALPHAS) AND THEIR DEPENDENCIES
'''

import numpy as np
from collections import deque
from copy import deepcopy
from typing import List, Dict

from alpha import Alpha
from engine import Engine
from models.pip_miner import PipMiner
from orders import Order
from utils import Bar, DotDict, debug # noqa: F401
from utils_tv import na #noqa: F401

exectypes = Order.ExecType

class PipMinerStrategy(Alpha):

    params = {
        'n_pivots' : 5, 
        'lookback' : 20, 
        'hold_period' : 6,
        'n_clusters' : 85
    }


    def __init__(self, name: str, engine: Engine, n_pivots:int, lookback:int, hold_period:int, n_clusters:int, train_split_percent:float=0.6) -> None:
        super().__init__(name, engine)

        # Update Strategy Parameters
        self.params.update(n_pivots=n_pivots, lookback=lookback, hold_period=hold_period, n_clusters=n_clusters)

        self._train_split_percent = train_split_percent
        self._miners = self._init_miners(engine)

        # Store Deque (windows) of price data for miner predictions
        self._windows = {ticker : deque([np.nan] * (lookback + 1), maxlen=lookback+1) for ticker in engine.tickers}
        self.warmup_period = 1

        # Store Trade Durations for each ticker
        self._trade_durations = {ticker : {} for ticker in engine.tickers}

        # Test : Store all the signal dates
        self._signals = {
            'date' : [],
            'ticker' : [],
            'signal' : []
        }


    def _init_miners(self, engine:Engine):
        '''
        Creates and Trains PipMiner Instances for all tickers in the engine data.
        '''
        print('Training Models')

        miners = {}
        
        # Loop Throuhgh All Ticker Data in the Engine
        for ticker, dataframe in engine.dataframes.items():
            # Instantiate a miner
            miner = PipMiner(**self.params) # use self.params to fill the arguments

            # Get the close data from the ticker dataframe
            data = np.log(dataframe['close'].to_numpy())

            # Split the data for training the model
            split_index = int(round(self._train_split_percent * len(data)))
            data_train = data[:split_index]

            # Train the miner 
            miner.train(data_train)

            # Add miner to miners dictionary
            miners[ticker] = miner

        print('Models trained successfully.')
        return miners


    def next(self, eligibles:List[str], datas:Dict[str, Bar], allocation_per_ticker:Dict[str, float]):
        super().next(eligibles, datas, allocation_per_ticker)

        if self.warmup_period:
            self.warmup(datas)
            return [], []


        # Update data windows for each ticker
        for ticker, window in self._windows.items():
            window.appendleft(datas[ticker].close) # Append the new close data to the window

        # Decision-making / Signal-generating Algorithm
        alpha_long, alpha_short = self.signal_generator(eligibles)
        eligible_assets = list(set(alpha_long + alpha_short))

        # Store signal
        for ticker in datas.keys():
            bar = datas[ticker]
            signal = 1 if ticker in alpha_long else -1 if ticker in alpha_short else 0
            self._signals['date'].append(bar.timestamp)
            self._signals['ticker'].append(ticker)
            self._signals['signal'].append(signal)
        
        for ticker in eligible_assets:
            bar = datas[ticker]

            # Calculate Risk Amount based on allocation_per_ticker
            risk_dollars =  self._calculate_asset_allocation(bar)

            # Tickers to Long
            if ticker in alpha_long:
                entry_price = bar.close

                position_size = risk_dollars / entry_price

                exit_tp = None # entry_price * (1 + .1)
                exit_sl = None # entry_price * (1 - .05)
                
                # Create and Send Long Order
                self.buy(bar, entry_price, position_size, exectypes.Market, exit_profit=exit_tp, exit_loss=exit_sl)                    

                # Tickers to Short
            elif ticker in alpha_short:
                entry_price = bar.close
                position_size = -1 * risk_dollars / entry_price

                exit_tp = None # entry_price * (1 - .1)
                exit_sl = None # entry_price * (1 + .05)
                
                # Create and Send Short Order
                self.sell(bar, entry_price, position_size, exectypes.Market, exit_profit=exit_tp, exit_loss=exit_sl)

            # Scan for Exits
            if len(self._trades[bar.ticker]):
                self.scan_exits(bar)

        return alpha_long, alpha_short 


    def signal_generator(self, eligibles:list[str]) -> tuple:
        '''
        Generate signals for long and short using the miners.
        '''
        
        alpha_long, alpha_short = [], []
        for ticker in eligibles:
            # Get the data window, and miner
            window = self._windows[ticker]
            miner = self._miners[ticker]
            
            # Generate signal using the miner
            signal = miner.generate_signal(np.log(window))

            # Add signalss
            if signal > 0:
                alpha_long.append(ticker)
                
            elif signal < 0:
                alpha_short.append(ticker)

        return alpha_long, alpha_short  

    
    def reset_alpha(self, engine:Engine):
        '''
        Resets the alpha with a new engine.
        '''
        self.__init__(self.name, engine, **self.params)
        self.logger.info(f'Alpha {self.name} successfully reset.')


    def warmup(self, datas: Dict[str, Bar]):
        full = True
        # Update data windows for each ticker
        for ticker, window in self._windows.items():
            window.appendleft(datas[ticker].close) # Append the new close data to the window
            
            if np.nan in window:
                full = False

        # Check if all are full
        if full:
            self.warmup_period = 0
            
    
    def scan_exits(self, bar : Bar):

        ticker = bar.ticker

        # Read self.trades list and track the number of bars they have been open
        trades = self._trades[ticker]
        remove = []

        # Loop through each trade for every ticker
        for trade_id, trade in trades.items():
            
            # If trade duration hasn't been registered, add it with duration of one 
            if trade_id not in self._trade_durations[ticker].keys():
                self._trade_durations[ticker][trade_id] = 1
                continue
            
            self._trade_durations[ticker][trade_id] += 1
            
            if self._trade_durations[ticker][trade_id] >= (self.params['hold_period'] + 1):
                remove.append(trade)

        # Remove the trades
        for trade in remove:
            self.close_trade(trade, bar, None)
            # print(f"Removed Trade {trade.id}. \nStart Date : {trade.entry_timestamp}, End Date : {trade.exit_timestamp}")


class PipMinerAlpha(Alpha):

    parameters = DotDict({
        'hold_period' : 6,
    })
    params = parameters

    def __init__(self, engine:Engine, miner : PipMiner, **kwargs) -> None:
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

        # Track Holding Periods
        self._holdings : Dict[str, int] = {ticker : 0
                                          for ticker in engine.tickers}


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
                if self._holdings[ticker] > 1:
                    self._holdings[ticker] -= 1

                # Holding period has been exhausted    
                else:
                    # Compute the exit params 
                    exit_params = {
                        "price": None,
                        "exectype": Order.ExecType.Market,
                        "size": trade.size,  # Default size should be the same size of the entry
                        "family_role": Order.FamilyRole.ChildExit,
                        }
                     
                    # Send order to close position
                    if trade.direction.value > 0:
                        self.sell(bar, 
                                  parent_id=trade.id, 
                                  **exit_params)
                    else:
                        self.buy(bar, 
                                 parent_id=trade.id,  
                                 **exit_params)                           
                continue

            # No open positions
            risk_dollars =  self._calculate_asset_allocation(bar)
            position_size = risk_dollars / bar.close

            if ticker in alpha_long:
                self.buy(bar, bar.close, position_size, exectypes.Market)    

            elif ticker in alpha_short:
                self.buy(bar, bar.close, position_size, exectypes.Market)  

            self._holdings[ticker] = self.hold_period

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

            # Get the last pivot for that ticker
            last_pivot = self._last_pivots[ticker]

            # Generate the signal
            signal, pivot_prices = self._miner.generate_signal(np.array(window), last_pivot)
            self._last_pivots[ticker] = pivot_prices

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

