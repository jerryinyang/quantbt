'''
CUSTOM STRATEGIES (ALPHAS) AND THEIR DEPENDENCIES
'''

import numpy as np
from collections import deque
from typing import List, Dict

from alpha import Alpha
from engine import Engine
from models.pip_miner import PipMiner
from orders import Order
from utils import Bar, debug # noqa: F401
from utils_tv import na #noqa: F401

exectypes = Order.ExecType

class BaseAlpha(Alpha):
    def __init__(self, name : str, engine: Engine, profit_perc:float, loss_perc:float) -> None:
        super().__init__(name, engine)

        self.profit_perc = profit_perc
        self.loss_perc = loss_perc

    def next(self, eligibles:List[str], datas:Dict[str, Bar], allocation_per_ticker:Dict[str, float]):
        super().next(eligibles, datas, allocation_per_ticker)

        # Decision-making / Signal-generating Algorithm
        alpha_long, alpha_short = self.signal_generator(eligibles)
        eligible_assets = list(set(alpha_long + alpha_short))
        
        for ticker in eligible_assets:
            bar = datas[ticker]

            # Calculate Risk Amount based on allocation_per_ticker
            risk_dollars =  self.sizer(bar)

            # Tickers to Long
            if ticker in alpha_long:
                entry_price = bar.close

                position_size = risk_dollars / entry_price

                exit_tp = entry_price * (1 + self.profit_perc)
                exit_sl = entry_price * (1 - self.loss_perc)
                
                # Create and Send Long Order
                self.buy(bar, entry_price, position_size, exectypes.Market, exit_profit=exit_tp, exit_loss=exit_sl)                    

                # Tickers to Short
            elif ticker in alpha_short:
                entry_price = bar.close
                position_size = -1 * risk_dollars / entry_price

                exit_tp = entry_price * (1 - self.profit_perc)
                exit_sl = entry_price * (1 + self.loss_perc)
                
                # Create and Send Short Order
                self.sell(bar, entry_price, position_size, exectypes.Market, exit_profit=exit_tp, exit_loss=exit_sl)

        return alpha_long, alpha_short 


    def signal_generator(self, eligibles:list[str]) -> tuple:
        alpha_scores = { key : np.random.rand() for key in eligibles}

        alpha_scores = {
            key : value \
                for key, value in sorted(alpha_scores.items(), key=lambda item : item[1])
                } # Sorts the dictionary
        
        list_scores = list(alpha_scores.keys())

        if not list_scores:
            return [], []
        
        alpha_long = [list_scores[0]] 
        alpha_short = [list_scores[-1]] 

        return alpha_long, alpha_short       


    def reset_alpha(self, engine:Engine):
        '''
        Resets the alpha with a new engine.
        '''

        self.__init__(self.name, engine, self.profit_perc, self.loss_perc)
        self.logger.info(f'Alpha {self.name} successfully reset.')



class PipMinerStrategy(Alpha):

    params = {
        'n_pivots' : 5, 
        'lookback' : 24, 
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

            debug(data_train)
            exit()

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
            risk_dollars =  self.sizer(bar)

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
            signal = miner.generate_signal(window)

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