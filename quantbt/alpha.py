import numpy as np

from engine import Engine
from indicators import EMA
from observers import Observer
from orders import Order
from trades import Trade
from utils import Bar, Logger, DotDict
from utils import debug, random_suffix   # noqa: F401
# from utils_tv import na

from typing import List, Dict, Tuple, Union
from abc import ABC, abstractmethod


exectypes = Order.ExecType

class Alpha(Observer, ABC):

    logger = Logger('logger_alpha', file_location='log_signals.log')
    parameters = DotDict({
    })

    # Alias
    params = parameters

    def __init__(self, engine:Engine, **kwargs) -> None:
        '''
        This class handles the creating and sending of orders to the engine
        Arguments: 
            engine : The broker emulator
        '''

        # Update params to self.params
        params = list(set(self.params.keys()).intersection(kwargs.keys()))
        self.params.update(**{_key : kwargs.get(_key) for _key in params})

        # Setting each key-value pair in parameters as a class attribute
        for key, value in self.params.items():
            setattr(self, key, value)

        self.engine = engine

        # Generate placeholder name
        self._name = self.__class__.__name__.lower() + '-' + random_suffix()
        self._name = kwargs.get('name', self._name)

        # Add self to engine.observers
        Observer.__init__(self, self._name)
        self.engine.add_observer(self)

        # Store Trades, History
        self._orders : List[Order] = []
        self._trades : Dict[str, Dict[int, Order]] = {key : {} for key in self.engine.tickers} # Syntax : Dict[ticker, Dict [ trade_id : Trade()]]
        self._history : List[Trade] = []

        # Store Allocations
        self._allocations : Dict[str, float] = {}


    @abstractmethod
    def next(self, eligibles:List[str], datas:Dict[str, Bar], allocation_per_ticker:Dict[str, float]):
        '''
        Update strategy values with Bar objects for each ticker

        Returns: A pandas dataframe with each ticker as a row, \
            and (signal_long, signal_short, signal_exit_long, signal_exit_short, \
            exit_profit_price, exit_loss_price, exit_
        '''

        # Update the allocations
        self._allocations.update(allocation_per_ticker)
        
        # Runs self.warmup, and return if alpha is warmed up
        return self._warmup(datas)
        

    @abstractmethod
    def reset_alpha(self, engine:Engine):
        '''
        Reset Alpha Engine.
        '''
        self.__init__(engine)


    @abstractmethod
    def _signal_generator(self, eligibles:list[str]) -> tuple:        
        return eligibles, []

    
    def _calculate_asset_allocation(self, bar : Bar):
        # Get the current balance
        balance = self.engine.portfolio.dataframe.loc[bar.index , 'balance']
        ticker = bar.ticker 
        
        # Calculate the risk amount, based on available balance
        return balance * self._allocations[ticker]
        

    @abstractmethod
    def _warmup(self, datas:Dict[str, Bar]) -> bool:
        """
        Warms up the class attributes where it applies. 
        
        Returns:
            bool : All class objects have been warmed (where applies)
        """

        return True
        

    def _update_positions(self, value : Union[Order,Trade]) -> None:
        """
        Update the positions of the alpha with the provided Order or Trade.

        Parameters:
        - value: Union[Order, Trade]
            The Order or Trade object to be processed.

        Returns: None

        Notes:
        - Accepted Orders are added to self._orders.
        - Filled or Canceled Orders are removed from self._orders.
        - Active Trades are added to self._trades, organized by ticker.
        - Closed Trades are moved from self._trades to self._history.
        """

        # Confirm that the passed order/trade belongs to this alpha
        if not value.alpha_name == self.name:
            return 
        
        # Add Orders to self.orders, when they are accepted
        if isinstance(value, Order):
            order = value
            
            # Newly Accepted Order
            if order.status == Order.Status.Accepted:
                if order not in self._orders:
                    self._orders.append(order)

            # Newly Removed Order
            elif order.status in [Order.Status.Filled, Order.Status.Canceled]:
                # Confirm that self.orders contain this order
                if order in self._orders:
                    self._orders.remove(order)

        elif isinstance(value, Trade):
            trade = value 

            # Add Active Trades to self.trades
            if trade.status == Trade.Status.Active:
                self._trades[trade.ticker][trade.id] = trade

            # Add Closed Trades to self.history
            elif trade.status == Trade.Status.Closed:
                # Remove the trade from self.trade dictionary (key = trade_id)
                self._trades[trade.ticker].pop(trade.id)
                self._history.append(trade)


    # Observer.update()
    def update(self, value : Union[Order,Trade]):
        self._update_positions(value=value)
    

    # region --- HANDLE ORDERS AND TRADES
    def buy(self, bar, price, size:float, exectype:Order.ExecType, 
            stoplimit_price:float=None, parent_id:str=None,
            exit_profit:float=None, exit_loss:float=None,
            exit_profit_percent:float=None, exit_loss_percent:float=None,
            trailing_percent:float=None, family_role=None, 
            expiry_date=None) -> Order:

        return self.engine.buy(
            bar, price, size, exectype, 
            stoplimit_price, parent_id,
            exit_profit, exit_loss,
            exit_profit_percent, exit_loss_percent,
            trailing_percent, family_role, 
            expiry_date, alpha_name=self.name
        )


    def sell(self, bar, price, size:float, exectype:Order.ExecType, 
            stoplimit_price:float=None, parent_id:str=None,
            exit_profit:float=None, exit_loss:float=None,
            exit_profit_percent:float=None, exit_loss_percent:float=None,
            trailing_percent:float=None, family_role=None, 
            expiry_date=None) -> Order:
        
        return self.engine.sell(
            bar, price, size, exectype, 
            stoplimit_price, parent_id,
            exit_profit, exit_loss,
            exit_profit_percent, exit_loss_percent,
            trailing_percent, family_role, 
            expiry_date, alpha_name=self.name
        )


    def cancel_order(self, order:Union[Order, List[Order]]):
        self.engine._cancel_order(order)


    def cancel_all_orders(self):
        self.cancel_order(self.engine.orders)
 

    def close_trade(self, trade : Trade, bar, price : float ):
        self.engine._close_trade(trade, bar, price)


    def close_all_trades(self, bars:Dict[str, Bar]):
        trades_list = self.engine.trades

        # For Each Ticker with Open Trades
        for ticker in trades_list.keys():
            trades = trades_list[ticker]
            bar = bars[ticker]
            price = bar.close

            # Close Each Open Trade
            for trade in trades:
                self.close_trade(trade, bar, price)

    # endregion


    # region --- PICKLE-COMPATIBILITY
    def __getstate__(self):
        state = self.__dict__.copy()
        return state


    def __setstate__(self, state):
        # Customize the object reconstruction
        self.__dict__.update(state)
    # endregion


class EmaCrossover(Alpha):

    parameters = DotDict({
        'source' : 'close',
        'fast_length' : 20,
        'slow_length' : 50,
        'profit_perc' : 0.1,
        'loss_perc' : 0.05,
    })
    params = parameters

    def __init__(self, engine:Engine, source :str, fast_length : float, slow_length : float, profit_perc:float, loss_perc:float, **kwargs) -> None:
        '''
        This class handles the creating and sending of orders to the engine
        Arguments: 
            engine : The broker emulator
        '''
        # Update params to self.params
        kwargs.update(source=source, 
                      fast_length=fast_length,
                      slow_length=slow_length,
                      profit_perc=profit_perc,
                      loss_perc=loss_perc)

        super().__init__(engine, **kwargs)
    
        # Store EMAs per ticker in engine data
        self.emas : Dict[str, Tuple[EMA, EMA]] = {}
        
        # Create EMA for all assets in the engine
        for ticker in self.engine.tickers:
            fast_ema = EMA(self.source, self.fast_length, name=f"{ticker}_fast_ema",)
            slow_ema = EMA(self.source, self.slow_length, name=f"{ticker}_slow_ema",)

            self.emas[ticker] : Tuple[EMA, EMA] = (fast_ema, slow_ema)


    def next(self, eligibles: List[str], datas: Dict[str, Bar], allocation_per_ticker: Dict[str, float]):
        # Check if all alpha attributes 
        if not super().next(eligibles, datas, allocation_per_ticker):
            # Returns, if alpha is still warming up
            return [], []

        # Decision-making / Signal-generating Algorithm
        alpha_long, alpha_short = self._signal_generator(eligibles)
        eligible_assets = list(set(alpha_long + alpha_short))
        
        for ticker in eligible_assets:
            bar = datas[ticker]

            # Calculate Risk Amount based on allocation_per_ticker
            risk_dollars =  self._calculate_asset_allocation(bar)

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

        
    def reset_alpha(self, engine:Engine):
        '''
        Resets the alpha with a new engine.
        '''
        self.__init__(engine, self.params.source, self.params.fast_length, self.params.slow_length, self.params.profit_perc, self.params.loss_perc)
        self.logger.info(f'Alpha {self.name} successfully reset.')


    def _signal_generator(self, eligibles:List[str]) -> tuple:        
        alpha_long = [] 
        alpha_short = []

        for ticker in eligibles:
            fast, slow = self.emas.get(ticker)

            # Find Crossovers
            cross_long = (fast[0] > slow[0]) and (fast[1] <= slow[1])
            cross_short = (fast[0] < slow[0]) and (fast[1] >= slow[1])

            if cross_long:
                alpha_long.append(ticker)
        
            elif cross_short:
                alpha_short.append(ticker)

        
        return alpha_long, alpha_short  


    def _warmup(self, datas:Dict[str, Bar]) -> bool:
        """
        Warms up the class attributes where it applies. 
        
        Returns:
            bool : All class objects have been warmed (where applies)
        """

        # Check all EMA.is_ready
        warmed = True

        # Iterate through all EMAs
        for ticker, ema_pair in self.emas.items():
            # Get the Data
            bar = datas[ticker]
            
            # Update Each EMA
            for ema in ema_pair:
                ema.update(bar)
                warmed = warmed and ema.is_ready

            self.logger.info(f"EMA : {ticker}. \nTime : {bar.timestamp} | Fast : {ema_pair[0][0]} | Slow : {ema_pair[1][0]}")

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
            risk_dollars =  self._calculate_asset_allocation(bar)

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
        
        # alpha_long = [list_scores[0]] 
        # alpha_short = [list_scores[-1]] 

        return eligibles, [] # alpha_long, alpha_short       


    def reset_alpha(self, engine:Engine):
        '''
        Resets the alpha with a new engine.
        '''

        self.__init__(self.name, engine, self.profit_perc, self.loss_perc)
        self.logger.info(f'Alpha {self.name} successfully reset.')


    def warmup(self, datas: Dict[str, Bar]):
        return super()._warmup(datas)


# class EmaCrossover(Alpha):
#     params = {}
        
#     def __init__(self, name : str, engine: Engine, source :str, fast_length : float, slow_length : float, profit_perc:float, loss_perc:float) -> None:
#         super().__init__(name, engine)

#         self.source = source
#         self.fast_length = min(fast_length, slow_length)
#         self.slow_length = max(fast_length, slow_length)
#         self.profit_perc = profit_perc
#         self.loss_perc = loss_perc

#         self.params['fast_length'] = self.fast_length
#         self.params['slow_length'] = self.slow_length

#         self.warmup_period = self.slow_length
    
#         self.emas : Dict[str, List[EMA, EMA]] = {}
        
#         # Create EMA Setup for all assets in the engine
#         for ticker in engine.tickers:
#             fast_ema = EMA(f"{ticker}_fast_ema", self.source, self.fast_length)
#             slow_ema = EMA(f"{ticker}_slow_ema", self.source, self.slow_length)

#             self.emas[ticker] = [fast_ema, slow_ema]            


#     def compute_values(self, datas:Dict[str, Bar]):
#         # Update Indicators Here, per ticker
#         for ticker, ema_pair in self.emas.items():
#             data = datas[ticker]
#             # Update each ema with the ticker's data
#             for ema in ema_pair:
#                 ema.update(data)

#                 # if 'slow_ema' in ema.name:
#                 #     debug(ema.value, datas[ticker], self.warmup_period)


#     def next(self, eligibles:List[str], datas:Dict[str, Bar], allocation_per_ticker:Dict[str, float]):
#         if super().next(eligibles, datas, allocation_per_ticker):
#             # Returns True if warmup period is active
#             return [], []
        
#         if self.warmup_period:
#             self.warmup
        
#         # Decision-making / Signal-generating Algorithm
#         alpha_long, alpha_short = self.signal_generator(eligibles)
#         eligible_assets = list(set(alpha_long + alpha_short))
        
#         for ticker in eligible_assets:
#             bar = datas[ticker]

#             # Calculate Risk Amount based on allocation_per_ticker
#             risk_dollars =  self._calculate_asset_allocation(bar)

#             # Tickers to Long
#             if ticker in alpha_long:
#                 entry_price = bar.close

#                 position_size = risk_dollars / entry_price

#                 exit_tp = entry_price * (1 + self.profit_perc)
#                 exit_sl = entry_price * (1 - self.loss_perc)
                
#                 # Create and Send Long Order
#                 self.buy(bar, entry_price, position_size, exectypes.Market, exit_profit=exit_tp, exit_loss=exit_sl)                    

#                 # Tickers to Short
#             elif ticker in alpha_short:
#                 entry_price = bar.close
#                 position_size = -1 * risk_dollars / entry_price

#                 exit_tp = entry_price * (1 - self.profit_perc)
#                 exit_sl = entry_price * (1 + self.loss_perc)
                
#                 # Create and Send Short Order
#                 self.sell(bar, entry_price, position_size, exectypes.Market, exit_profit=exit_tp, exit_loss=exit_sl)

#         return alpha_long, alpha_short 


#     def signal_generator(self, eligibles:list[str]) -> tuple:        
#         alpha_long = [] 
#         alpha_short = []

#         for ticker in eligibles:
#             fast, slow = self.emas[ticker]

#             if (not fast[0]) or (not slow[0]):
#                 break

#             # Find Crossovers
#             cross_long = (fast[0] > slow[0]) and (fast[1] <= slow[1])
#             cross_short = (fast[0] <= slow[0]) and (fast[1] > slow[1])

#             if cross_long:
#                 alpha_long.append(ticker)
#             elif cross_short:
#                 alpha_short.append(ticker)

        
#         return alpha_long, alpha_short  


#     def reset_alpha(self, engine:Engine):
#         '''
#         Resets the alpha with a new engine.
#         '''

#         self.__init__(self.name, engine, self.source, self.fast_length, self.slow_length, self.profit_perc, self.loss_perc)
#         self.logger.info(f'Alpha {self.name} successfully reset.')


#     def warmup(self, datas: Dict[str, Bar]):
#         self.warmup_period -= 1

