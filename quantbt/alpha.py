import pandas as pd
import numpy as np

from engine import Engine
from orders import Order
from observers import Observer
from trades import Trade
from utils import Bar


from typing import List, Dict
from abc import ABC, abstractmethod
from collections import OrderedDict as odict


exectypes = Order.ExecType

class Alpha(Observer, ABC):

    params = {

    }
    def __init__(self, name:str, engine:Engine) -> None:
        '''
        This class handles the creating and sending of orders to the engine
        Arguments: 
            engine : The broker emulator
        '''
        self.name = name
        self.engine = engine

        # Store Trades, History
        self.orders : List[Order] = []
        self.trades : Dict[str, odict[int, Order]] = {key : odict(callback=self.onTrade) for key in self.tickers} # Syntax : Dict[ticker, Dict [ trade_id : Trade()]]
        self.history : List[Trade] = []

    @abstractmethod
    def next(self, eligibles:List[str], datas:Dict[str, Bar], allocation_per_ticker:Dict[str, float]):
        '''
        Update strategy values with Bar objects for each ticker

        Returns: A pandas dataframe with each ticker as a row, \
            and (signal_long, signal_short, signal_exit_long, signal_exit_short, \
            exit_profit_price, exit_loss_price, exit_
        '''
        
        return pd.DataFrame
    

    @abstractmethod
    def sizer(self):
        pass


    @abstractmethod
    def reset_alpha(self, engine:Engine):
        '''
        Reset Alpha Engine.
        '''
        engine = self.engine

        self.__init__(engine, self.allocation)


    def update(self, value : Order|Trade):
        # Confirm that the passed order/trade belongs to this alpha
        if not value.alpha_name == self.name:
            return 
        
        # Add Orders to self.orders, when they are accepted
        if isinstance(value, Order):
            order = value
            
            # Newly Accepted Order
            if order.status == Order.Status.Accepted:
                self.orders.append(order)

            # Newly Removed Order
            elif order.status in [Order.Status.Filled, Order.Status.Canceled]:
                # Confirm that self.orders contain this order
                if order in self.orders:
                    self.orders.remove(order )

        else:
            trade = value 

            # Add Active Trades to self.trades
            if trade.status == Trade.Status.Active:
                self.trades[trade.ticker][trade.id] = trade

            # Add Closed Trades to self.history
            else:
                self.history.append(trade)
    


    # HANDLE ORDERS AND TRADES
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
            expiry_date
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
            expiry_date
        )


    def cancel_order(self, order:Order | List[Order]):
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



class BaseAlpha(Alpha):
    def __init__(self, engine: Engine, capital_allocation: float, profit_perc:float, loss_perc:float) -> None:
        super().__init__(engine, capital_allocation)

        self.profit_perc = profit_perc
        self.loss_perc = loss_perc


    def next(self, eligibles:List[str], datas: Dict[str, Bar]):

        # Decision-making / Signal-generating Algorithm
        alpha_long, alpha_short = self.signal_generator(eligibles)
        eligible_assets = list(set(alpha_long + alpha_short))
        
        for ticker in eligible_assets:
            bar = datas[ticker]

            # Calculate Allocation for Each Symbols (Equal Allocation)
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


    def sizer(self, bar:Bar):
        # Get current balance for that ticker, scaled to the strategy's allocation percent
        return self.engine.portfolio.loc[bar.index, 'balance'] * self.engine.tickers_weight[bar.ticker] * self.allocation


    def reset_alpha(self, engine:Engine):
        '''
        Reset Alpha Engine.
        '''
        engine = self.engine

        self.__init__(engine, self.allocation, self.profit_perc, self.loss_perc)