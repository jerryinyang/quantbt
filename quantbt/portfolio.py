import pandas as pd

from typing import Dict, List
from collections import OrderedDict, namedtuple

from orders import Order
from trades import Trade
from dataloader import DataLoader
from utils import ObservableDict as odict

class Portfolio:
    # Each record is identified by an index, and contains balance, equity and open_pnl values
    Record = namedtuple('Record', ['date', 'balance', 'equity', 'open_pnl'])

    def __init__(self, dataloader : DataLoader, capital : float) -> None:
        self.records = OrderedDict()
        self.df = self.init_portfolio(dataloader.date_range, dataloader.tickers, capital )

        self.trades : Dict[str, odict[int, Order]] = {key : odict(callback=self.onTrade) for key in self.tickers} # Syntax : Dict[ticker, Dict [ trade_id : Trade()]]
        self.history : List[Trade] = []


    def init_portfolio(self, date_range : pd.DatetimeIndex, tickers : List[str], capital : float):

        # Initialize Portfolio Dataframe: this would contain all the portfolio attributes
        portfolio =  pd.DataFrame({'timestamp': date_range}) # Initialize the full date range for the backtest
        portfolio.loc[0, 'balance'] = capital # Initialize the backtest capital (initial balance)
        portfolio.loc[0, 'equity'] =  capital # Initialize the equity
        portfolio.loc[0, 'open_pnl'] = 0

        for ticker in tickers:
            portfolio.loc[0, f'{ticker} units'] = 0
            portfolio.loc[0, f'{ticker} open_pnl'] = 0
            portfolio.loc[0, f'{ticker} closed_pnl'] = 0

        return portfolio
    
    
    def add_record(self, index, balance, equity, open_pnl):
        # Create an immutable Record namedtuple
        record = self.Record(balance=balance, equity=equity, open_pnl=open_pnl)
        # Add to the OrderedDict
        self.records[index] = record
