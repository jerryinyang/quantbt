import pandas as pd

from typing import List
from collections import OrderedDict, namedtuple

from dataloader import DataLoader
from utils import Logger

# Each record is identified by an index, and contains balance, equity and open_pnl values
Record = namedtuple('Record', ['date', 'balance', 'equity', 'open_pnl'])

class Portfolio:
    logger = Logger('logger_portfolio')


    def __init__(self, dataloader : DataLoader, capital : float) -> None:
        self.records = OrderedDict()
        self.dataframe : pd.DataFrame = self.init_portfolio(dataloader.date_range, dataloader.tickers, capital)


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

        # Initialize the self.records data as well
        self.add_record(index=0, 
                        date=portfolio.loc[0, 'timestamp'],
                        balance=capital,
                        equity=capital,
                        open_pnl=0)

        return portfolio
    
    
    def add_record(self, index, date, balance, equity, open_pnl):
        # Create an immutable Record namedtuple
        record = Record(date=date, balance=balance, equity=equity, open_pnl=open_pnl)
        # Add to the OrderedDict
        self.records[index] = record

    
    def get_record(self, lookback : int):
        # Check if there are records in the OrderedDict
        if self.records:
            
            # If lookback period is larger than the available data
            if lookback > len(self.records):
                self.logger.warning(f'Lookback period {lookback} exceeds available records.')
                return None

            # Access the record based on the lookback period
            record = list(self.records.values())[lookback - 1]

            return record
        else:
            # Return None or handle the case where there are no records
            return None
        

    def to_dataframe(self):
        records_list = []

        for index, record in self.records.items():
            record_dict = {
                'index': index,
                'date': record.date,
                'balance': record.balance,
                'equity': record.equity,
                'open_pnl': record.open_pnl,
            }

            # Add ticker-specific columns dynamically
            for ticker in self.dataframe.columns:
                if ticker.endswith(' units') or ticker.endswith(' open_pnl') or ticker.endswith(' closed_pnl'):
                    record_dict[ticker] = self.dataframe.loc[index, ticker]

            records_list.append(record_dict)

        dataframe = pd.DataFrame(records_list).set_index('index')
        return dataframe