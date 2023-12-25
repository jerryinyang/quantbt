import pandas as pd
import pytz

from dateutil.parser import parse
from typing import Dict
from utils import debug, clear_terminal # noqa

class DataLoader:
    DATE_FORMAT = '%Y-%m-%d'# %H:%M:%S'
    TZ = pytz.timezone('UTC')

    def __init__(self, dataframes:Dict[str,pd.DataFrame], resolution:str, start_date, end_date) -> None:
        self.resolution = resolution # TODO : Implement Resolution class to handle resolution resampling
        self.tickers = list(dataframes.keys())
        self.start_date = start_date
        self.end_date = end_date
        
        self.date_range = pd.date_range(
            start=parse(start_date, ignoretz=True), 
            end=parse(end_date, ignoretz=True), 
            freq=self.resolution, tz=self.TZ
        )

        self.dataframes = self._init_data(dataframes, self.date_range)

    
    def _init_date_range(self, dataframes:dict[str,pd.DataFrame]):
        start_date = parse('9999-12-12')
        end_date = parse('1111-01-01')

        # Find the full date raneg for the entire dataset
        for ticker, data in dataframes.items():
            # Get the start and end date for each ticker
            try:
                start = pd.to_datetime(data.index[0])
                end = pd.to_datetime(data.index[-1])
            except Exception:
                raise  ValueError(f'Unable to extract date from {ticker} data. Ensure that the index contains date/time values.')

            # Store the mimimum start and maximum end dates found
            start_date = min(start_date, start)
            end_date = max(end_date, end)

        # Return Date Range
        date_range = pd.date_range(start=parse(start_date, ignoretz=True), 
                                        end=parse(end_date, ignoretz=True), 
                                        freq=self.resolution, tz=self.TZ)
        
        return date_range


    def _init_data(self, dataframes:dict[str,pd.DataFrame], date_range):
        # Modify Dataframes to Desired Format
        for ticker, data in dataframes.items():    
            df = pd.DataFrame(index=date_range)
    
            data.index = pd.to_datetime(data.index.strftime(self.DATE_FORMAT)).tz_localize(self.TZ)
            data.rename(columns={old_column : old_column.lower() for old_column in list(data.columns)}, inplace=True)
            data = data[['open', 'high', 'low', 'close', 'volume']]

            df = df.join(data, how='left').ffill().bfill().fillna(0)
            # df['price_change'] = pd.Series(df['close']).diff().shift().fillna(0)
            df['price_change'] = (df['close'] - df['close'].shift()).fillna(0)     
    
            df['market_open'] = self._confirm_new_data(
                    df, 
                    ['open', 'high', 'low', 'close', 'volume']
                )
            df['market_open'] = df['market_open'].astype(int)

            dataframes[ticker] = df

        # TODO : Initial Universe Filter : Select the tickers to be included in the backtest
        # Example : For Volume Based Strategies, exclude Forex data, because they don't apply

        return dataframes
    

    def _confirm_new_data(self, df, columns):
        df_shifted = df[columns].shift(1)

        # Compare Each Column with the shift column value 
        # Returns True is the value is different from the previous value 
        df_different = df[columns].ne(df_shifted)

        # Filter out rows where all values are zero
        df_nonzero = df[~(df == 0).all(axis=1)]

        # Confirms if each row contains new data in any of the passed columns
        return df_different.any(axis=1) & df_nonzero.any(axis=1)


    def reset_dataloader(self, dataframes:dict[str,pd.DataFrame]):
        self.__init__(
             dataframes,
             self.resolution,
             self.start_date,
             self.end_date
        )

    
    # PICKLE-COMPATIBILITY
    def __getstate__(self):
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        # Customize the object reconstruction
        self.__dict__.update(state)