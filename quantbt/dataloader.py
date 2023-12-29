import pandas as pd
import pytz

from dateutil.parser import parse
from typing import Dict
from utils import Resolution
from utils import debug, clear_terminal # noqa

class DataLoader:
    TZ = pytz.timezone('UTC')

    def __init__(self, dataframes:Dict[str,pd.DataFrame], start_date, end_date) -> None:

        self.start_date = start_date
        self.end_date = end_date

        self.tickers = None
        self.dataframes = None
        self.date_range = None

        self.dataframes = self._init_data(dataframes)


    def _init_data(self, dataframes:dict[str,pd.DataFrame]):
        '''
        Preprocess all data passed.
        '''

        # Preprocess Data
        dataframes = self._preprocess_data(dataframes=dataframes)

        # Create additional columns
        for ticker, data in dataframes.items():    
            df = pd.DataFrame(index=self.date_range)
    
            df = df.join(data, how='left').ffill().bfill().fillna(0)

            df['price_change'] = (df['close'] - df['close'].shift()).fillna(0)     
    
            df['market_open'] = self._confirm_unique_row(
                    df, 
                    ['open', 'high', 'low', 'close', 'volume']
                )
            df['market_open'] = df['market_open'].astype(int)

            dataframes[ticker] = df

        return dataframes


    def _preprocess_data(self, dataframes:dict[str,pd.DataFrame]):
        '''
        Preprocess all data passed.

        Returns:
            dataframe (Dict) : Dictionary containing the preprocessed data.
        '''
        resolutions = []

        # Modify Dataframes
        for ticker, data in dataframes.items():   
             
            # Change column names to lowercase
            data.columns = data.columns.str.lower()

            # Handle dataframes with wrong index (integer)
            if not isinstance(data.index, pd.DatetimeIndex) and data.index.dtype == int:

                # Check for Epoch index (index greater than January 1st, 2000)
                if pd.to_datetime(data.index[0]) > pd.to_datetime('2000-01-01'):
                    data.index = pd.to_datetime(data.index)

                elif 'date' in data.columns:
                    data.set_index('date', inplace=True)

                else:
                    continue
    
            # Make Index timezone-aware datetime 
            data.index = pd.to_datetime(data.index).tz_localize(self.TZ)

            # Keep neccessary columns
            data = data[['open', 'high', 'low', 'close', 'volume']]

            # Detect resolution
            resolution = DataLoader.detect_frequency(data)
            
            # If resolution is a multiple of daily
            if (resolution >= 1440) and (resolution % 1440 == 0):
                data.index = data.index.normalize()

            # Set the preprocessed data
            dataframes[ticker] = data
            resolutions.append(resolution)

        # Initialize self.resolution with the minimum resolution detected
        self.resolution = Resolution(min(resolutions)) if resolutions else Resolution('1D')

        # Initialize self.date_range
        self.date_range = pd.date_range(
            start=parse(self.start_date, ignoretz=True), 
            end=parse(self.end_date, ignoretz=True), 
            freq=self.resolution.name, tz=self.TZ)
        
        # Initialize self.tickers
        self.tickers = dataframes.keys()


        return dataframes
    

    def _confirm_unique_row(self, df, columns):
        '''
        Find rows with unique data.
        '''
        df_shifted = df[columns].shift(1)

        # Compare Each Column with the shift column value 
        # Returns True is the value is different from the previous value 
        df_different = df[columns].ne(df_shifted)

        # Filter out rows where all values are zero
        df_nonzero = df[~(df == 0).all(axis=1)]

        # Confirms if each row contains new data in any of the passed columns
        return df_different.any(axis=1) & df_nonzero.any(axis=1)


    def reset_dataloader(self, dataframes:dict[str,pd.DataFrame]):
        '''
        Re-initialize the dataloader with new a datasets.
        '''
        self.__init__(
             dataframes,
             self.start_date,
             self.end_date
        )


    @staticmethod
    def detect_frequency(data: pd.DataFrame):
        '''
        Estimates the datetime frequency from a dataframe.
        '''
        df = data.copy()

        # Calculate the difference between each date
        df['difference'] = df.index.diff().total_seconds() / 60

        # Most common difference can be a good estimate of the frequency
        estimated_frequency = df['difference'].mode().iloc[0]
        
        return int(round(estimated_frequency))
    

    # PICKLE-COMPATIBILITY
    def __getstate__(self):
        state = self.__dict__.copy()
        return state


    def __setstate__(self, state):
        # Customize the object reconstruction
        self.__dict__.update(state)