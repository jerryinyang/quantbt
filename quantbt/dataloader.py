import pandas as pd

from dateutil.parser import parse

class DataLoader:
    def __init__(self, dataframes:dict[str,pd.DataFrame], resolution:str, start_date, end_date) -> None:
        self.resolution = resolution # TODO : Implement Resolution class to handle resolution resampling
        self.tickers = list(dataframes.keys())
        
        self.date_range = pd.date_range(start=parse(start_date, ignoretz=True), 
                                        end=parse(end_date, ignoretz=True), 
                                        freq=self.resolution, tz=self.TZ)

        self.dataframes = self._prepare_data(dataframes, self.date_range)


    def _prepare_data(self, dataframes:dict[str,pd.DataFrame], date_range):

        # Modify Dataframes to Desired Format
        for ticker, data in dataframes.items():
            df = pd.DataFrame(index=date_range)
    
            data.index = pd.to_datetime(data.index.strftime(self.DATE_FORMAT)).tz_localize(self.TZ)
            data.rename(columns={old_column : old_column.lower() for old_column in list(data.columns)}, inplace=True)
            data = data[['open', 'high', 'low', 'close', 'volume']]

            df = df.join(data, how='left').ffill().bfill()
            df['price_change'] = df['close'] - df['close'].shift().fillna(0)
            dataframes[ticker] = df

            df['market_open'] = self._confirm_new_data(
                    df, 
                    ['open', 'high', 'low', 'close', 'volume']
                )
            df['market_open'] = df['market_open'].astype(int)

        # TODO : Initial Universe Filter : Select the tickers to be included in the backtest
        # Example : For Volume Based Strategies, exclude Forex data, because they don't aaply

        return dataframes
    

    def _confirm_new_data(self, df, columns):
            df_shifted = df[columns].shift(1)
    
            # Compare Each Column with the shift column value; Returns True is the value is different from the previous value 
            df_different = df[columns].ne(df_shifted)
            
            # Confirms if each row contains new data in any of the passed columns
            return df_different.any(axis=1)
