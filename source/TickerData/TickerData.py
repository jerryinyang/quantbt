import pandas as pd
import warnings

from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

from binance import Client
from dotenv import load_dotenv, dotenv_values

# Load Environment Variables
load_dotenv()
config = dotenv_values('.env')

# Import Environment Variables
binance_key = config.get("API_KEY")
binance_secret = config.get("SECRET_KEY")

# Initialize Binance Client
binance_client = Client(binance_key, binance_secret)


class Ticker():
    DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

    def __init__(self) -> None:
        # self.symbol = symbol
        # self.market = market
        # self.resolution = resolution
        # self.start_date = start_date
        # self.end_date = end_date
        # self.dataframe = data.
        pass

    def parse_data(self):
        pass
    
    def save_data(self):
        pass

    def load_data(self):
        pass

# BinanceData Class
class BinanceData(Ticker):
    """
    BinanceData class for downloading historical price data from Binance.

    Parameters:
    - symbol (str): The trading pair symbol (e.g., 'BTCUSDT').
    - resolution (str): The resolution for the data (e.g., '1h', '15m').
    - **kwargs: Additional keyword arguments for setting data range (optional).

    Recognized Parameters in **kwargs:
    - 'seconds' (int): Number of seconds for data range.
    - 'minutes' (int): Number of minutes for data range.
    - 'hours' (int): Number of hours for data range.
    - 'days' (int): Number of days for data range.
    - 'weeks' (int): Number of weeks for data range.
    - 'months' (int): Number of months for data range.
    - 'years' (int): Number of years for data range.

    Attributes:
    - resolution (str): The set resolution for the data.
    - symbol (str): The trading pair symbol in uppercase.
    - extras (dict): Additional keyword arguments.
    - start_date (str): The start date for the data range.
    - end_date (str): The end date for the data range.
    - data (pd.DataFrame): The downloaded historical price data.
    - data_name (str): The generated name for the downloaded data.

    Methods:
    - download(ticker='') -> tuple: Download historical price data for a specific symbol or the default symbol.
    - download_bulk(symbols: list[str]) -> tuple: Download historical price data for multiple symbols.
    """
    
    def __init__(self, symbol:str, resolution:str, **period_args) -> None:
        # Set the tiemframe for the data
        self.symbol = symbol.upper()
        self.resolution = self.__set_resolution(resolution)
        
        self.period_args = period_args
        
        # Default Data Range is 5 days to current time
        self.start_date, self.end_date = self.__set_data_range(period_args)

        self.data = pd.DataFrame()
        self.data_name = ''
    
    def __generate_name(self, symbol: str, resolution: str, start_date: str, end_date: str) -> str:
        start = datetime.strptime(start_date, super().DATE_FORMAT)
        end = datetime.strptime(end_date, super().DATE_FORMAT)

        return f'{symbol}|{resolution}|{start:%d-%b-%y}|{end:%d-%b-%y}'

    def __set_data_range(self, period_args: dict = {}, start_date: datetime = None, end_date: datetime = None) -> tuple:
        # Default start_date and end_date
        start_date = start_date or (datetime.now() - timedelta(days=5))
        end_date = end_date or datetime.now()

        # Define the empty period
        duration = {
            'seconds': 0,
            'minutes': 0,
            'hours': 0,
            'days': 0,
            'weeks': 0,
            'months': 0,
            'years': 0,
        }

        # Update the duration with only elements that are found in both dictionaries
        duration = {key: period_args[key] for key in duration.keys() & period_args.keys()}

        # Checks if the default duration has been updated by the passed period arguments
        period_updated = any(value > 0 for value in duration.values())

        # If not updated, set the default duration to 30 days
        if not period_updated:
            duration.update({'days': 30})

        # Set start_date
        start = (end_date - relativedelta(**duration)) if period_updated else start_date

        return start.strftime(super().DATE_FORMAT), end_date.strftime(super().DATE_FORMAT)
    
    def __set_resolution(self, resolution:str):
        resolutions = {
            '1m': binance_client.KLINE_INTERVAL_1MINUTE,
            '3m': binance_client.KLINE_INTERVAL_3MINUTE,
            '5m': binance_client.KLINE_INTERVAL_5MINUTE,
            '15m': binance_client.KLINE_INTERVAL_15MINUTE,
            '30m': binance_client.KLINE_INTERVAL_30MINUTE,
            '1h': binance_client.KLINE_INTERVAL_1HOUR,
            '2h': binance_client.KLINE_INTERVAL_2HOUR,
            '4h': binance_client.KLINE_INTERVAL_4HOUR,
            '6h': binance_client.KLINE_INTERVAL_6HOUR,
            '8h': binance_client.KLINE_INTERVAL_8HOUR,
            '12h': binance_client.KLINE_INTERVAL_12HOUR,
            '1d': binance_client.KLINE_INTERVAL_1DAY,
            '3d': binance_client.KLINE_INTERVAL_3DAY,
            '1w': binance_client.KLINE_INTERVAL_1WEEK,
            '1M': binance_client.KLINE_INTERVAL_1MONTH,
        }
        
        if resolution not in resolutions:
            warnings.warn(f'Unsupported resolution : "{resolution}" is not a recognized resolution. Resolution has been changed to 1-hour.', UserWarning)
            return resolutions['1h']
        else:
            return resolutions[resolution]   

    def __download_bulk(self, symbols: list[str]) -> tuple:
        datas = [self.__download(symbol.upper()) for symbol in symbols]

        dataframes, datanames = zip(*datas)

        datanames = self.__generate_name('|'.join([symbol.upper() for symbol in symbols]), self.resolution, self.start_date, self.end_date)

        return pd.concat(dataframes, axis=0), datanames

    def __download(self, ticker: str = '') -> tuple:
        if not ticker:
            if not self.symbol:
                self.symbol = "BTCUSDT"
                warnings.warn("Unspecified Symbol: You have not specified a symbol/asset. Symbol has been set to 'BTCUSDT'.")
            symbol = self.symbol
        else:
            symbol = ticker.upper()

        klines = binance_client.get_historical_klines(symbol, self.resolution, self.start_date, self.end_date)

        # Parse kline data into pandas dataframe
        data = pd.DataFrame(
            klines, columns=['time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'number_of_trades', '_', '_', '_', '_'], 
        ).set_index('time').astype({'open': 'float', 'high': 'float', 'low': 'float', 'close': 'float', 'volume': 'float'}).drop(['_'], axis=1)

        data['symbol'] = symbol
        data.index = pd.to_datetime(data.index) # Changed : removed the additional 1 millisecond

        data_name = self.__generate_name(symbol, self.resolution, self.start_date, self.end_date)

        # Set the object values, if only one asset is downloaded
        if not ticker:
            self.data = data
            self.data_name = data_name

        return data, data_name
    
    def fetch_data(self, symbols):
        if isinstance(symbols, str):
            # If the input is a string, call the download method
            return self.__download(symbols)
        elif isinstance(symbols, list):
            # If the input is a list, call the download_bulk method
            return self.__download_bulk(symbols)
        else:
            # Handle other types or raise an exception
            raise ValueError("Unsupported parameter type for Binance download. Please provide a string or a list.")

if __name__ == '__main__':
    # ethusdt = BinanceData('ETHUSDT', '1h', days=30)
    # ethusdt.download()
    # print(ethusdt.data.index.dtype)

    # Download Bulk Data
    data, tickers = BinanceData('', '1d', years=5).fetch_data(['BTCUSDT', 'ETHUSDT', 'GMTUSDT'])
    print(data.columns, '\n\n\n', tickers)
