import json
import logging
import pandas as pd
import yfinance as yf

from abc import ABC, abstractmethod
from binance import Client
from datetime import datetime, timedelta
from dateutil.parser import parse
from dateutil.relativedelta import relativedelta
from dotenv import load_dotenv, dotenv_values
from pathlib import Path


# Load Environment Variables
load_dotenv()
config = dotenv_values(".env")

# Import Environment Variables
binance_key = config.get("BINANCE_API_KEY")
binance_secret = config.get("BINANCE_SECRET_KEY")


class TickerData(ABC):
    # CLASS CONSTANT ATTRIBUTES
    DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
    MARKET_TYPES = ["crypto", "forex", "futures", "stocks"]
    OHLC_COLUMNS = [
        "ticker_id",
        "resolution",
        "market_type",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]
    RESOLUTIONS = {
        key: key
        for key in ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "1d", "1w", "1M"]
    }

    def __init__(
        self, symbol: str | list[str], resolution: str, market: str, **period_args
    ) -> None:
        assert (isinstance(market, str)) and (
            market.lower() in self.MARKET_TYPES
        ), f"Unrecognized market : {market}"

        # Set the timeframe for the data
        self.symbol = self.__set_symbol(symbol)
        self.resolution = self.__set_resolution(resolution)
        self.market_type = market.lower()

        self.start_date, self.end_date = self.__set_data_range(period_args)

        # OHLC DataFrames
        self.raw_dataframe = pd.DataFrame()
        self.__dataframe = pd.DataFrame()

        # Load the Data
        self.load_data()

    @property  # Read Only Getter for self.dataframe
    def dataframe(self):
        return self.__dataframe

    @dataframe.setter  # Setter for self.dataframe
    def dataframe(self, data: pd.DataFrame) -> None:
        if not self.data_loaded():
            self.__dataframe = self.parse_data(data)

    @abstractmethod
    def fetch_data(self):
        # Must be over-ridden, to populate the symbol dataframe
        return None

    def load_data(self):
        """
        This runs the process of fetching, parsing data, and storing it to self.data_frame.
        """
        try:
            # Download (fetch) the data, and assign it to self.raw_data
            self.fetch_data()

            # Assign the data to self.dataframe
            self.dataframe = self.raw_dataframe

            loaded = self.data_loaded()
            if loaded:
                self.raw_dataframe = pd.DataFrame()

            return loaded

        except Exception as e:
            logging.error(f"Error Loading Data: {e}")
            raise e

    def parse_data(self, data: pd.DataFrame = None):
        """
        Ensures the presence of data in the raw DataFrame, fetches data if needed,
        and performs necessary preprocessing steps.

        Raises a ValueError if data cannot be fetched after a specified number of attempts.

        Parameters:
            - self: The instance of the class containing the raw data and relevant attributes.
            - data: Optional parameter for passing an existing DataFrame.

        Returns:
            pd.DataFrame: Processed DataFrame.

        Raises:
            ValueError: If data cannot be fetched after the specified number of attempts or if
                        required columns are missing in the DataFrame.
            Exception: Any other exceptions encountered during the process are logged and re-raised.
        """

        try:
            # If data parameter is not provided, attempt to fetch data up to 5 times
            if data is None:
                for attempts in range(6):
                    if self.__has_data():
                        break
                    else:
                        logging.warning("Dataframe is empty. Fetching data now...")
                        self.fetch_data()

                if not self.__has_data():
                    logging.error("Unable to fetch data.")
                    raise ValueError("Unable to fetch data.")

                # Assign raw DataFrame to the local variable
                data = self.raw_dataframe.copy()

            # Rename all columns to lowercase
            data.rename(
                columns={col: col.lower() for col in data.columns}, inplace=True
            )

            # Rename possible 'ticker_id' column
            data.rename(
                columns={
                    col: "ticker_id"
                    for col in ["symbol", "ticker", "symbols", "tickers"]
                    if col in data.columns
                },
                inplace=True,
            )

            # Set the 'timestamp' column
            data["timestamp"] = data.index.strftime(TickerData.DATE_FORMAT)

            # Ensure all the expected columns are present.
            while not all(column in data.columns for column in self.OHLC_COLUMNS):
                missing_columns = set(self.OHLC_COLUMNS) - set(data.columns)

                # If 'market_type' is missing, assign the specified value and check again.
                if "market_type" in missing_columns:
                    data["market_type"] = self.market_type

                # If 'resolution' is missing, assign the specified value and check again.
                if "resolution" in missing_columns:
                    data["resolution"] = self.resolution

                else:
                    # If other columns are missing, raise a ValueError.
                    error_message = f"Missing columns: {', '.join(missing_columns)}. Cannot set TickerData.dataframe."
                    logging.error(error_message)
                    raise ValueError(error_message)

            # Assign expected data types to the columns
            desired_dtypes = {
                "ticker_id": "object",
                "resolution": "object",
                "market_type": "object",
                "open": "float",
                "high": "float",
                "low": "float",
                "close": "float",
                "volume": "int",
            }
            data = data.astype(desired_dtypes)

            # Drop rows with any empty cells
            data.dropna(axis=0, inplace=True)

            return data

        except Exception as e:
            logging.error(f"Error parsing data: {e}")
            raise e

    def store_data(self):
        assert self.data_loaded(), ValueError("Object does not have data to be stored")

        try:
            self.DATABASE = pd.concat(
                [self.DATABASE, self.dataframe], ignore_index=True
            )
            self.DATABASE = self.DATABASE.drop_duplicates()

        except Exception as e:
            logging.error(f"Error moving data into the main database: {e}")
            raise e

    def __set_data_range(self, period_args: dict = {}) -> tuple:
        """
        Set the data range based on the given period arguments, start date, and end date.

        Args:
            period_args (dict): Dictionary containing period information along with start_date and end_date.

        Returns:
            tuple: A tuple containing formatted start and end dates.
        """
        # Check if both start_date and end_date are provided in period_args
        if "start_date" in period_args and "end_date" in period_args:
            start_date = parse(period_args["start_date"])
            end_date = parse(period_args["end_date"])
        else:

            # Default start_date and end_date
            start_date = period_args.get("start_date")
            end_date = period_args.get("end_date")

            # Check if start_date and end_date are valid datetime objects, otherwise use default values.
            if not isinstance(start_date, datetime):
                start_date = (
                    parse(start_date) if start_date else datetime.now() - timedelta(days=30)
                )

            if not isinstance(end_date, datetime):
                end_date = parse(end_date) if end_date else datetime.now()

            # Define the empty period
            duration = {
                "seconds": 0,
                "minutes": 0,
                "hours": 0,
                "days": period_args.get("days", 30),
                "weeks": 0,
                "months": 0,
                "years": 0,
            }

            # Update the duration with only elements that are found in both dictionaries
            duration = {
                key: period_args[key] for key in duration.keys() & period_args.keys()
            }

            # Checks if the default duration has been updated by the passed period arguments
            period_updated = any(value > 0 for value in duration.values())

            # If start_date and end_date were passed and parsed successfully.
            if period_updated:
                if start_date:
                    end_date = start_date + relativedelta(**duration)
                elif end_date:
                    start_date = end_date - relativedelta(**duration)

        return start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")

    def __set_symbol(self, symbol: str | list[str]):
        """Set the symbol for your class.

        Args:
            symbol (str | list[str]): The symbol or list of symbols.

        Returns:
            str | list[str]: The processed symbol or list of symbols.

        Raises:
            ValueError: If the passed symbol is None.
            TypeError: If the data type of the symbol is unsupported.
        """

        if symbol is None:
            error_message = "Passed symbol cannot be None"
            logging.warning(error_message)
            raise ValueError(error_message)
        if isinstance(symbol, str):
            return symbol.upper()
        elif isinstance(symbol, list):
            return [str_.upper() for str_ in symbol]
        else:
            error_message = f"Unsupported data type for symbol: {type(symbol)}"
            logging.warning(error_message)
            raise TypeError(error_message)

    def __set_resolution(self, resolution: str) -> str:
        """Set the resolution for Binance data.

        Args:
            resolution (str): The desired resolution.

        Returns:
            str: The resolved resolution.

        Raises:
            UserWarning: If the resolution is not recognized, defaults to '1d'.
        """
        resolutions = self.RESOLUTIONS
        default_resolution = "1d"

        if resolution not in resolutions:
            warning_msg = f'Unsupported resolution: "{resolution}" is not recognized. Defaulting to "{default_resolution}".'
            logging.warning(warning_msg)
            return default_resolution
        else:
            return resolutions[resolution]

    def __has_data(self):
        return not self.raw_dataframe.empty

    def data_loaded(self):
        return not self.dataframe.empty


class BinanceData(TickerData):

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

    Methods:
    - download(ticker='') -> tuple: Download historical price data for a specific symbol or the default symbol.
    - download_bulk(symbols: list[str]) -> tuple: Download historical price data for multiple symbols.
    """

    MARKET = "crypto"

    def __init__(self, symbol: str | list[str], resolution: str, **period_args) -> None:
        self.client = self.__init_client()
        super().__init__(symbol, resolution.lower(), BinanceData.MARKET, **period_args)

    def __init_client(self):
        # Initialize Binance Client
        client = Client(binance_key, binance_secret)

        self.RESOLUTIONS = {
            "1m": client.KLINE_INTERVAL_1MINUTE,
            "3m": client.KLINE_INTERVAL_3MINUTE,
            "5m": client.KLINE_INTERVAL_5MINUTE,
            "15m": client.KLINE_INTERVAL_15MINUTE,
            "30m": client.KLINE_INTERVAL_30MINUTE,
            "1h": client.KLINE_INTERVAL_1HOUR,
            "2h": client.KLINE_INTERVAL_2HOUR,
            "4h": client.KLINE_INTERVAL_4HOUR,
            "1d": client.KLINE_INTERVAL_1DAY,
            "1w": client.KLINE_INTERVAL_1WEEK,
            "1M": client.KLINE_INTERVAL_1MONTH,
        }

        return client

    def __download_bulk(self, symbols: list[str]) -> pd.DataFrame:
        dataframes = [self.__download(symbol.upper()) for symbol in symbols]

        return pd.concat(dataframes, axis=0)

    def __download(self, symbol: str) -> pd.DataFrame:
        if not symbol:  # symbol is None or ''
            logging.warning("Object symbol cannot be None or an empty string.")
            raise ValueError("Object symbol cannot be None or an empty string.")

        symbol = symbol.upper()

        if not self.symbol:  # if object.symbol isn't also initialised
            logging.warning(
                "Unspecified Symbol: You have not specified a symbol/asset. Symbol has been set to 'BTCUSDT'."
            )
            self.symbol = symbol

        klines = self.client.get_historical_klines(
            symbol, self.resolution, self.start_date, self.end_date
        )

        # Parse kline data into pandas dataframe
        data = (
            pd.DataFrame(
                klines,
                columns=[
                    "time",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "_",
                    "_",
                    "_",
                    "_",
                    "_",
                    "_",
                ],
            )
            .set_index("time")
            .astype(
                {
                    "open": "float",
                    "high": "float",
                    "low": "float",
                    "close": "float",
                    "volume": "float",
                }
            )
            .drop(["_"], axis=1)
        )

        # Set the `symbol` column
        data["symbol"] = symbol
        data.index = pd.to_datetime(data.index, unit="ms")

        return data


    def fetch_data(self):
        if isinstance(self.symbol, str):
            # If the input is a string, call the download method
            self.raw_dataframe = self.__download(self.symbol)
        elif isinstance(self.symbol, list):
            # If the input is a list, call the download_bulk method
            self.raw_dataframe = self.__download_bulk(self.symbol)
        else:
            # Handle other types or raise an exception
            error_message = f"Unsupported symbol type: type{self.symbol}"
            logging.warning(error_message)
            raise ValueError(error_message)


class YFinanceData(TickerData):
    RESOLUTIONS = {
        "1m": "1m",
        "5m": "5m",
        "15m": "15m",
        "30m": "30m",
        "1h": "1h",
        "1d": "1d",
        "1w": "1wk",
        "1M": "1mo",
    }

    def __init__(
        self, symbol: str, resolution: str, market: str, **period_args
    ) -> None:
        symbol = (
            f"{symbol}=x"
            if market == "forex"
            else f"{symbol}=f"
            if market == "futures"
            else symbol
        )
        super().__init__(symbol, resolution, market, **period_args)

    def fetch_data(self):
        try:
            data = yf.download(
                self.symbol,
                start=self.start_date,
                end=self.end_date,
                interval=self.resolution,
            )
            data.index = pd.to_datetime(data.index)

            if data.empty:
                error_message = (
                    "WARNING: Fetch Data Unsuccesful. Object Dataframe did not recieve any data."
                    + " Ensure the symbol(s) are valid, and the start/end dates are allowed for that resolution."
                )
                logging.warning(error_message)
                raise ValueError(error_message)

            self.raw_dataframe = self.__reorder_columns(data)

        except Exception as e:
            logging.warning(f"Fetch Data Unsuccesful: {e}")
            raise e

    def __reorder_columns(self, data: pd.MultiIndex):
        """
        Reorders a MultiIndex DataFrame by stacking one level and resetting the other.

        Parameters:
        - data (pd.DataFrame): Input DataFrame with a MultiIndex.

        Returns:
        pd.DataFrame: Reordered DataFrame with a 'symbol' column.

        Raises:
        ValueError: If the input DataFrame does not have a MultiIndex.
        """

        # Check if the input DataFrame has a MultiIndex
        if not isinstance(data.index, pd.MultiIndex):
            data["symbol"] = self.symbol
            return data

        # Stack one level of the DataFrame and rename the resulting axis
        stacked = data.stack(level=1).rename_axis(index=[data.index.name, "symbol"])

        # Reset the 'symbol' level to become a regular column
        stacked.reset_index(level="symbol", inplace=True)

        return stacked


class DataBentoData(TickerData):
    def __init__(self, file_location: str | Path, market: str) -> None:
        self.market = market

        self.file_location = file_location
        # Read the metadata.json file
        [
            self.file_extension,
            resolution,
            symbol,
            start_date,
            end_date,
        ] = self.__read_metadata(file_location)

        # Initialize the TickerData
        try:
            super().__init__(
                symbol,
                resolution,
                self.market,
                start_date=start_date,
                end_date=end_date,
            )
        except Exception as e:
            logging.warning("Failed to initialize the TickerData (parent)")
            raise e

    def fetch_data(self):
        # Read the OHLC data files
        try:
            data = self.__read_data_files(self.file_location, self.file_extension)
            data.index = pd.to_datetime(data.index)

            if data.empty:
                error_message = (
                    "WARNING: Fetch Data Unsuccesful. Object Dataframe did not recieve any data."
                    + "Check the filepath provided."
                )
                logging.warning(error_message)
                raise ValueError(error_message)

            self.raw_dataframe = self.__clean_data(data)

        except Exception as e:
            logging.warning(f"Fetch Data Unsuccesful: {e}")
            raise e

    def __read_metadata(self, file_location: str | Path):
        # Open the directory from the string or path
        directory = Path(file_location)

        # Read the metadata.json file in the directory, or raise an Error if it doesn't exist
        metadata_file = directory / "metadata.json"
        if not metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found in {directory}")

        # Fetch relevant data in the metadata.json file
        with open(metadata_file, "r") as file:
            metadata = json.load(file)

        query = metadata.get("query", {})

        # Parse/modify the collected data
        schema = query.get("schema", "")
        filetype = query.get("encoding", "")
        resolution = schema.split("-")[-1] if schema else ""
        symbol = (
            query.get("symbols", [""])[0].split(".")[0] if query.get("symbols") else ""
        )
        start_date = datetime.utcfromtimestamp(
            query.get("start", 0) / 1e9
        )  # convert nanoseconds to seconds
        end_date = datetime.utcfromtimestamp(
            query.get("end", 0) / 1e9
        )  # convert nanoseconds to seconds

        # Remove anything that follows the '.' in the symbol
        symbol = symbol.split(".")[0]

        file_extension = f".{schema}.{filetype}"

        return file_extension, resolution, symbol, start_date, end_date

    def __read_data_files(self, file_path: str | Path, file_extension: str):
        # Convert the input to a Path object if it's a string
        file_path = Path(file_path) if isinstance(file_path, str) else file_path

        # Get a list of all files in the directory with the specified file type
        files = list(file_path.glob(f"*{file_extension}"))

        # Check if there are any matching files
        if not files:
            print(f"No {file_extension} files found in the directory.")
            return None

        # Initialize an empty list to store DataFrames
        dataframes = []

        # Iterate through each file and read it into a DataFrame
        for file in files:
            try:
                # Assuming CSV files have headers, if not, set header=None
                dataframes.append(pd.read_csv(file))
            except Exception as e:
                print(f"Error reading file {file}: {e}")

        # Concatenate all DataFrames into a single DataFrame
        data = pd.concat(dataframes, ignore_index=True)

        return data

    def __clean_data(self, data :pd.DataFrame):
        
        data.index = pd.to_datetime(data['ts_event'])
        data.rename(columns={'ts_event' : 'time'})
        
        # Drop Unnecesary Columns
        data = data.loc[:, ['open', 'high', 'low', 'close', 'volume', 'symbol']]

        # Normalize the OHLC data 
        data[['open', 'high', 'low', 'close']] /= 1e9

        # Drop Columns whose 'symbol' column contains a '-'
        data = data[~data['symbol'].str.contains('-')]

        return data
    

if __name__ == "__main__":
    # ethusdt = BinanceData(['ETHUSDT', 'BTCUSDT', 'CELOUSDT'], '1h', days=30)
    # ethusdt.download()
    # print(ethusdt.data.index.dtype)

    # ethusdt = BinanceData('btcusdt', '10', days=20)
    # print(ethusdt.has_data())

    # aapl = YFinanceData('aapl', '1d', 'futures', days=20)
    # aapl.fetch_data()
    # print(aapl.has_data())

    # Download Bulk Data
    symbols = ['BTCUSDT', 'ETHUSDT', 'GMTUSDT', 'CELOUSDT', 'DOGEUSDT', 'SOLUSDT']

    for symbol in symbols:
        _ = BinanceData(symbol, '1h', years=6)
        _.fetch_data()

        data = _.raw_dataframe
        data.to_parquet(f'/Users/jerryinyang/Code/quantbt/data/prices/{symbol}.parquet')

    # ethusdt = BinanceData('ETHUSDT', '1h', days=30)

    # DataBentoData('/Users/jerryinyang/databento/GLBX-20231115-7N8E9R8WKG')
    pass
