from datetime import datetime

class Bar:
    def __init__(self, open: float, high: float, low: float, close: float, volume: int, index: int, timestamp: int | datetime, resolution: str, ticker: str) -> None:
        """
        Represents a OHLCV bar with open, high, low, close prices, volume, index, timestamp, and resolution.

        Attributes:
        - open (float): The opening price of the bar.
        - high (float): The highest price reached during the bar period.
        - low (float): The lowest price reached during the bar period.
        - close (float): The closing price of the bar.
        - volume (int): The volume of trading activity during the bar period.
        - index (int): The index of the bar within a dataset or sequence.
        - timestamp (int | datetime): The timestamp representing the time when the bar occurred.
          It can be either an integer (Unix timestamp) or a datetime object.
        - resolution (str): The resolution or timeframe of the bar, such as '1D' for daily, '1H' for hourly, etc.

        Returns:
        - None: This constructor does not return any value.

        Example:
        >>> from datetime import datetime
        >>> bar = Bar(open=100.0, high=105.0, low=98.0, close=102.5, volume=10000, index=1, timestamp=datetime.now(), resolution='1D')
        >>> print(bar.open)
        100.0
        >>> print(bar.close)
        102.5
        >>> print(bar.timestamp)
        2023-11-22 12:30:45.678901
        """
        
        self.open = open
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume
        self.index = index
        self.timestamp = timestamp
        self.resolution = resolution
        self.ticker = ticker


    def fills_price(self, price:float):
        """
            Check if the given price is within the high and low values of the bar.
            
            Parameters:
            - price (float): The price to check.

            Returns:
            - bool: True if the price is within the bar, False otherwise.
        """
        return self.high >= price >= self.low
