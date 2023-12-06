from datetime import datetime  # , timedelta
import logging
import os
# import math
from bisect import bisect_right

# from enum import Enum
# import re
# from typing import Any


class Bar:
    def __init__(
        self,
        open: float,
        high: float,
        low: float,
        close: float,
        volume: int,
        index: int,
        timestamp: int | datetime,
        resolution: str,
        ticker: str,
    ) -> None:
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

    def __repr__(self) -> str:
        return f"Bar(open={self.open}, high={self.high}, low={self.low}, \
                close={self.close}, volume={self.volume}, index={self.index}, \
                    timestamp={self.timestamp}, resolution='{self.resolution}', \
                        ticker='{self.ticker}')"

    def fills_price(self, price: float):
        """
        Check if the given price is within the high and low values of the bar.

        Parameters:
        - price (float): The price to check.

        Returns:
        - bool: True if the price is within the bar, False otherwise.
        """
        return self.high >= price >= self.low


class LogFormatter(logging.Formatter):
    def format(self, record):
        level = record.levelname
        if level in ['DEBUG', 'INFO']:
            self._style._fmt = '%(asctime)s - %(levelname)s [%(name)s] - %(message)s'
        elif level in ['WARNING', 'ERROR']:
            self._style._fmt = '%(asctime)s - %(levelname)s [%(name)s] \n %(message)s - %(pathname)s:%(lineno)d'
        return super().format(record)


class Logger(logging.Logger):

    def __init__(self, name=__name__, level=logging.NOTSET):
        super().__init__(name, level)

        # Create and set the handler with the custom formatter
        custom_handler = logging.StreamHandler()
        custom_handler.setLevel(logging.DEBUG)  # Set the level to the lowest level you want to capture
        custom_handler.setFormatter(LogFormatter())

        # Create and set the FileHandler with the custom formatter
        file_handler = logging.FileHandler('logs.log')
        file_handler.setLevel(logging.DEBUG)  # Set the level for file logging
        file_handler.setFormatter(LogFormatter())
        self.addHandler(file_handler)

        # Add the handler to the logger
        self.addHandler(custom_handler)


class ObservableList(list):
    def __init__(self, callback, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._callback = self._set_callback(callback)

    def _set_callback(self, callback):
        assert callback, ValueError('Callback Function cannot be reset to None.')
        return callback

    def append(self, value):
        self._on_update(value)
        super().append(value)

    def extend(self, values):
        self._on_update(values)
        super().extend(values)

    def __setitem__(self, index, value):
        self._on_update(value)
        super().__setitem__(index, value)

    def remove(self, value):
        self._on_update(value)
        super().remove(value)

    def pop(self, index=-1):
        result = super().pop(index)
        self._on_update(result)
        return result

    def insert(self, index, value):
        self._on_update(value)
        super().insert(index, value)

    # To be called when olist is updated
    def _on_update(self, changed_object):
        if self._callback:
            self._callback(changed_object)


class ObservableDict(dict):
    def __init__(self, callback, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._callback = self._set_callback(callback)

    def _set_callback(self, callback):
        assert callback, ValueError('Callback Function cannot be reset to None.')
        return callback

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        self._on_update(value)

    def __delitem__(self, key):
        value = self[key]
        super().__delitem__(key)
        self._on_update(value)

    # To be called when olist is updated
    def _on_update(self, changed_object):
        if self._callback:
            self._callback(changed_object)



# Resolution('1m')
def clear_terminal():
        # Check the operating system
        if os.name == "posix":  # For Linux and macOS
            os.system("clear")


def debug(texts):
    if not isinstance(texts, list):
        texts = [texts]
        
    display_text = "\n".join([str(text) for text in texts])
    print(display_text)
    x = input(" " )

    if x == 'x':
        clear_terminal()
        
        file_path = "logs.log"
        with open(file_path, 'w'):
            pass

        exit()
    

def sorted_index(array, value):
    # Inserts an item into an array in ascending order

    # Type Checking
    if not (value and isinstance(array, list)):
        return 
    
    # Empty Array
    if not array:
        return 0
    
    # Return the right index to keep the array sorted
    return bisect_right(array, value)


def test(test):
    print(f'Yepp : {test}')

    
    

    