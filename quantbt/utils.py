import logging
import os
import string
import random

from bisect import bisect_right
from datetime import datetime 
from enum import Enum
from typing import Literal, Union


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
    
    # PICKLE-COMPATIBILITY
    def __getstate__(self):
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        # Customize the object reconstruction
        self.__dict__.update(state)


class LogFormatter(logging.Formatter):
    def format(self, record):
        level = record.levelname
        if level in ['DEBUG', 'INFO']:
            self._style._fmt = '%(asctime)s - %(levelname)s [%(name)s] - %(message)s \n'
        elif level in ['WARNING', 'ERROR']:
            self._style._fmt = '%(asctime)s - %(levelname)s [%(name)s] \n %(message)s - %(pathname)s:%(lineno)d \n'
        return super().format(record)


class Logger(logging.Logger):

    def __init__(self, name=__name__, file_location : str = 'logs.log', level=logging.DEBUG):
        super().__init__(name, level)

        # Create and set the handler with the custom formatter
        custom_handler = logging.StreamHandler()
        custom_handler.setLevel(logging.WARNING)  # Set the level to the lowest level you want to capture
        custom_handler.setFormatter(LogFormatter())

        # Create and set the FileHandler with the custom formatter
        file_handler = logging.FileHandler(file_location)
        file_handler.setLevel(logging.INFO)  # Set the level for file logging
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
        if hasattr(self, '_callback') and self._callback:
            self._callback(changed_object)

    # PICKLE-COMPATIBILITY
    def __getstate__(self):
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        # Customize the object reconstruction
        self.__dict__.update(state)


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

    def pop(self, key, default=None):
        value = super().pop(key, default)
        self._on_update(value)
        return value

    # To be called when olist is updated
    def _on_update(self, changed_object):
        if hasattr(self, '_callback') and self._callback:
            self._callback(changed_object)

    # PICKLE-COMPATIBILITY
    def __getstate__(self):
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        # Customize the object reconstruction
        self.__dict__.update(state)


class Source:
    def __init__(self, source:Literal['open', 'high', 'low', 'close', 'hl2', 'hlc3', 'hlcc4', 'ohlc4']) -> None:
        if source.lower() not in ['open', 'high', 'low', 'close', 'hl2', 'hlc3', 'hlcc4', 'ohlc4']:
            UserWarning(f"{source} is not a valid value for the source argument. Defaulting to 'close'.")
            source = 'close'
        
        self.source = source.lower()
    
    def __call__(self, bar:Bar) -> float:
        source = self.source

        if not isinstance(bar, Bar):
            # If integer or float value is passed, return the same value
            if isinstance(bar, int) or isinstance(bar, float):
                return bar # If 
            
            raise ValueError("Passed 'bar' argument is not a Bar object.")
        
        if not isinstance(source, str):
            raise ValueError("Passed 'source' argument is not valid.")
        
        if source in ['open', "Open"]:
            return bar.open
        
        elif source in ['high', 'High']:
            return bar.high
        
        elif source in ['low', 'Low']:
            return bar.low
        
        elif source in ['close', 'Close']:
            return bar.close
        
        elif source == 'hl2':
            return (bar.high  + bar.low) / 2
        
        elif source == 'hlc3':
            return (bar.high  + bar.low + bar.close) / 3
        
        elif source == 'hlcc4':
            return (bar.high  + bar.low + bar.close + bar.close) / 4
        
        elif source == 'ohlc4':
            return (bar.open + bar.high + bar.low + bar.close) / 4


class Timeframe(Enum):
    TIMEFRAME_1M = (1, '1min')
    TIMEFRAME_5M = (5, '5min')
    TIMEFRAME_15M = (15, '15min')
    TIMEFRAME_30M = (30, '30min')
    TIMEFRAME_1H = (60, '1H')
    TIMEFRAME_4H = (240, '4H')
    TIMEFRAME_1D = (1440, '1D')
    TIMEFRAME_1W = (7 * 1440, '1W')


class Resolution:
    RESOLUTION_MAP = {
        '1': Timeframe.TIMEFRAME_1M, '1M': Timeframe.TIMEFRAME_1M, 'M1': Timeframe.TIMEFRAME_1M, '1MIN': Timeframe.TIMEFRAME_1M,
        '5': Timeframe.TIMEFRAME_5M, '5M': Timeframe.TIMEFRAME_5M, 'M5': Timeframe.TIMEFRAME_5M, '5MIN': Timeframe.TIMEFRAME_5M,
        '15': Timeframe.TIMEFRAME_15M, '15M': Timeframe.TIMEFRAME_15M, 'M15': Timeframe.TIMEFRAME_15M, '15MIN': Timeframe.TIMEFRAME_15M,
        '30': Timeframe.TIMEFRAME_30M, '30M': Timeframe.TIMEFRAME_30M, 'M30': Timeframe.TIMEFRAME_30M, '30MIN': Timeframe.TIMEFRAME_30M,
        '60': Timeframe.TIMEFRAME_1H, '1H': Timeframe.TIMEFRAME_1H, 'H1': Timeframe.TIMEFRAME_1H, 'H': Timeframe.TIMEFRAME_1H,
        '240': Timeframe.TIMEFRAME_4H, 'H4': Timeframe.TIMEFRAME_4H, '4H': Timeframe.TIMEFRAME_4H,
        '1440': Timeframe.TIMEFRAME_1D, 'D': Timeframe.TIMEFRAME_1D, '1D': Timeframe.TIMEFRAME_1D,
        '10080': Timeframe.TIMEFRAME_1W, 'W': Timeframe.TIMEFRAME_1W, '1W': Timeframe.TIMEFRAME_1W,
    }


    def __init__(self, value):
        if not value:
            raise ValueError('Cannot initialize a resolution with a value of None')
        
        if isinstance(value, int|float):
            value = str(int(round(value)))
        
        self.timeframe = self._assign(value)
        self.name = self.timeframe.value[1]
        self.value = self.timeframe.value[0]

    def _assign(self, value):
        value = value.upper()

        if value not in self.RESOLUTION_MAP.keys():
            value = '1'
            logging.warning(f"Resolution `{value}` is not recognized. Defaulting to `1D`")    
        
        return self.RESOLUTION_MAP.get(value.upper(), value)
    
    def __eq__(self, other):
        if isinstance(other, str) and (other in self.RESOLUTION_MAP):
            other = self.RESOLUTION_MAP[other].value
            
        if isinstance(other, Resolution):
            return self.value == other.value
        elif isinstance(other, Union[int, float]):
            return self.value == other
        else:
            raise TypeError("unsupported operand type(s) for ==: 'Resolution' and '{}'".format(type(other).__name__))

    def __ne__(self, other):
        return not self.__eq__(other)

    def __lt__(self, other):
        if isinstance(other, str) and (other in self.RESOLUTION_MAP):
            other = self.RESOLUTION_MAP[other].value
            
        if isinstance(other, Resolution):
            return self.value < other.value
        elif isinstance(other, Union[int, float]):
            return self.value < other
        else:
            raise TypeError("unsupported operand type(s) for <: 'Resolution' and '{}'".format(type(other).__name__))

    def __le__(self, other):
        if isinstance(other, str) and (other in self.RESOLUTION_MAP):
            other = self.RESOLUTION_MAP[other].value
            
        if isinstance(other, Resolution):
            return self.value <= other.value
        elif isinstance(other, Union[int, float]):
            return self.value <= other
        else:
            raise TypeError("unsupported operand type(s) for <=: 'Resolution' and '{}'".format(type(other).__name__))

    def __gt__(self, other):
        if isinstance(other, str) and (other in self.RESOLUTION_MAP):
            other = self.RESOLUTION_MAP[other].value

        if isinstance(other, Resolution):
            return self.value > other.value
        elif isinstance(other, Union[int, float]):
            return self.value > other
        else:
            raise TypeError("unsupported operand type(s) for >: 'Resolution' and '{}'".format(type(other).__name__))

    def __ge__(self, other):
        if isinstance(other, str) and (other in self.RESOLUTION_MAP):
            other = self.RESOLUTION_MAP[other].value

        if isinstance(other, Resolution):
            return self.value >= other.value
        elif isinstance(other, Union[int, float]):
            return self.value >= other
        else:
            raise TypeError("unsupported operand type(s) for >=: 'Resolution' and '{}'".format(type(other).__name__))


class DotDict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


# FUNCTIONS
def clear_terminal():
        # Check the operating system
        if os.name == "posix":  # For Linux and macOS
            os.system("clear")


def debug(*texts):
    texts = list(texts)

    display_text = "\n".join([str(text) for text in texts])
    print(display_text)
    x = input(" " )

    if x == 'x':
        clear_terminal()
        
        file_path = "logs.log"
        with open(file_path, 'w'):
            pass

        exit()
    

def random_suffix(char_len:int=8):
    """
    Generate a random suffix
    """
    # using random.choices()
    # generating random strings
    return ''.join(random.choices(string.ascii_letters + string.digits , k=char_len))


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

    

    