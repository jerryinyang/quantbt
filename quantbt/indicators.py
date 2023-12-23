from abc import ABC, abstractmethod
from collections import deque
from typing import Any, Literal, Union
from utils import Bar, Source, Logger

class Indicator(ABC):

    logger = Logger('logger_indicator')

    params = {

    }

    def __init__(self, name) -> None:
        self.name = name
        self.value = None

    @abstractmethod
    def update(self, bar : Bar):
        self.value = bar.close


class EMA(Indicator):
    
    params = {
        'period' : 20
    }

    properties = {
        'max_lookback' : 100
    }

    def __init__(self, 
                 name : str, 
                 source:Literal['open', 'high', 'low', 'close', 'hl2', 'hlc3', 'hlcc4', 'ohlc4'], 
                 period : int) -> None:
        self.name = name
        self.source = Source(source)
        self.params['period'] = period
        
        self.value = deque([], maxlen=self.properties['max_lookback'])
        self.window = deque([], maxlen=period)


    def update(self, bar : Union[Bar, Any]):
        data = self.source(bar) if isinstance(bar, Bar) else bar

        # Add New Data to Data Window
        self.window.appendleft(data)

        # Check if data window is filled
        if self.is_ready:
            return 

        # Convert Data to list and reverse the order
        data = list(self.window)
        data.reverse()

        # Calculate EMA value
        ema  = [data[0]]  # Initial value is the same as the first data point
        alpha = 2 / (len(data) + 1)

        for i in range(1, len(data)):
            ema_value = alpha * data[i] + (1 - alpha) * ema[i - 1]
            ema.append(ema_value)

        # Add new value to self.value        
        self.value.appendleft(ema[-1])

        return ema[-1]

    @property
    def is_ready(self):
        return len(self.window) < self.window.maxlen
        

    def __getitem__(self, index):
        if index > (len(self.value) - 1):
            self.logger.warning('Index exceeds available data.')
            return
        
        return self.value[index]
