from abc import ABC, abstractmethod
from collections import deque
from utils import Bar, Source

class Indicator(ABC):
    
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

    def __init__(self, name : str, source : str, period : int) -> None:
        self.name = name
        self.source = Source(source)
        self.params['period'] = period
        
        self.value = deque([], maxlen=self.properties['max_lookback'])
        self.window = deque([], maxlen=period)


    def update(self, bar : Bar):
        # Add New Data to Data Window
        self.window.appendleft(self.source(bar))

        if len(self.window) < self.window.maxlen:
            return 

        # Convert Data to list and reverse the order
        data = list(self.window)
        data.reverse()

        # Calculate 
        ema  = [data[0]]  # Initial value is the same as the first data point
        alpha = 2 / (len(data) + 1)

        for i in range(1, len(data)):
            ema_value = alpha * data[i] + (1 - alpha) * ema[i - 1]
            ema.append(ema_value)

        # Add new value to self.value        
        self.value.appendleft(ema[-1])