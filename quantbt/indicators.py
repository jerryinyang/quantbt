import numpy as np
from abc import ABC, abstractmethod
from collections import deque
from typing import Any, Literal, Union

import utils_tv as tv
from utils import Bar, Source, Logger, DotDict, debug # noqa
from utils import random_suffix


class Indicator(ABC):

    logger = Logger('logger_indicator')

    def __init__(self, **kwargs) -> None:

        # Initialize the class attributes
        self._init_parameters(**kwargs)

        # Generate placeholder name
        self._name = self.__class__.__name__.lower() + '-' + random_suffix()
        self._name = kwargs.get('name', self._name)

        # Store source data
        self._window = deque([np.nan] * self.props.max_lookback, maxlen=self.props.max_lookback)

        # Store Indicator Values
        self.value = deque([np.nan] * self.props.max_lookback, maxlen=self.props.max_lookback)


    @abstractmethod
    def update(self, bar : Bar):
        self.value = bar.close


    @abstractmethod
    def _init_parameters(self, **kwargs):
        """
        Update Class Attributes, with passed keyword arguments
        """
        self.parameters = DotDict({
        })

        self.properties = DotDict({
            'max_lookback' : 1, 
        })

        # Get Default Parameter Definition
        if kwargs.get('parameters'):
            # Attribute is redefined in kwargs
            setattr(self, 'parameters', kwargs.get('parameters'))
        
        # Get Default Properties Definition
        if kwargs.get('properties'):
            # Attribute is redefined in kwargs
            setattr(self, 'properties', kwargs.get('properties'))

        # Aliases
        self.params = self.parameters
        self.props = self.properties

        # Update params to self.params
        required_params = list(set(self.params.keys()).intersection(kwargs.keys()))
        self.params.update(**{_key : kwargs.get(_key) for _key in required_params})

        # Update properties to self.props
        properties = list(set(self.props.keys()).intersection(kwargs.keys()))
        self.props.update(**{_key : kwargs.get(_key) for _key in properties})


    @property
    @abstractmethod
    def is_ready(self):
        """
        Checks if all source data is available, and if self.value has been calculated
        """
        return (not tv.na(self.value[0])) and \
            (not any([tv.na(_value) for _value in self._window]))


    def __getitem__(self, index):
        if index >= len(self.value):
            self.logger.warning('Index exceeds available data.')
            return
        
        return self.value[index]       


class EMA(Indicator):
    logger = Logger('logger_indicator')

    def __init__(self, 
                 source:Literal['open', 'high', 'low', 'close', 'hl2', 'hlc3', 'hlcc4', 'ohlc4'],
                 period : int, **kwargs) -> None:
        
        # Generate self._name, self._window, self._value
        super().__init__(**kwargs)
        
        # Update params to self.params
        kwargs.update(source=Source(source=source), 
                      period=period)
        
        # Initialize/Update the class attributes
        self._init_parameters(**kwargs)

        # Store source data
        self._window = deque([np.nan] * self.params.period, maxlen=self.params.period)

        # Store Indicator Values
        self.value = deque([np.nan] * self.props.max_lookback, maxlen=self.props.max_lookback)
    

    def update(self, data : Union[Bar, Any]):
        data = self.params.source(data) if isinstance(data, Bar) else data

        # Add New Data to Data Window
        self._window.appendleft(data)

        # Check if data window is filled
        if any(tv.na(value) for value in self._window):
            return np.nan

        # Convert Data to list and reverse the order
        values = list(self._window)
        values.reverse()

        # Calculate EMA value
        ema  = [values[0]]  # Initial value is the same as the first values point
        alpha = 2 / (len(values) + 1)

        for i in range(1, len(values)):
            ema_value = alpha * values[i] + (1 - alpha) * ema[i - 1]
            ema.append(ema_value)

        # Add new value to self.value        
        self.value.appendleft(ema[-1])

        return ema[-1]     


    def _init_parameters(self, **kwargs):
        """
        Update Class Attributes, with passed keyword arguments
        """
        parameters = DotDict({
            'source' : 'close', 
            'period' : 200,
            })

        properties = DotDict({
            'max_lookback' : 10, 
            })
        return super()._init_parameters(**kwargs, 
                                        parameters=parameters, 
                                        properties=properties)

    @property
    def is_ready(self):
        """
        Checks if all source data is available, and if self.value has been calculated
        """
        return (not tv.na(self.value[0])) and \
            (not any([tv.na(_value) for _value in self._window]))


#  OLD IMPLEMENTATION
# import math
# import pandas as pd
# import pandas_ta as ta
    
# class MarketBias(Indicator):

#     params = {
#         'period' : 100,
#         'smoothing' : 100,
#         'oscillator' : 7,
#     }
    

#     def __init__(self, name:str, period:int, smoothing:int) -> None:
#         super().__init__(name)
        
#         self.period = period
#         self.smoothing = smoothing
#         self.params.update(period=period, smoothing=smoothing)

#         # Smoothen the OHLC values 
#         self._ema_o = EMA('ema_open', 'open', period)
#         self._ema_h = EMA('ema_high', 'high', period)
#         self._ema_l = EMA('ema_low', 'low', period)
#         self._ema_c = EMA('ema_close', 'close', period)

#         self._ema_o2 = None
#         self._ema_h2 = None
#         self._ema_l2 = None
#         self._ema_c2 = None

#         self._ema_osc = None
        

#         # Windows
#         self._window_xhaopen = deque([float('nan'), float('nan')], maxlen=2)
#         self._window_haclose = deque([float('nan'), float('nan')], maxlen=2)

#     def update(self, bar: Bar):
#         # Update Indicators
#         self._ema_o.update(bar)
#         self._ema_h.update(bar)
#         self._ema_l.update(bar)
#         self._ema_c.update(bar)

#         # EMA Values Not Ready
#         if not (
#             self._ema_o.is_ready and
#             self._ema_h.is_ready and
#             self._ema_l.is_ready and
#             self._ema_c.is_ready
#             ):
#             return   

#         o : float = self._ema_o[0]
#         h : float = self._ema_h[0]
#         l : float = self._ema_l[0]  # noqa: E741
#         c : float = self._ema_c[0]     

#         # Calculate the Heikin Ashi OHLC values from it
#         haclose = (o + h + l + c) / 4
#         xhaopen = (o + c) / 2
#         haopen = tv.na(
#             tv.ternary(
#                 not tv.na(self._window_xhaopen[1]), # Condition : if xhaopen[1] exists
#                 (o + c) / 2, # if condition is True
#                 (self._window_xhaopen[0] + self._window_haclose[0]) # if condition is False
#             )
#         ) / 2
#         hahigh = max(h, max(haopen, haclose))
#         halow = min(l, min(haopen, haclose))

#         # Update Windows with new values
#         self._window_haclose.appendleft(haclose)
#         self._window_xhaopen.appendleft(xhaopen)

#         # Smoothen the Heiken Ashi Candles
#         if tv.na([self._ema_o2, self._ema_o2, self._ema_o2, self._ema_o2]):
#             period = self.params['smoothing']

#             self._ema_o2 = EMA('ema_o2', 'open', period)
#             self._ema_h2 = EMA('ema_h2', 'high', period)
#             self._ema_l2 = EMA('ema_l2', 'low', period)
#             self._ema_c2 = EMA('ema_c2', 'close', period)
#         else: 
#             self._ema_o2.update(haopen)
#             self._ema_h2.update(haclose)
#             self._ema_l2.update(hahigh)
#             self._ema_c2.update(halow)

#         o2 : float = self._ema_o2[0]
#         h2 : float = self._ema_h2[0] # noqa
#         l2 : float = self._ema_l2[0] # noqa
#         c2 : float = self._ema_c2[0] 

#         # Oscillator 
#         osc_bias = 100 * (c2 - o2)

#         if tv.na(self._ema_osc):
#             self._ema_osc = EMA('ema_osc', 'close', self.params['oscillator'])
#         else:
#             self._ema_osc.update(osc_bias)

#         # Generate Signal
#         if (osc_bias > 0) and (osc_bias >= self._ema_osc[0]):
#             signal = 2
#         elif (osc_bias > 0) and (osc_bias < self._ema_osc[0]):
#             signal = 1
#         elif (osc_bias < 0) and (osc_bias <= self._ema_osc[0]):
#             signal = -2
#         elif (osc_bias < 0) and (osc_bias > self._ema_osc[0]):
#             signal = -1
#         else:
#             signal = 0

#         # Assign Signal to self.value
#         self.value = signal


# class HawkesProcess(Indicator):

#     params = {
#         'kappa' : 0.1,
#         'lookback' : 14,
#         'percentile' : 5,
#     }
    
#     def __init__(self, name:str, kappa:float, lookback:int, percentile:float) -> None:
#         super().__init__(name)

#         self.kappa = kappa
#         self.lookback = lookback
#         self.percentile = percentile
#         self.params.update(kappa=kappa, lookback=lookback, percentile=percentile)

#         self._hawkes = deque([float('nan')] * lookback, maxlen=lookback)
#         self._upper = deque([float('nan')] * 2, maxlen=2)
#         self._lower = deque([float('nan')] * 2, maxlen=2)


#     def update(self, bar: Bar):
#         alpha = math.exp(self.kappa)
#         hawkes = self._hawkes[0]

#         if tv.na(hawkes):
#             hawkes = bar.close
#         else:
#             hawkes = (hawkes * alpha + bar.close) * self.kappa
            
#         # Rolling Quantiles
#         upper_band = np.percentile(np.array((self._hawkes), 100 - self.params['percentile']), axis=None, method='closest_observation')
#         lower_band = np.percentile(np.array((self._hawkes), self.params['percentile']), axis=None, method='closest_observation')

#         # Update Windows
#         self._hawkes.appendleft(hawkes)
#         self._upper.appendleft(upper_band)
#         self._lower.appendleft(lower_band)


# class ATR(Indicator):
    
#     params = {
#         'period' : 14
#     }

#     properties = {
#         'max_lookback' : 500
#     }

#     def __init__(self, 
#                  name : str, 
#                  period : int) -> None:
        
#         self.name = name
#         self.params.update(period=period)
        
#         self.value = deque([], maxlen=self.props['max_lookback'])
#         self._window : List[Bar] = deque([], maxlen=period+20)


#     def update(self, bar : Union[Bar, Any]):
#         # Add New Data to Data Window
#         self._window.appendleft(bar)

#         # Check if data window is filled
#         if self.is_ready:
#             return 

#         # Convert Data to list and reverse the order
#         data = list(self._window)
#         data.reverse()

#         # Extract OHLC data from each bar into a dictionary
#         data_ohlc = {
#             'time' : [], 
#             'high' : [],
#             'low' : [],
#             'close' : [],
#         }

#         for bar in data:
#             data_ohlc['time'].append(bar.timestamp)
#             data_ohlc['high'].append(bar.high)
#             data_ohlc['low'].append(bar.low)
#             data_ohlc['close'].append(bar.close)

#         # Create a dataframe with the values
#         data = pd.DataFrame(data_ohlc)
#         data['atr'] = ta.atr(data['high'], data['low'], data['close'], length=self.params.get('period', 14))
#         atr = data['atr']
#         value = atr.iloc[-1]

#         # Add new value to self.value        
#         self.value.appendleft(value)
        
#         print(data.iloc[-1])

#         return value

#     @property
#     def is_ready(self):
#         return len(self._window) < self._window.maxlen
        

#     def __getitem__(self, index):
#         if index > (len(self.value) - 1):
#             self.logger.warning('Index exceeds available data.')
#             return
        
#         return self.value[index]
