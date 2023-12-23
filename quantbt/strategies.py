'''
CUSTOM STRATEGIES (ALPHAS) AND THEIR DEPENDENCIES
'''

import utils_tv as tv
from alpha import Alpha
from engine import Engine
from orders import Order
from indicators import Indicator, EMA
from utils import Bar 

import math
import numpy as np
from collections import deque
from typing import List, Dict

exectypes = Order.ExecType

_ = '''
open
high
low
close
'''


# INDICATORS
class MarketBias(Indicator):

    params = {
        'period' : 100,
        'smoothing' : 100,
        'oscillator' : 7,
    }
    

    def __init__(self, name:str, period:int, smoothing:int) -> None:
        super().__init__(name)
        
        self.period = period
        self.smoothing = smoothing
        self.params.update(period=period, smoothing=smoothing)

        # Smoothen the OHLC values 
        self._ema_o = EMA('ema_open', 'open', period)
        self._ema_h = EMA('ema_high', 'high', period)
        self._ema_l = EMA('ema_low', 'low', period)
        self._ema_c = EMA('ema_close', 'close', period)

        self._ema_o2 = None
        self._ema_h2 = None
        self._ema_l2 = None
        self._ema_c2 = None

        self._ema_osc = None
        

        # Windows
        self._window_xhaopen = deque([float('nan'), float('nan')], maxlen=2)
        self._window_haclose = deque([float('nan'), float('nan')], maxlen=2)

    def update(self, bar: Bar):
        # Update Indicators
        self._ema_o.update(bar)
        self._ema_h.update(bar)
        self._ema_l.update(bar)
        self._ema_c.update(bar)

        # EMA Values Not Ready
        if not (
            self._ema_o.is_ready and
            self._ema_h.is_ready and
            self._ema_l.is_ready and
            self._ema_c.is_ready
            ):
            return   

        o : float = self._ema_o[0]
        h : float = self._ema_h[0]
        l : float = self._ema_l[0]  # noqa: E741
        c : float = self._ema_c[0]     

        # Calculate the Heikin Ashi OHLC values from it
        haclose = (o + h + l + c) / 4
        xhaopen = (o + c) / 2
        haopen = tv.na(
            tv.ternary(
                not tv.na(self._window_xhaopen[1]), # Condition : if xhaopen[1] exists
                (o + c) / 2, # if condition is True
                (self._window_xhaopen[0] + self._window_haclose[0]) # if condition is False
            )
        ) / 2
        hahigh = max(h, max(haopen, haclose))
        halow = min(l, min(haopen, haclose))

        # Update Windows with new values
        self._window_haclose.appendleft(haclose)
        self._window_xhaopen.appendleft(xhaopen)

        # Smoothen the Heiken Ashi Candles
        if tv.na([self._ema_o2, self._ema_o2, self._ema_o2, self._ema_o2]):
            period = self.params['smoothing']

            self._ema_o2 = EMA('ema_o2', 'open', period)
            self._ema_h2 = EMA('ema_h2', 'high', period)
            self._ema_l2 = EMA('ema_l2', 'low', period)
            self._ema_c2 = EMA('ema_c2', 'close', period)
        else: 
            self._ema_o2.update(haopen)
            self._ema_h2.update(haclose)
            self._ema_l2.update(hahigh)
            self._ema_c2.update(halow)

        o2 : float = self._ema_o2[0]
        h2 : float = self._ema_h2[0] # noqa
        l2 : float = self._ema_l2[0] # noqa
        c2 : float = self._ema_c2[0] 

        # Oscillator 
        osc_bias = 100 * (c2 - o2)

        if tv.na(self._ema_osc):
            self._ema_osc = EMA('ema_osc', 'close', self.params['oscillator'])
        else:
            self._ema_osc.update(osc_bias)

        # Generate Signal
        if (osc_bias > 0) and (osc_bias >= self._ema_osc[0]):
            signal = 2
        elif (osc_bias > 0) and (osc_bias < self._ema_osc[0]):
            signal = 1
        elif (osc_bias < 0) and (osc_bias <= self._ema_osc[0]):
            signal = -2
        elif (osc_bias < 0) and (osc_bias > self._ema_osc[0]):
            signal = -1
        else:
            signal = 0

        # Assign Signal to self.value
        self.value = signal


class HawkesProcess(Indicator):

    params = {
        'kappa' : 0.1,
        'lookback' : 14,
        'percentile' : 5,
    }
    
    def __init__(self, name:str, kappa:float, lookback:int, percentile:float) -> None:
        super().__init__(name)

        self.kappa = kappa
        self.lookback = lookback
        self.percentile = percentile
        self.params.update(kappa=kappa, lookback=lookback, percentile=percentile)

        self._hawkes = deque([float('nan')] * lookback, maxlen=lookback)
        self._upper = deque([float('nan')] * 2, maxlen=2)
        self._lower = deque([float('nan')] * 2, maxlen=2)


    def update(self, bar: Bar):
        alpha = math.exp(self.kappa)
        hawkes = self._hawkes[0]

        if tv.na(hawkes):
            hawkes = bar.close
        else:
            hawkes = (hawkes * alpha + bar.close) * self.kappa
            
        # Rolling Quantiles
        upper_band = np.percentile(np.array((self._hawkes), 100 - self.params['percentile']), axis=None, method='closest_observation')
        lower_band = np.percentile(np.array((self._hawkes), self.params['percentile']), axis=None, method='closest_observation')

        # Update Windows
        self._hawkes.appendleft(hawkes)
        self._upper.appendleft(upper_band)
        self._lower.appendleft(lower_band)





# STRATEGIES
class StrategyHawkesProcess(Alpha):

    def __init__(self, name: str, engine: Engine) -> None:
        super().__init__(name, engine)


class BaseAlpha(Alpha):
    def __init__(self, name : str, engine: Engine, profit_perc:float, loss_perc:float) -> None:
        super().__init__(name, engine)

        self.profit_perc = profit_perc
        self.loss_perc = loss_perc


    def next(self, eligibles:List[str], datas:Dict[str, Bar], allocation_per_ticker:Dict[str, float]):
        super().next(eligibles, datas, allocation_per_ticker)

        # Decision-making / Signal-generating Algorithm
        alpha_long, alpha_short = self.signal_generator(eligibles)
        eligible_assets = list(set(alpha_long + alpha_short))
        
        for ticker in eligible_assets:
            bar = datas[ticker]

            # Calculate Risk Amount based on allocation_per_ticker
            risk_dollars =  self.sizer(bar)

            # Tickers to Long
            if ticker in alpha_long:
                entry_price = bar.close

                position_size = risk_dollars / entry_price

                exit_tp = entry_price * (1 + self.profit_perc)
                exit_sl = entry_price * (1 - self.loss_perc)
                
                # Create and Send Long Order
                self.buy(bar, entry_price, position_size, exectypes.Market, exit_profit=exit_tp, exit_loss=exit_sl)                    

                # Tickers to Short
            elif ticker in alpha_short:
                entry_price = bar.close
                position_size = -1 * risk_dollars / entry_price

                exit_tp = entry_price * (1 - self.profit_perc)
                exit_sl = entry_price * (1 + self.loss_perc)
                
                # Create and Send Short Order
                self.sell(bar, entry_price, position_size, exectypes.Market, exit_profit=exit_tp, exit_loss=exit_sl)

        return alpha_long, alpha_short 


    def signal_generator(self, eligibles:list[str]) -> tuple:
        alpha_scores = { key : np.random.rand() for key in eligibles}

        alpha_scores = {
            key : value \
                for key, value in sorted(alpha_scores.items(), key=lambda item : item[1])
                } # Sorts the dictionary
        
        list_scores = list(alpha_scores.keys())

        if not list_scores:
            return [], []
        
        alpha_long = [list_scores[0]] 
        alpha_short = [list_scores[-1]] 

        return alpha_long, alpha_short       


    def reset_alpha(self, engine:Engine):
        '''
        Resets the alpha with a new engine.
        '''

        self.__init__(self.name, engine, self.profit_perc, self.loss_perc)
        self.logger.info(f'Alpha {self.name} successfully reset.')