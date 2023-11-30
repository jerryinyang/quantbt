import pandas as pd
import numpy as np # noqa: F401
import matplotlib.pyplot as plt # noqa: F401
import concurrent.futures # noqa: F401
from typing import Dict, List # noqa: F401

from backtester import Backtester



class Analyser:
    def __init__(self, backtester:Backtester) -> None:
        '''
        This class embodies the analyst performing series of backtests.
        '''
        self.backtester = backtester

    def analyse_robustness_price(self, iterations:int):
        '''
        For this analysis, synthetic OHLCV data is created by modifying the actual backtest data. 
        Then, the backtest is run `n` times, and the performance metrics are recalculated and compared.
        '''

    
    def synthetic_price_permute(self, data:pd.DataFrame):
        assert 'close' in data.columns, ValueError("Missing required `close` column in dataframe.")
        
        data['price_change'] = None
        

    def synthetic_price_noise(self, data:pd.DataFrame):
        pass
    
