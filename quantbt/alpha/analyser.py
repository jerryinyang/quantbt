import pandas as pd
import numpy as np # noqa: F401
import matplotlib.pyplot as plt # noqa: F401

from typing import List, Dict

from quantbt.alpha.reporters import Reporter # noqa: F401
from engine import Engine


class Analyser:
    def __init__(self) -> None:
        '''
        This class embodies the analyst performing series of backtests.
        '''

        self.engine : Engine = None 


    def add_engine(self, tickers:List[str], dataframes:Dict[str,pd.DataFrame], resolution:str, start_date, end_date):
        # Assert datatypes for each arguments
        assert tickers, ValueError('`tickers` list cannot be empty.')
        assert dataframes, ValueError('`dataframes` dictionary cannot be None.')
        assert resolution, ValueError('`resolution` cannot be set to None.')
        assert start_date, ValueError('`start_date` cannot be set to None.')
        assert end_date, ValueError('`end_date` cannot be set to None.')

        assert isinstance(tickers, list), ValueError('`tickers` must be passed in a list.')
        assert isinstance(dataframes, dict), ValueError('`dataframes` must be passed in a dictionary.')
        assert isinstance(resolution, str), ValueError('`resolution` must be a string.')

        assert (len(dataframes) == len(tickers)) and (set(dataframes.keys()) == set(tickers)), \
            ValueError('Tickers in `tickers` and `dataframes` must match.')