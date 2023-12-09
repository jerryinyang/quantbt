from alpha import Alpha
from engine import Engine
from utils import Logger

class Sizer:
    logger = Logger('logger_sizer')

    def __init__(self, engine:Engine, alphas : list[Alpha], max_exposure : float) -> None:
        self.engine = engine
        self.alphas = alphas

        # Parameters
        self.max_exposure = max_exposure # Maximum % of Balance to Risk at a time; Can be volatility-weighted or performace_weighted (Kelly's Criterion)
        self.asset_weights = {ticker : 1 for ticker in self.engine.tickers} # Weights per ticker
        self.strategy_weights = {alpha.name : 1 for alpha in self.alphas} # Weights per strategy    


    def calculate_risk(self):

        # Check if the lengths of asset_weights and self.strategy_weights match
        if not self.engine.tickers:
            self.logger.exception("Assets cannot be empty.", ValueError('Engine contains no price data.'))
                                  
        # Check if the lengths of asset_weights and self.strategy_weights match
        if not self.alphas:
            self.logger.exception("Alphas list cannot be empty.", ValueError('Sizers contains no alphas.'))

       # Check if the sum of weights is 1 for both assets and strategies
        if not self._is_close(sum(self.asset_weights.values()), 1):        
            self.asset_weights = {asset : weight / sum(self.asset_weights.values())
                                  for asset, weight
                                  in self.asset_weights.items()}
        
        # Check if the sum of weights is 1 for both assets and strategies
        if not self._is_close(sum(self.strategy_weights.values()), 1):
            self.strategy_weights = {asset : weight / sum(self.strategy_weights.values())
                                  for asset, weight
                                  in self.strategy_weights.items()}           

        allocation_matrix = {}
        for asset_name, asset_weight in self.asset_weights.items():
            allocation = {}
            for strategy_name, strategy_weight in self.strategy_weights.items():
                allocation[strategy_name] = strategy_weight * asset_weight * self.max_exposure

            allocation_matrix[asset_name] = allocation
            
        return allocation_matrix


    def rebalance(self):
        """
        Adjust the maximum exposure, ticker weights and strategy weights based on current performance state.
        """


    def _is_close(self, a : float, b : float, rel_tol : float=1e-9, abs_tol : float=0.0) -> bool :
        return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)
    

    # PICKLE-COMPATIBILITY
    def __getstate__(self):
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        # Customize the object reconstruction
        self.__dict__.update(state)
