import logging
from typing import List


from engine import Engine, Alpha
from orders import Order
from utils import Bar, ObservableList as olist, ObservableDict as odict  # noqa: F401



logging.basicConfig(filename='logs.log', level=logging.INFO)

exectypes = Order.ExecType

class Backtester:

    def __init__(self, engine:Engine) -> None:
        self.engine = engine
        self.alphas : List[Alpha] = []


    def add_alpha(self, alpha:Alpha):
        self.alphas.append(alpha)


    def backtest(self):
        print('Initiating Backtest')

        # Set Backtest Range for Engine
        self.engine.portfolio= self.engine.init_portfolio(self.engine.date_range)
        
        # Iterate through each bar/index (timestamp) in the backtest range
        for bar_index in self.engine.portfolio.index:
            date = self.engine.portfolio.loc[bar_index, 'timestamp']

            # Create the bar objects for easier access to data
            bars = {}
            for ticker in self.engine.tickers:

                bar = Bar(
                    open=self.engine.dataframes[ticker].loc[date, 'open'],
                    high=self.engine.dataframes[ticker].loc[date, 'high'],
                    low=self.engine.dataframes[ticker].loc[date, 'low'],
                    close=self.engine.dataframes[ticker].loc[date, 'close'],
                    volume=self.engine.dataframes[ticker].loc[date, 'volume'],
                    index=bar_index,
                    timestamp=date,
                    resolution=self.engine.resolution,
                    ticker=ticker
                )
                bars[ticker] = bar

            # If Date is not the first date
            if bar_index > 0:
                # Update Portfolio
                self.engine.compute_portfolio_stats(bar_index)

                # Process Orders
                self.engine.compute_orders(bars)

                # Process Active Trades
                self.engine.compute_trades_stats(bars)

            # Filter Universe for Assets Eligible for Trading in a specific date
            eligible_assets, non_eligible_assets = self.engine.filter_eligible_assets(date)

            # Executing Signals
            for ticker in non_eligible_assets:
                # Units of asset in holding (Set to zero)
                self.engine.portfolio.loc[bar_index, f'{ticker} units'] += 0
                self.engine.portfolio.loc[bar_index, f'{ticker} open_pnl'] += 0

            # Run All Alphas in the backtester
            for alpha in self.alphas:
                alpha_long, alpha_short = alpha.next(eligibles=eligible_assets, datas=bars)

                non_eligible_assets = list(set(eligible_assets) - set(alpha_long + alpha_short))
                for ticker in non_eligible_assets:
                    # Units of asset in holding (Set to zero)
                    self.engine.portfolio.loc[bar_index, f'{ticker} units'] += 0
                    self.engine.portfolio.loc[bar_index, f'{ticker} open_pnl'] += 0
            
        print('Backtest Complete')
        return self.engine.history['trades']
