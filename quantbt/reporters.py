import pandas as pd
import quantstats as qs

from typing import Any # noqa : F401

from trades import Trade
from portfolio import Portfolio
from backtester import Backtester

class AutoReporter:
    def __init__(self, mode : str = 'basic') -> None:
        self.mode = mode
        self.reports = pd.DataFrame()
        ...
    

    def compute_report(self, backtester : Backtester, smart:bool=True):
        # Get Necessary Data
        id = backtester.id
        portfolio = backtester.engine.portfolio
        history = backtester.engine.history
        mode = self.mode
        
        # Select the columns to be include, based on the mode
        modes = ['basic', 'metrics', 'full']
        if mode not in modes:
            mode = 'basic'

        if mode == 'basic':
            columns = [ 'id',
                'net_profit',   'average_pnl',
                'max_drawdown', 'max_drawdown_percent',
                'total_closed_trades',  'profit_factor',
                'payoff_ratio', 'win_rate %',
                'average_duration', 'max_consecutive_wins',
                'max_consecutive_losses',   'sharpe_ratio',
                'cagr', 'sortino'
            ]

        elif mode == 'metrics':
            columns = [ 'id',
                'max_drawdown',
                'sharpe_ratio',
                'cagr',
                'sortino',
                'kurtosis',
                'skew',
                'adjusted_sortino',
                'risk_return_ratio',
            ]

        else:
            columns = [ 'id',
                'net_profit', 'gross_profit', 'gross_loss', 'average_pnl',
                'average_profit', 'average_loss', 'largest_profit', 'largest_loss',
                'net_profit_percent', 'gross_profit_percent', 'gross_loss_percent',
                'average_pnl_percent', 'average_profit_percent', 'average_loss_percent',
                'largest_profit_percent', 'largest_loss_percent', 'max_runup',
                'max_runup_percent', 'max_drawdown', 'max_drawdown_percent',
                'profit_factor', 'total_closed_trades', 'count_profit', 'count_loss',
                'count_breakeven', 'win_rate %', 'payoff_ratio', 'average_duration',
                'average_duration_winning', 'average_duration_losing',
                'max_consecutive_wins', 'max_consecutive_losses', 'sharpe_ratio',
                'cagr', 'sortino', 'kurtosis', 'skew', 'adjusted_sortino',
                'risk_return_ratio'
            ]

        # Compute Trade and Portfolio Metrics
        metrics_portfolio = self.compute_portfolio_metrics(id, portfolio, smart)
        metrics_trades = self.compute_trade_metrics(id, history)

        # Merge Metrics, Reset Index
        metrics = pd.merge(metrics_portfolio, metrics_trades, left_index=True, right_index=True)
        metrics.reset_index(inplace=True)
        metrics.rename(columns={'index': 'id'}, inplace=True)

        # Append new row into self.reports
        self.reports = pd.concat([self.reports, metrics[columns]], ignore_index=True)

        return metrics[columns]
        

    def compute_trade_metrics(self, id:str, history : list[Trade]):
        trades = self._process_trade_history(history)

        # PNL
        net_profit = trades['profit'].sum()         # Net Profit
        gross_profit = trades[trades['profit'] > 0]['profit'].sum() # Gross Profit
        gross_loss = trades[trades['profit'] < 0]['profit'].sum() # Gross Loss
        average_pnl = trades['profit'].mean() # Avg Trade
        average_profit = trades[(trades['profit'] > 0)]['profit'].mean() # Avg Winning Trade
        average_loss = trades[(trades['profit'] < 0)]['profit'].mean() # Avg Losing Trade
        largest_profit = trades[(trades['profit'] > 0)]['profit'].max() # Largest Winning Trade
        largest_loss = trades[(trades['profit'] < 0)]['profit'].min() # Largest Losing Trade

        # PNL PERCENTAGES
        net_profit_percent = trades['profit_percent'].sum() # Net Profit
        gross_profit_percent = trades[trades['profit_percent'] > 0]['profit_percent'].sum() # Gross Profit
        gross_loss_percent = trades[trades['profit_percent'] < 0]['profit_percent'].sum() # Gross Loss
        average_pnl_percent = trades['profit_percent'].max() # Avg Trade
        average_profit_percent = trades[(trades['profit_percent'] > 0)]['profit_percent'].mean() # Avg Winning Trade
        average_loss_percent = trades[(trades['profit_percent'] < 0)]['profit_percent'].mean() # Avg Losing Trade
        largest_profit_percent = trades[(trades['profit_percent'] > 0)]['profit_percent'].max() # Largest Winning Trade
        largest_loss_percent = trades[(trades['profit_percent'] < 0)]['profit_percent'].min() # Largest Losing Trade

        profit_factor = gross_profit / abs(gross_loss) # Profit Factor
        total_closed_trades = len(trades) # Total Closed Trades
        count_profit = len(trades[trades['profit'] > 0]) # Number Winning Trades
        count_loss = len(trades[trades['profit'] < 0]) # Number Losing Trades
        count_breakeven = len(trades[trades['profit'] == 0]) # Number Breakeven Trades
        win_rate = (count_profit / total_closed_trades) * 100 # Percent Profitable
        payoff_ratio = average_profit / abs(average_loss) # Ratio Avg Win / Avg Loss
        average_duration = trades['duration'].mean().total_seconds() # Avg Duration in Trades
        average_duration_winning = pd.to_timedelta(trades[(trades['profit'] > 0)]['duration'].mean().total_seconds(), unit='s') # Avg Duration in Winning Trades
        average_duration_losing = pd.to_timedelta(trades[(trades['profit'] < 0)]['duration'].mean().total_seconds(), unit='s')  # Avg Duration in Losing history

        # Calculate Maximum Consecutive Wins and Losses
        trades['streak'] = (trades['profit'] >= 0).astype(int)
        groups = trades['streak'].ne(trades['streak'].shift()).cumsum()
        streak_lengths = trades.groupby(groups)['streak'].transform('count')

        max_consecutive_losses = streak_lengths[trades['profit'] < 0].max()
        max_consecutive_wins = streak_lengths[trades['profit'] >= 0].max()
        trades = trades.drop(columns=['streak'])

        # Move to Portfolio Metrics
        max_runup, max_runup_percent = self._calculate_max_runup(trades) # Max Run-up 
        max_drawdown, max_drawdown_percent = self._calculate_max_drawdown(trades) # Max Drawdown

        # Create a dictionary for each set of metrics
        metrics_data = {
            # PNL
            'net_profit': [net_profit],
            'gross_profit': [gross_profit],
            'gross_loss': [gross_loss],
            'average_pnl': [average_pnl],
            'average_profit': [average_profit],
            'average_loss': [average_loss],
            'largest_profit': [largest_profit],
            'largest_loss': [largest_loss],

            # PNL PERCENTAGE
            'net_profit_percent': [net_profit_percent],
            'gross_profit_percent': [gross_profit_percent],
            'gross_loss_percent': [gross_loss_percent],
            'average_pnl_percent': [average_pnl_percent],
            'average_profit_percent': [average_profit_percent],
            'average_loss_percent': [average_loss_percent],
            'largest_profit_percent': [largest_profit_percent],
            'largest_loss_percent': [largest_loss_percent],

            # STATISTICS

            'max_runup': [max_runup], 
            'max_runup_percent': [max_runup_percent], 
            'max_drawdown': [max_drawdown], 
            'max_drawdown_percent': [max_drawdown_percent], 

            'profit_factor': [profit_factor],
            'total_closed_trades': [total_closed_trades],
            'count_profit': [count_profit],
            'count_loss': [count_loss],
            'count_breakeven': [count_breakeven],
            'win_rate %': [win_rate],
            'payoff_ratio': [payoff_ratio],
            'average_duration': [average_duration],
            'average_duration_winning': [average_duration_winning],
            'average_duration_losing': [average_duration_losing],
            'max_consecutive_wins': [max_consecutive_wins],
            'max_consecutive_losses': [max_consecutive_losses]
        }

        metrics = pd.DataFrame(metrics_data, index=[id])

        # Set the columns datatypes
        metrics = metrics.astype({
            'net_profit': float,
            'gross_profit': float,
            'gross_loss': float,
            'average_pnl': float,
            'average_profit': float,
            'average_loss': float,
            'largest_profit': float,
            'largest_loss': float,
            'net_profit_percent': float,
            'gross_profit_percent': float,
            'gross_loss_percent': float,
            'average_pnl_percent': float,
            'average_profit_percent': float,
            'average_loss_percent': float,
            'largest_profit_percent': float,
            'largest_loss_percent': float,
            'max_runup': float,
            'max_drawdown': float,
            'profit_factor': float,
            'total_closed_trades': int,
            'count_profit': int,
            'count_loss': int,
            'count_breakeven': int,
            'win_rate %': float,
            'payoff_ratio': float,
            'average_duration': 'timedelta64[ns]',
            'average_duration_winning': 'timedelta64[ns]',
            'average_duration_losing': 'timedelta64[ns]',
            'max_consecutive_wins': int,
            'max_consecutive_losses': int,
        })

        return metrics


    def compute_portfolio_metrics(self, id:str, portfolio:Portfolio, smart:bool=True):
        portfolio = self._prepare_portfolio(portfolio)
        
        # METRICS
        metrics_data = {
            'sharpe_ratio' : qs.stats.sharpe(portfolio, smart=smart),
            'cagr' : qs.stats.cagr(portfolio),
            'sortino' : qs.stats.sortino(portfolio, smart=smart),
            'kurtosis' : qs.stats.kurtosis(portfolio),
            'skew' : qs.stats.skew(portfolio),
            'adjusted_sortino' : qs.stats.adjusted_sortino(portfolio, smart=smart),
            'risk_return_ratio' : qs.stats.risk_return_ratio(portfolio)
        }

        metrics = pd.DataFrame(metrics_data, index=[id])

        # Set the columns datatypes
        metrics = metrics.astype({
            'sharpe_ratio' : float,
            'cagr' : float,
            'sortino' : float,
            'kurtosis' : float,
            'skew' : float,
            'adjusted_sortino' : float,
            'risk_return_ratio' : float
        })

        return metrics
   

    def _calculate_max_drawdown(self, data : pd.DataFrame):
        max_dd = data['cumm_profit'].cummax() - data['cumm_profit']
        max_dd_percent = data['cumm_profit_perc'].cummax() - data['cumm_profit_perc']
        return max_dd.max(), max_dd_percent.max()


    def _calculate_max_runup(self, data : pd.DataFrame):
        max_runup = data['cumm_profit'] - data['cumm_profit'].cummin()
        max_runup_percent = data['cumm_profit_perc'] - data['cumm_profit_perc'].cummin()
        return max_runup.max(), max_runup_percent.max()
    

    def _prepare_portfolio(self, portfolio : Portfolio):        
        # Extract Returns From Portfolio
        data = portfolio.dataframe['equity'].pct_change()

        # Add and Rename Index to Date; Remove Timezone
        data.index = portfolio.dataframe['timestamp']
        data.index.name = 'Date'
        return data


    def _process_trade_history(self, history : list[Trade]):
    
        """
        Compute and return a trades report including all trades and trades per ticker.

        Returns:
        - all_trades (pd.DataFrame): DataFrame containing properties of all trades.
        - trades_per_ticker (dict): Dictionary with tickers as keys and trade DataFrames as values.
        """

        # Initialize an empty list to store trade properties for all trades
        trades_list = []

        # Iterate through each trade in the history
        for trade in history:

            # Extract relevant properties from the trade
            properties = {
                'id': trade.id,
                'direction': trade.direction.value,
                'ticker': trade.ticker,
                'entry_timestamp': pd.Timestamp(trade.entry_timestamp),
                'exit_timestamp': pd.Timestamp(trade.exit_timestamp),
                'entry_price': trade.entry_price,
                'exit_price': trade.exit_price,
                'size': trade.size,
                'profit': trade.params.pnl,
                'profit_percent': trade.params.pnl_perc,
                'max_runup': trade.params.max_runup,
                'max_runup_percent': trade.params.max_runup_perc,
                'max_drawdown': trade.params.max_drawdown,
                'max_drawdown_percent': trade.params.max_drawdown_perc,
            }

            # Append the trade properties to the list
            trades_list.append(properties)

        # Compile all trade properties into a Pandas DataFrame
        trades = pd.DataFrame(trades_list)

        # Convert data types of DataFrame columns for consistency
        trades = trades.astype({
            'id': int,
            'direction': int,
            'ticker': str,
            'entry_price': float,
            'exit_price': float,
            'size': float,
            'profit': float,
            'profit_percent': float,
            'max_runup': float,
            'max_runup_percent': float,
            'max_drawdown': float,
            'max_drawdown_percent': float
        })

        '''
        Adds the Cummulative Profit (and %) and Duration columns. Also converts the entry and exit timestamp columsn to datetime objects
        '''
        trades['cumm_profit'] = trades['profit'].cumsum()
        trades['cumm_profit_perc'] = trades['profit_percent'].cumsum()
        trades['entry_timestamp'] = pd.to_datetime(trades['entry_timestamp'])
        trades['exit_timestamp'] = pd.to_datetime(trades['exit_timestamp'])
        trades['duration'] = trades['exit_timestamp'] - trades['entry_timestamp']

        trades.index = trades.entry_timestamp
        trades.index.name = 'date'
        trades = trades.sort_values(by='date')

        # Return the overall trades DataFrame and the dictionary of trades per ticker
        return trades
    
