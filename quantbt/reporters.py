import threading

import numpy as np
import pandas as pd
import quantstats as qs
from backtester import Backtester
from portfolio import Portfolio
from trades import Trade


class AutoReporter:
    def __init__(self, metrics_mode:str = 'basic', earnings_mode : str = 'full') -> None:
        """
        Initialize the AutoReporter.

        Parameters:
        - metrics_mode (str): Options: 'basic', 'metrics', 'full'. Determines the level of detail in computed metrics.
        - returns_mode (str): Options: 'trades' | 'trades_only', 'full' | 'portfolio'. Specifies the type of earnings data to be computed.

        Attributes:
        - metrics_mode (str): The chosen metrics mode.
        - earnings_mode (str): The chosen returns mode.
        - metrics (pd.DataFrame): Container for computed metrics.
        - earnings (pd.DataFrame): Container for computed earnings.
        - earnings_trades (dict): Dictionary to store trade-specific earnings data.
        - earnings_portfolio (dict): Dictionary to store portfolio-specific earnings data.
        - metrics_lock (threading.Lock): Lock for ensuring thread safety when updating metrics.
        - earnings_trades_lock (threading.Lock): Lock for ensuring thread safety when updating trade earnings.
        - earnings_portfolio_lock (threading.Lock): Lock for ensuring thread safety when updating portfolio earnings.
        """

        self.metrics_mode = metrics_mode # Options : 'basic', 'metrics', 'full'
        self.earnings_mode = earnings_mode.replace('-','_').replace(' ', '_') # Options : 'trades' | 'trades_only', 'full' | 'portfolio'

        # Containers
        self.metrics = pd.DataFrame()
        self.earnings = pd.DataFrame()
        
        self.earnings_trades = {}
        self.earnings_portfolio = {}

        # Locks for thread safety
        self.metrics_lock = threading.Lock()
        self.earnings_trades_lock = threading.Lock()
        self.earnings_portfolio_lock = threading.Lock()

    
    def report(self):
        """
        Generate and retrieve a report containing both computed metrics and earnings.

        If earnings data is not available, it computes the earnings before generating the report.

        Returns:
        - Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the computed earnings and metrics DataFrames.
        """

        # Check if earnings data is available, compute if not
        if self.earnings.empty:
            self.compute_earnings()

        # Return the computed earnings and metrics
        return self.earnings, self.metrics


    def compute_report(self, backtester : Backtester, smart:bool=True):
        """
        Compute and store the trading performance metrics for a given backtest scenario.

        Parameters:
        - backtester (Backtester): The Backtester instance containing the backtest results.
        - smart (bool): Flag to enable smart computation of portfolio metrics.

        Returns:
        - pd.DataFrame: The computed metrics for the specified mode.
        """

        # Get Necessary Data
        id = backtester.id
        portfolio = backtester.engine.portfolio
        history = backtester.engine.history
        capital = backtester.engine.CAPITAL
        mode = self.metrics_mode
        
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
        metrics_portfolio = self._compute_portfolio_metrics(id, portfolio, smart)
        metrics_trades = self._compute_trade_metrics(id, history, capital)

        # Merge Metrics, Reset Index
        metrics = pd.merge(metrics_portfolio, metrics_trades, left_index=True, right_index=True)
        metrics.reset_index(inplace=True)
        metrics.rename(columns={'index': 'id'}, inplace=True)

        # Append new row into self.reports; Use locks for thread safety
        with self.metrics_lock:
            self.metrics = pd.concat([self.metrics, metrics[columns]], ignore_index=True)

        return metrics[columns]
    
    
    def compute_earnings(self):
        """
        Compute and store earnings based on the specified earnings mode.

        Returns:
        - pd.DataFrame: The computed earnings data.
        """

        # If self.earnings_mode is portfolio, 
        # Combine all the columns in a dataframe, with 'date' as the index
        # Assuming self.earnings_portfolio is defined as a dictionary
        if self.earnings_mode in ['full', 'portfolio']:
            earnings = pd.DataFrame(self.earnings_portfolio)
            earnings['date'] = pd.to_datetime(earnings['date'])
            earnings.set_index('date', inplace=True)

        else:
            # Find the maximum length of arrays
            max_length = max(len(arr) for arr in self.earnings_trades.values())

            # Create a dictionary with keys as column names and values as lists of array values
            formatted_data = {key: list(arr) + [np.nan] * (max_length - len(arr)) for key, arr in self.earnings_trades.items()}

            # Create a Pandas DataFrame
            earnings = pd.DataFrame(formatted_data)

        # Assign Earnings
        self.earnings = earnings
        
        return earnings


    def plot_equity_curves(self, highlight=None, highlight_message=None):
        """
        Plot equity curves for different backtest scenarios.

        Parameters:
        - highlight (str): The key of the scenario to highlight.
        - highlight_message (str): A message to display when highlighting a scenario.
        """

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()

        for key, values in self.earnings.items():
            color = 'orange' if key == 'original' else 'gray'
            alpha = 0.5 if key != highlight else 1.0
            label = key if key == 'original' else None

            ax.plot(np.arange(len(values)), values, color=color, alpha=alpha, label=label)

        if highlight and highlight in self.earnings:
            ax.plot(np.arange(len(self.earnings[highlight])), self.earnings[highlight], color='blue', label=highlight)

        if highlight_message:
            ax.text(0.5, 0.9, highlight_message, color='blue', transform=ax.transAxes, ha='center', va='center')

        ax.legend()
        plt.show()


    def plot_backtests(self, highlight=None, highlight_message=None):
        """
        Plot backtest equity curves using Plotly.

        Parameters:
        - highlight (str): The key of the scenario to highlight.
        - highlight_message (str): A message to display when highlighting a scenario.
        """
        
        import plotly.graph_objects as go

        fig = go.Figure()

        for column in self.earnings.columns:
            if column == 'original':
                color = 'orange'
                label = 'Original'
            elif column == highlight:
                color = 'blue'
                label = highlight_message if highlight_message else None
            else:
                color = 'gray'
                label = None

            fig.add_trace(go.Scatter(x=list(range(len(self.earnings[column]))), y=self.earnings[column], mode='lines', name=label, line=dict(color=color, width=2), text=label))

        fig.update_layout(title='Backtest Equity Curves',
                        xaxis_title='Time',
                        yaxis_title='Equity',
                        legend=dict(orientation='h'),
                        showlegend=False)

        fig.show()


    def _compute_trade_metrics(self, id:str, history : list[Trade], capital : float):
        trades = self._process_trade_history(history, capital)

        # Check for NaN and inf values and replace with defaults
        trades.replace([np.inf, -np.inf, np.nan], 0, inplace=True)

        # Store Trade Earnings. 'date' is not required; Index is trade count
        with self.earnings_trades_lock:
            self.earnings_trades[id] = trades['earnings'].to_numpy()

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

        profit_factor = gross_profit / abs(gross_loss) if gross_loss != 0 else 1 # Profit Factor
        total_closed_trades = len(trades) # Total Closed Trades
        count_profit = len(trades[trades['profit'] > 0]) # Number Winning Trades
        count_loss = len(trades[trades['profit'] < 0]) # Number Losing Trades
        count_breakeven = len(trades[trades['profit'] == 0]) # Number Breakeven Trades
        win_rate = (count_profit / total_closed_trades) * 100 # Percent Profitable
        payoff_ratio = average_profit / abs(average_loss) if average_loss != 0 else 1 # Ratio Avg Win / Avg Loss
        average_duration = trades['duration'].mean().total_seconds() # Avg Duration in Trades
        average_duration_winning = trades[(trades['profit'] > 0)]['duration'].mean().total_seconds() # Avg Duration in Winning Trades
        average_duration_losing = trades[(trades['profit'] < 0)]['duration'].mean().total_seconds()  # Avg Duration in Losing history

        # Calculate Maximum Consecutive Wins and Losses
        trades['streak'] = np.where(trades['profit'] >= 0, 1, 0)
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

        # Check for NaN and inf values and replace with defaults
        metrics.replace([np.inf, -np.inf, np.nan], 0, inplace=True)

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


    def _compute_portfolio_metrics(self, id:str, portfolio:Portfolio, smart:bool=True):
        records = self._prepare_portfolio(portfolio)
        portfolio = records['returns']

       # Store Portfolio Earnings
        with self.earnings_portfolio_lock:
            self.earnings_portfolio[id] = records['earnings'].to_numpy()
        
            # Add 'date' columns as index, if it does not exist
            if 'date' not in self.earnings_portfolio.keys():
                self.earnings_portfolio['date'] = records['date'].to_numpy()
        

        # CALCULATE METRICS
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
        data = portfolio.dataframe[['timestamp', 'equity']].copy(deep=True)
        data.loc[:, ['returns']] = data['equity'].pct_change()
        data['returns'].fillna(0)
        
        # Add and Rename Index to Date; Remove Timezone
        data.index = portfolio.dataframe['timestamp']
        data.index.name = 'date'

        # Rename 'equity' to 'earnings', and 'timestamp' to 'date'
        data.rename(columns={'equity' : 'earnings', 'timestamp' : 'date'}, inplace=True)

        return data[['date', 'earnings', 'returns']]


    def _process_trade_history(self, history : list[Trade], capital : float):
    
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
        trades['earnings'] = capital + trades['cumm_profit']
        trades['profit'].replace([np.nan, np.inf, -np.inf], 0, inplace=True)

        trades.index = trades.entry_timestamp
        trades.index.name = 'date'
        trades = trades.sort_values(by='date')

        # Return the overall trades DataFrame and the dictionary of trades per ticker
        return trades
    
    
    # PICKLE-COMPATIBILITY
    def __getstate__(self):
        state = self.__dict__.copy()
        del state['metrics_lock']
        del state['earnings_trades_lock']
        del state['earnings_portfolio_lock']
        return state


    def __setstate__(self, state):
        # Customize the object reconstruction
        self.__dict__.update(state)
    
