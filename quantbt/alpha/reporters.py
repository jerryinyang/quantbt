import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from trades import Trade
# from engine import Engine

class Reporter:
    def __init__(self, trades_history : list[Trade]) -> None:
        '''
        This class would handle the collection and storage of backtest results.
        '''
        self.history = trades_history
        self.reports = self.compute_reports()


    # FOR COMPUTING REPORTS
    def compute_reports(self):
        '''
        Generate Reports for the Backtest.
        
        Returns:
            A dictionary containing tuples of trades history and metrics pairs;
            for the full backtest, and for each ticker.
        '''

        # Contains Reports for all tickers 
        reports = {}


        # Compute Full Trade Report ; This would be used for robustness analysis 
        
        # Compute the Trades History
        trades_list = self.history
        report_trades= self.compute_trades_report(trades_list)
        report_trades = self._process_trade_history(report_trades)

        self.tickers = report_trades['ticker'].unique().tolist()

        # Compute the metrics
        report_metrics = self.compute_metrics_report(report_trades)

        reports['full'] = (report_trades, report_metrics)


        # For each ticker, these would be used for parameter and scenario analysis
        for ticker in self.tickers:
            trades_list = [trade for trade in self.history if trade.ticker == ticker]

            # Compute the Trades History
            report_trades= self.compute_trades_report(trades_list)
            report_trades = self._process_trade_history(report_trades)

            # Compute the metrics
            report_metrics = self.compute_metrics_report(report_trades)

            # Add report bundle to reports dictionary 
            reports[ticker] = (report_trades, report_metrics)

        return reports
    

    def _process_trade_history(self, history):
        '''
        Adds the Cummulative Profit (and %) and Duration columns. Also converts the entry and exit timestamp columsn to datetime objects
        '''
        history['cumm_profit'] = history['profit'].cumsum()
        history['cumm_profit_perc'] = history['profit_percent'].cumsum()
        history['entry_timestamp'] = pd.to_datetime(history['entry_timestamp'])
        history['exit_timestamp'] = pd.to_datetime(history['exit_timestamp'])
        history['duration'] = history['exit_timestamp'] - history['entry_timestamp']

        history.index = history.entry_timestamp
        history.index.name = 'date'
        history = history.sort_values(by='date')

        return history


    def _calculate_max_drawdown(self, data : pd.DataFrame):
        max_dd = data['cumm_profit'].cummax() - data['cumm_profit']
        max_dd_percent = data['cumm_profit_perc'].cummax() - data['cumm_profit_perc']
        return max_dd.max(), max_dd_percent.max()


    def _calculate_max_runup(self, data : pd.DataFrame):
        max_runup = data['cumm_profit'] - data['cumm_profit'].cummin()
        max_runup_percent = data['cumm_profit_perc'] - data['cumm_profit_perc'].cummin()
        return max_runup.max(), max_runup_percent.max()


    def compute_trades_report(self, trades_history : list[Trade]):
        """
        Compute and return a trades report including all trades and trades per ticker.

        Returns:
        - all_trades (pd.DataFrame): DataFrame containing properties of all trades.
        - trades_per_ticker (dict): Dictionary with tickers as keys and trade DataFrames as values.
        """
        # Initialize an empty list to store trade properties for all trades
        trades_list = []

        # Extract the 'trades' history from the object
        history = trades_history

        # Iterate through each trade in the history
        for index in range(len(history)):
            # Access the trade object
            trade: Trade = history[index]

            # Extract relevant properties from the trade
            properties = {
                'id': trade.id,
                'direction': trade.direction.value,
                'ticker': trade.ticker,
                'entry_timestamp': trade.entry_timestamp.tz_localize(None),
                'exit_timestamp': trade.exit_timestamp.tz_localize(None),
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
            'entry_timestamp': 'datetime64[ns]',
            'exit_timestamp': 'datetime64[ns]',
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

        # Return the overall trades DataFrame and the dictionary of trades per ticker
        return trades


    def compute_metrics_report(self, trades_report:pd.DataFrame):
        
        trades = trades_report

        net_profit = trades['profit'].sum()         # Net Profit
        gross_profit = trades[trades['profit'] > 0]['profit'].sum() # Gross Profit
        gross_loss = trades[trades['profit'] < 0]['profit'].sum() # Gross Loss
        average_pnl = trades['profit'].max() # Avg Trade
        average_profit = trades[(trades['profit'] > 0)]['profit'].mean() # Avg Winning Trade
        average_loss = trades[(trades['profit'] < 0)]['profit'].mean() # Avg Losing Trade
        largest_profit = trades[(trades['profit'] > 0)]['profit'].max() # Largest Winning Trade
        largest_loss = trades[(trades['profit'] < 0)]['profit'].min() # Largest Losing Trade
        net_profit_percent = trades['profit_percent'].sum() # Net Profit
        gross_profit_percent = trades[trades['profit_percent'] > 0]['profit_percent'].sum() # Gross Profit
        gross_loss_percent = trades[trades['profit_percent'] < 0]['profit_percent'].sum() # Gross Loss
        average_pnl_percent = trades['profit_percent'].max() # Avg Trade
        average_profit_percent = trades[(trades['profit_percent'] > 0)]['profit_percent'].mean() # Avg Winning Trade
        average_loss_percent = trades[(trades['profit_percent'] < 0)]['profit_percent'].mean() # Avg Losing Trade
        largest_profit_percent = trades[(trades['profit_percent'] > 0)]['profit_percent'].max() # Largest Winning Trade
        largest_loss_percent = trades[(trades['profit_percent'] < 0)]['profit_percent'].min() # Largest Losing Trade
        max_runup, max_runup_percent = self._calculate_max_runup(trades) # Max Run-up
        max_drawdown, max_drawdown_percent = self._calculate_max_drawdown(trades) # Max Drawdown

        profit_factor = gross_profit / abs(gross_loss) # Profit Factor
        total_closed_trades = len(trades) # Total Closed Trades
        count_profit = len(trades[trades['profit'] > 0]) # Number Winning Trades
        count_loss = len(trades[trades['profit'] < 0]) # Number Losing Trades
        count_breakeven = len(trades[trades['profit'] == 0]) # Number Breakeven Trades
        win_rate = (count_profit / total_closed_trades) * 100 # Percent Profitable
        average_win_rate_percent = average_profit / average_loss # Ratio Avg Win / Avg Loss
        average_duration = trades['duration'].mean().total_seconds() # Avg Duration in Trades
        average_duration_winning = trades[(trades['profit'] > 0)]['duration'].mean().total_seconds() # Avg Duration in Winning Trades
        average_duration_losing = trades[(trades['profit'] < 0)]['duration'].mean().total_seconds()  # Avg Duration in Losing history
          
        # Create a dictionary for each set of metrics
        metrics_data = {
            # NET PROFIT
            'net_profit': [net_profit],
            'gross_profit': [gross_profit],
            'gross_loss': [gross_loss],
            'average_pnl': [average_pnl],
            'average_profit': [average_profit],
            'average_loss': [average_loss],
            'largest_profit': [largest_profit],
            'largest_loss': [largest_loss],

            # PERCENTAGE
            'net_profit_percent': [net_profit_percent],
            'gross_profit_percent': [gross_profit_percent],
            'gross_loss_percent': [gross_loss_percent],
            'average_pnl_percent': [average_pnl_percent],
            'average_profit_percent': [average_profit_percent],
            'average_loss_percent': [average_loss_percent],
            'largest_profit_percent': [largest_profit_percent],
            'largest_loss_percent': [largest_loss_percent],

            # OTHERS
            'max_runup': [max_runup],
            'max_runup_percent': [max_runup_percent],
            'max_drawdown': [max_drawdown],
            'max_drawdown_percent': [max_drawdown_percent],
            'profit_factor': [profit_factor],
            'total_closed_trades': [total_closed_trades],
            'count_profit': [count_profit],
            'count_loss': [count_loss],
            'count_breakeven': [count_breakeven],
            'win_rate': [win_rate],
            'average_win_rate_percent': [average_win_rate_percent],
            'average_duration': [average_duration],
            'average_duration_winning': [average_duration_winning],
            'average_duration_losing': [average_duration_losing]
        }

        metrics = pd.DataFrame(metrics_data, index=['Overall'])

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
            'win_rate': float,
            'average_win_rate_percent': float,
            'average_duration': float,
            'average_duration_winning': float,
            'average_duration_losing': float
        })
        
        return metrics
    

    def report(self):
        return self.reports
    
    
    def plot_result(self, array):
        import matplotlib.pyplot as plt

        # Generate x-axis values (assuming indices as x-axis)
        x_values = np.arange(len(array))

        # Plot the values
        plt.plot(x_values, array, label='Equity Curve')

        # Add labels and title
        plt.xlabel('Time')
        plt.ylabel('Equity Value')
        plt.title('Equity Curve Plot')

        # Add a legend
        plt.legend()

        # Show the plot
        plt.show()
    

    def plot_results(self, data_dict):
        '''
        Plot results from a dictionary of NumPy arrays.

        Parameters:
        - data_dict (dict): A dictionary where keys are plot titles and values are NumPy arrays.
        '''

        # Create a new plot
        plt.figure()

        # Iterate through the dictionary items
        for title, values in data_dict.items():
            # Plot the values
            plt.plot(values, label=title)

        # Add labels and legend
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.legend()

        # Show the plot
        plt.show()
