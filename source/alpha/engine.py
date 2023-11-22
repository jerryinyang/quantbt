import pandas as pd
from dateutil.parser import parse
import pytz
import numpy as np
from orders import Order
from trades import Trade
from utils import Bar
from typing import List, Dict

exectypes = Order.ExecType

class Engine:
    TZ = pytz.utc
    DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
    CAPITAL = 100000
    PYRAMIDING = 100

    # FEES
    SWAP = 0
    COMMISSION = 0
    SPREAD = 0

    # PARAMETERS
    order_id = 0 # Initialize Order Counter
    trade_id = 0 # Initialize Trade Counter

    def __set_dataframe_timezone(self, dataframes:dict[str,pd.DataFrame]):

        for ticker, df in dataframes.items():

            df.index = pd.to_datetime(df.index.strftime(self.DATE_FORMAT)).tz_localize(self.TZ)
            df.rename(columns={old_column : old_column.lower() for old_column in list(df.columns)}, inplace=True)
            df = df[['open', 'high', 'low', 'close', 'volume']]
            dataframes[ticker] = df

        return dataframes


    def __init__(self, tickers:list[str], dataframes:dict[str,pd.DataFrame], resolution:str, start_date, end_date) -> None:
        self.resolution = resolution
        self.tickers = tickers
        self.dataframes = self.__set_dataframe_timezone(dataframes)
        self.start_date = parse(start_date, tzinfos={'UTC': self.TZ})
        self.end_date = parse(end_date, tzinfos={'UTC': self.TZ})

        self.portfolio : pd.DataFrame = None
        self.orders : List[Order] = None
        self.trades : Dict[str, List[Order]] = {
            'active' : [],
            'closed' : []
        }

    
    def init_portfolio(self, date_range:pd.DatetimeIndex):
        # Initialize Orders
        orders = [] # Manages All The Orders

        # Initialize Trades
        trades = [[],[]]

        # Initialize Portfolio Dataframe: this would contain all the portfolio attributes
        portfolio =  pd.DataFrame({'timestamp': date_range}) # Initialize the full date range for the backtest
        portfolio.loc[0, 'balance'] = self.CAPITAL  # Initialize the backtest capital (initial balance)
        portfolio.loc[0, 'equity'] =  self.CAPITAL  # Initialize the equity
        portfolio.loc[0, 'pnl'] = 0

        return portfolio, orders, trades


    def filter_asset_universe(self, date_range:pd.DatetimeIndex) -> None:
        '''
        Compute Available/ / Eligible Assets to Trade (apply Universe Filtering Rules).\n
        Asset Eligibility can include asset tradable days (crypto -> 24/7, forex -> 24/5, futures -> Market Hours), holidays, etc.
        In summary, this will check which data is available for trading in what days, and mark them as eligible.
        '''
        def check_changes(row, columns):
            changes = all(row[columns] != row[columns].shift(1))
            return changes
        
        for ticker in self.tickers:
            df = pd.DataFrame(index=date_range)
    
            self.dataframes[ticker] = df.join(self.dataframes[ticker], how='left').ffill().bfill()
            self.dataframes[ticker]['change_percent'] = self.dataframes[ticker]['close'].pct_change()
            self.dataframes[ticker]['eligibility'] = check_changes(self.dataframes[ticker], ['open', 'high', 'low', 'close', 'volume'])
            self.dataframes[ticker]['eligibility'] = self.dataframes[ticker]['eligibility'].astype(int)
            
        return          

    
    def run_backtest(self):
        print('Initiating Backtest')

        # Set Backtest Range
        backtest_range = pd.date_range(start=self.start_date, end=self.end_date, freq=self.resolution, tz=self.TZ) # Full Backtest Date Range
        self.portfolio, self.orders, self.trades = self.init_portfolio(backtest_range)
        
        
        # Iterate through each bar/index (timestamp) in the backtest range
        for bar_index in self.portfolio.index:
            date = self.portfolio.loc[bar_index, 'timestamp']
            
            # Create the bar objects for easier access to data
            bars = {}
            for ticker in tickers:
                bar = Bar(
                    self.dataframes[ticker].loc[date, 'open'],
                    self.dataframes[ticker].loc[date, 'high'],
                    self.dataframes[ticker].loc[date, 'low'],
                    self.dataframes[ticker].loc[date, 'close'],
                    self.dataframes[ticker].loc[date, 'volume'],
                    bar_index,
                    date,
                    self.resolution
                )
                bars[ticker] = bar

            if bar_index > 0:
                previous_date = self.portfolio.loc[bar_index - 1, 'timestamp']
                # Manage Pending Orders, Update Open Orders
                self.compute_position_stats(bars, bar_index)

                # Update self.portfolio Attributes (capital, pnl)
                self.compute_portfolio_stats(self.portfolio, bar_index,  date, previous_date)
 

            # Filter Asset Universe
            self.filter_asset_universe(backtest_range)
            
            eligible_assets = [ticker for ticker in self.tickers if self.dataframes[ticker].loc[date, 'eligibility']]
            non_eligible_assets = list(set(self.tickers) - set(eligible_assets))

            # Decision-making / Signal-generating Algorithm
            alpha_long, alpha_short = self.signal_generator(eligible_assets)
            
            non_eligible_assets = list(set(non_eligible_assets).union((set(eligible_assets) - set(alpha_long + alpha_short)))  )
            eligible_assets = list(set(alpha_long + alpha_short))

            # Executing Signals
            for ticker in non_eligible_assets:
                # Units of asset in holding (Set to zero)
                self.portfolio.loc[bar_index, f"{ticker} units"] = 0
            
            for ticker in eligible_assets:

                # Calculate Allocation for Each Symbols (Equal Allocation)
                risk_dollars = self.portfolio.loc[bar_index, 'balance'] / (len(alpha_long) + len(alpha_short)) 

                # Tickers to Long
                if ticker in alpha_long:
                    entry_price = bars[ticker].close
                    position_size = 1 * risk_dollars / entry_price
                    
                    # Create and Send Long Order
                    self.buy(bars[ticker], entry_price, position_size, exectypes.Market)
                    
                
                 # Tickers to Short
                elif ticker in alpha_short:
                    entry_price = bars[ticker].close
                    position_size = -1 * risk_dollars / entry_price
                    
                    # Create and Send Short Order
                    self.sell(bars[ticker], entry_price, position_size, exectypes.Market)

                # Update Ticker Units in Portfolio
                self.portfolio.loc[bar_index, f"{ticker} units"] = position_size
        
        print('Backtest Complete')


    def buy(self, bar, price, size:float, order_type:Order.ExecType, 
            stoplimit_price:float=None, parent_id:str=None,
            exit_profit:float=None, exit_loss:float=None,
            trailing_percent:float=None, bracket_role=None, expiry_date=None) -> Order:
        
        if size:
            self.order_id += 1
            order = Order( # Create New Order Object 
                self.order_id, bar, Order.Direction.Long, price, order_type, size,
                stoplimit_price, parent_id, exit_profit, exit_loss,
                trailing_percent, bracket_role=None, expiry_date=None
            )

            # Add order into self.orders collection
            order.accept()
            self.orders.appendleft(order)

            return order

        return None 
    

    def sell(self, bar, price, size:float, order_type:Order.ExecType, 
            stoplimit_price:float=None, parent_id:str=None,
            exit_profit:float=None, exit_loss:float=None,
            trailing_percent:float=None, bracket_role=None, expiry_date=None) -> Order:
        
        if size:
            self.order_id += 1
            order = Order( # Create New Order Object 
                self.order_id, bar, Order.Direction.Short, price, order_type, size,
                stoplimit_price, parent_id, exit_profit, exit_loss,
                trailing_percent, bracket_role=None, expiry_date=None
            ) 

            # Add order into self.orders collection
            order.accept()
            self.orders.appendleft(order)

            return order

        return None 
        

    def signal_generator(self, eligibles:list[str]) -> tuple:
        import numpy as np

        alpha_scores = { key : np.random.rand() for key in eligibles}

        alpha_scores = {key : value for key, value in sorted(alpha_scores.items(), key=lambda item : item[1])} # Sorts the dictionary
        list_scores = list(alpha_scores.keys())
        
        alpha_long = [asset for asset in list_scores if alpha_scores[asset] >= .8]
        alpha_short = [asset for asset in list_scores if alpha_scores[asset] <= .2]

        return alpha_long, alpha_short
             
  
    def compute_portfolio_stats(self, portfolio, bar_index,  date, prev_date) -> tuple:
        pnl = 0

        for ticker in self.tickers:
            units_traded = portfolio.loc[bar_index - 1, f"{ticker} units"]
            if units_traded == 0:
                continue

            delta = self.dataframes[ticker].loc[date, 'close'] - self.dataframes[ticker].loc[prev_date, 'close']
            ticker_pnl = delta * units_traded

            pnl += ticker_pnl
 

        portfolio.loc[bar_index, 'capital'] = portfolio.loc[bar_index - 1, 'capital'] + pnl
        portfolio.loc[bar_index, 'pnl'] = pnl

        
        return pnl


    def compute_position_stats(self, bars : dict[Bar], bar_index:int):
        # TODO : Simulate realtime order execution (sorted)

        # Loop Through Code
        order_index = 0
        
        while order_index < len(self.orders):
            order = self.orders[order_index]
            bar = bars[order.ticker]
            
            # Checks if Order is Expired
            if order.expired(bar.timestamp):
                # Cancel Order 
                order.cancel()

                # Add it to the closed_trades list; self.trades = [ [active_trades] , [closed_trades] ]
                self.trades['closed'].append(order)

                # Delete order from self.orders
                self.orders.pop(order_index)
                print(f'Order {order.id} Cancelled : Order Expired.')
                
                # Next Order
                break

            # Checks if order.price is filled and if number of open trades is less than the maximum allowed
            filled, fill_price = order.filled(bar) 
            if filled and (len(self.trades['active']) < Engine.PYRAMIDING):
                # Checks if current balance can accomodate the risk amount 
                # (order.size * current price (fill price for limit orders)))
                if (self.portfolio.loc[bar.index, 'balance'] > order.size * fill_price):
                    order.price = fill_price
                    order.complete()

                    # Add it to the closed_trades list; self.trades = [ [active_trades] , [closed_trades] ]
                    self.trades['active'].append(order)

                    # Delete order from self.orders
                    self.orders.pop(order_index)
                    print(f'Order {order.id} Rejected : Order is too Large.')

                else:
                    # Not Enough Cash to take the Trade
                    # Cancel Order 
                    order.reject()

                    # Add it to the closed_trades list; self.trades = [ [active_trades] , [closed_trades] ]
                    self.trades['closed'].append(order)

                    # Delete order from self.orders
                    self.orders.pop(order_index)
                    print(f'Order {order.id} Rejected : Order is too Large.')

            order_index += 1
    

    def execute_order(self, bar:Bar, order:Order):
        '''
        Generates Trade Objects from Order Object
        '''
        self.trade_id += 1

        Trade(self.trade_id, order, bar, exit_price=None)
        pass


    def plot(self, array):
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


# Default Class
class BaseAlpha:
    TZ = pytz.utc
    DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

    def __init__(self, tickers:list[str], dataframes:dict[str,pd.DataFrame], resolution:str, start_date, end_date) -> None:
        self.resolution = resolution
        self.tickers = tickers
        self.dataframes = self.__set_dataframe_timezone(dataframes)
        self.start_date = parse(start_date, tzinfos={'UTC': self.TZ})
        self.end_date = parse(end_date, tzinfos={'UTC': self.TZ})

        self.capital = 10000

    
    def init_portfolio(self, date_range:pd.DatetimeIndex):
        # Initialize Portfolio Dataframe: this would contain all the portfolio attributes

        portfolio =  pd.DataFrame({'timestamp': date_range}) # Initialize the full date range for the backtest
        portfolio.loc[0, 'capital'] = 10000 # Initialize the backtest capital

        return portfolio


    def __set_dataframe_timezone(self, dataframes:dict[str,pd.DataFrame]):

        for ticker, df in dataframes.items():

            df.index = pd.to_datetime(df.index.strftime(self.DATE_FORMAT)).tz_localize(self.TZ)
            df.rename(columns={old_column : old_column.lower() for old_column in list(df.columns)}, inplace=True)
            df = df[['open', 'high', 'low', 'close', 'volume']]
            dataframes[ticker] = df

        return dataframes


    def filter_asset_universe(self, date_range:pd.DatetimeIndex) -> None:
        '''
        Compute Available/ / Eligible Assets to Trade (apply Universe Filtering Rules).\n
        Asset Eligibility can include asset tradable days (crypto -> 24/7, forex -> 24/5, futures -> Market Hours), holidays, etc.
        In summary, this will check which data is available for trading in what days, and mark them as eligible.
        '''
        def check_changes(row, columns):
            changes = all(row[columns] != row[columns].shift(1))
            return changes
        
        for ticker in self.tickers:
            df = pd.DataFrame(index=date_range)
    
            self.dataframes[ticker] = df.join(self.dataframes[ticker], how='left').ffill().bfill()
            self.dataframes[ticker]['return'] = self.dataframes[ticker]['close'].pct_change()
            self.dataframes[ticker]['eligibility'] = check_changes(self.dataframes[ticker], ['open', 'high', 'low', 'close', 'volume'])
            self.dataframes[ticker]['eligibility'] = self.dataframes[ticker]['eligibility'].astype(int)
            
        return          

    
    def run_backtest(self):
        print('Initiating Backtest')

        # Set Backtest Range
        backtest_range = pd.date_range(start=self.start_date, end=self.end_date, freq=self.resolution, tz=self.TZ) # Full Backtest Date Range
        portfolio = self.init_portfolio(backtest_range)

        equity = np.array([])
        
        # Iterate through each bar/index (timestamp) in the backtest range
        for bar_index in portfolio.index:
            date = portfolio.loc[bar_index, 'timestamp']

            if bar_index > 0:
                # Manage Pending Orders
                # Update Open Orders
                # Update Portfolio Attributes (capital)

                # Compute the PnL
                previous_date = portfolio.loc[bar_index - 1, 'timestamp']
                pnl, capital_return = self.compute_pnl_stats(portfolio, bar_index,  date, previous_date)
                equity = np.append(equity, portfolio.loc[bar_index, 'capital'])

            # Filter Asset Universe
            self.filter_asset_universe(backtest_range)
            
            eligible_assets = [ticker for ticker in self.tickers if self.dataframes[ticker].loc[date, 'eligibility']]
            non_eligible_assets = list(set(self.tickers) - set(eligible_assets))

            # Decision-making / Signal-generating Algorithm
            alpha_long, alpha_short = self.signal_generator(eligible_assets)
            
            non_eligible_assets = list(set(non_eligible_assets).union((set(eligible_assets) - set(alpha_long + alpha_short)))  )
            eligible_assets = list(set(alpha_long + alpha_short))

            for ticker in non_eligible_assets:
                # Units of asset in holding (Set to zero)
                portfolio.loc[bar_index, f"{ticker} weight"] = 0
                portfolio.loc[bar_index, f"{ticker} units"] = 0
            
            total_nominal_value = 0

            for ticker in eligible_assets:
                direction = 1 if ticker in alpha_long else -1 if ticker in alpha_short else 0

                risk_dollars = portfolio.loc[bar_index, 'capital'] / (len(alpha_long) + len(alpha_short))
                position_size = direction * risk_dollars / self.dataframes[ticker].loc[date, 'close']
                portfolio.loc[bar_index, f"{ticker} units"] = position_size
                total_nominal_value += abs(position_size * self.dataframes[ticker].loc[date, 'close'])
            
            for ticker in eligible_assets:
                units = portfolio.loc[bar_index, f"{ticker} units"]
                ticker_nominal_value = units * self.dataframes[ticker].loc[date, 'close']
                instrument_weight = ticker_nominal_value / total_nominal_value 
                portfolio.loc[bar_index, f"{ticker} weight"] = instrument_weight

            portfolio.loc[bar_index, 'total nominal value'] = total_nominal_value
            portfolio.loc[bar_index, 'leverage'] = total_nominal_value / portfolio.loc[bar_index, 'capital']

        
        self.plot(equity)
        print('Backtest Complete')


    def signal_generator(self, eligibles:list[str]):
        import numpy as np

        alpha_scores = { key : np.random.rand() for key in eligibles}

        alpha_scores = {key : value for key, value in sorted(alpha_scores.items(), key=lambda item : item[1])} # Sorts the dictionary
        list_scores = list(alpha_scores.keys())
        
        alpha_long = [asset for asset in list_scores if alpha_scores[asset] >= .8]
        alpha_short = [asset for asset in list_scores if alpha_scores[asset] <= .2]

        return alpha_long, alpha_short
             
  
    def compute_pnl_stats(self, portfolio, bar_index,  date, prev_date):
        pnl = 0
        nominal_return = 0

        for ticker in self.tickers:
            units_traded = portfolio.loc[bar_index - 1, f"{ticker} units"]
            if units_traded == 0:
                continue

            # TODO : Use pct_change already calculated in filter_universe
            delta = self.dataframes[ticker].loc[date, 'close'] - self.dataframes[ticker].loc[prev_date, 'close']
            ticker_pnl = delta * units_traded

            pnl += ticker_pnl
            nominal_return += portfolio.loc[bar_index - 1, f"{ticker} weight"] * self.dataframes[ticker].loc[date, 'return']

        capital_return = nominal_return * portfolio.loc[bar_index - 1, 'leverage']

        portfolio.loc[bar_index, 'capital'] = portfolio.loc[bar_index - 1, 'capital'] + pnl
        portfolio.loc[bar_index, 'pnl'] = pnl
        portfolio.loc[bar_index, 'nominal_return'] = nominal_return
        portfolio.loc[bar_index, 'capital_return'] = capital_return
        
        return pnl, capital_return


    def plot(self, array):
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




if not __name__ == '__main__':
    exit()
    
tickers = 'AAPL TSLA GOOG'.split(' ')
ticker_path = [f'source/alpha/{ticker}.parquet' for ticker in tickers]

# Read Data
dataframes = dict(zip(tickers, [pd.read_parquet(path) for path in ticker_path]))

alpha = Engine(tickers, dataframes, '1d', '2020-01-01', '2023-12-31')

alpha.run_backtest()