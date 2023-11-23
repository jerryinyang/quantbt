import pandas as pd
from dateutil.parser import parse
import pytz
import numpy as np
from orders import Order
from trades import Trade
from utils import Bar, Log as log
from typing import List, Dict


exectypes = Order.ExecType

class Engine:
    TZ = pytz.timezone('UTC')

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
    family_id = 0

    def __set_dataframe_timezone(self, dataframes:dict[str,pd.DataFrame], date_range):

        for ticker, data in dataframes.items():
            df = pd.DataFrame(index=date_range)
    
            data.index = pd.to_datetime(data.index.strftime(self.DATE_FORMAT)).tz_localize(self.TZ)
            data.rename(columns={old_column : old_column.lower() for old_column in list(data.columns)}, inplace=True)
            data = data[['open', 'high', 'low', 'close', 'volume']]

            df = df.join(data, how='left').ffill().bfill()
            df['change_percent'] = df['close'].pct_change(fill_method=None).fillna(0)
            dataframes[ticker] = df

        return dataframes


    def __init__(self, tickers:list[str], dataframes:dict[str,pd.DataFrame], resolution:str, start_date, end_date) -> None:
        self.resolution = resolution
        self.tickers = tickers
        self.date_range = pd.date_range(start=parse(start_date).astimezone(self.TZ), 
                                        end=parse(end_date).astimezone(self.TZ), 
                                        freq=self.resolution, tz=self.TZ, normalize=True)
        self.dataframes = self.__set_dataframe_timezone(dataframes, self.date_range)

        self.portfolio : pd.DataFrame = None
        self.orders : List[Order] = []
        self.trades : Dict[int, Order] = {} # Syntax : Dict [ trade_id : Trade()]
        self.history : Dict[str , List[Order]] = {
            'orders' : [],
            'trades' : []
        }
        
    
    def init_portfolio(self, date_range:pd.DatetimeIndex):
        # Initialize Portfolio Dataframe: this would contain all the portfolio attributes
        portfolio =  pd.DataFrame({'timestamp': date_range}) # Initialize the full date range for the backtest
        portfolio.loc[0, 'balance'] = self.CAPITAL  # Initialize the backtest capital (initial balance)
        portfolio.loc[0, 'equity'] =  self.CAPITAL  # Initialize the equity
        portfolio.loc[0, 'pnl'] = 0

        return portfolio


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
            self.dataframes[ticker]['eligibility'] = check_changes(self.dataframes[ticker], ['open', 'high', 'low', 'close', 'volume'])
            self.dataframes[ticker]['eligibility'] = self.dataframes[ticker]['eligibility'].astype(int)
            
        return          

    
    def run_backtest(self):
        print('Initiating Backtest')

        # Set Backtest Range
        self.portfolio= self.init_portfolio(self.date_range)
        
        # Iterate through each bar/index (timestamp) in the backtest range
        for bar_index in self.portfolio.index:
            date = self.portfolio.loc[bar_index, 'timestamp']
            
            # Create the bar objects for easier access to data
            bars = {}
            for ticker in tickers:

                bar = Bar(
                    open=self.dataframes[ticker].loc[date, 'open'],
                    high=self.dataframes[ticker].loc[date, 'high'],
                    low=self.dataframes[ticker].loc[date, 'low'],
                    close=self.dataframes[ticker].loc[date, 'close'],
                    volume=self.dataframes[ticker].loc[date, 'volume'],
                    index=bar_index,
                    timestamp=date,
                    resolution=self.resolution,
                    ticker=ticker
                )
                bars[ticker] = bar

            if bar_index > 0:
                previous_date = self.portfolio.loc[bar_index - 1, 'timestamp']
                # Manage Pending Orders, Update Open Orders
                self.compute_positions_stats(bars, bar_index)
                

                # Update self.portfolio Attributes (capital, pnl)
                self.compute_portfolio_stats(self.portfolio, bar_index,  date, previous_date)
 

            # Filter Asset Universe
            self.filter_asset_universe(self.date_range)
            
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

                print('Next')

        print('Backtest Complete')


    def buy(self, bar, price, size:float, order_type:Order.ExecType, 
            stoplimit_price:float=None, parent_id:str=None,
            exit_profit:float=None, exit_loss:float=None,
            trailing_percent:float=None, family_role=None, 
            family_id=None, expiry_date=None) -> Order:
        
        if size:
            self.order_id += 1
            order = Order( # Create New Order Object 
                self.order_id, bar, Order.Direction.Long, price, order_type, size,
                stoplimit_price, parent_id, exit_profit, exit_loss,
                trailing_percent, family_role, 
                family_id, expiry_date
            )

            # Add order into self.orders collection
            order.accept()
            self.orders.append(order)

            return order

        return None 
    

    def sell(self, bar, price, size:float, order_type:Order.ExecType, 
            stoplimit_price:float=None, parent_id:str=None,
            exit_profit:float=None, exit_loss:float=None,
            trailing_percent:float=None, family_role=None, 
            family_id=None, expiry_date=None) -> Order:
        
        if size:
            self.order_id += 1
            order = Order( # Create New Order Object 
                self.order_id, bar, Order.Direction.Short, price, order_type, size,
                stoplimit_price, parent_id, exit_profit, exit_loss,
                trailing_percent, family_role, 
                family_id, expiry_date
            ) 

            # Add order into self.orders collection
            order.accept()
            self.orders.append(order)

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

            pnl += ticker_pnl # TODO: Use self.trades to sum up all the open trades pnl

        portfolio.loc[bar_index, 'balance'] = portfolio.loc[bar_index - 1, 'balance'] + pnl
        portfolio.loc[bar_index, 'pnl'] = pnl

        return pnl


    def compute_orders(self, bars : dict[Bar], bar_index:int):
        # TODO : Simulate realtime order execution (sorted)

        cancelled_orders = []
        rejected_orders = []
        filled_orders = []
        
        # Manage Orders
        for order in self.orders:
            bar = bars[order.ticker]

            if (order in cancelled_orders) or (order in rejected_orders) or (order in filled_orders):
                # Order has already been removed
                continue
            
            # Checks if Order is Expired
            if order.expired(bar.timestamp):
                # Add the order to cancelled orders
                cancelled_orders.append(order)
                log.info(f"Order {order.id} Expired, and has been cancelled")
                continue

            # If order is a child order (order.parent_id is set)
            # Check if the parent order is active, skip if not
            if order.parent_id:
                if order.parent_id not in self.trades.keys():
                    continue

                # If the parent order is active, Handle Child Orders (Take Profit, Stop Loss, Trailing)
                else:
                    # Get the parent order, and other children orders where applicable
                    parent = self.trades[order.parent_id]
                    children = [child for child in self.orders if (child.parent_id == order.parent_id)] # Find orders with the same parent_id
                    
                    # Check if order is filled
                    filled, fill_price = order.filled(bar) 
                    if filled:
                        order.complete()
                        order.price = fill_price # TODO: Maybe redundant, except slippage is to be applied

                        # Close Parent Trade
                        self.close_trade(parent) # TODO : Implement close_trade

                        # Add children to the list of cancelled orders
                        cancelled_orders.extend(children)
                        log.info(f"Trade {parent.id} Closed. Its children orders [{children}] have been cancelled")
 

            # Checks if order.price is filled and if number of open trades is less than the maximum allowed
            filled, fill_price = order.filled(bar) 
            if filled and (len(self.trades) < self.PYRAMIDING):
                # Checks if current balance can accomodate the risk amount 
                # (order.size * current price (fill price for limit orders)))
                if (self.portfolio.loc[bar.index, 'balance'] > (order.size * fill_price)):
                    order.price = fill_price # TODO: Maybe redundant, except slippage is to be applied
                    
                    # Execute the order
                    self.execute_order(order)
                    
                    # Add order to filled orders
                    filled_orders.append(order)
                    log.info(f"Order {order.id} Filled Successfully.")

                else:
                    # Not Enough Cash to take the Trade
                    # Add the order to rejected orders
                    rejected_orders.append(order)
            
            elif (len(self.trades) > self.PYRAMIDING):
                log.info(f'Maximum Open Trades reached. Order {order.id} has been skipped.')
    
        # Update self.orders
        self.manage_orders(cancelled_orders, rejected_orders, filled_orders)

        # Update Trades
        for trade in self.trades.values():
            self.update_trade(trade)
            

    def compute_trades_stats(self, bars : dict[Bar], bar_index:int):
        
        for trade in self.trades.values():
            # Update Trade Parameters
            bar = bars[trade.ticker]

            pnl = (bar.close - trade.entry_price) * trade.size
            trade.params.commission = 0 # TODO: Model Commissions

            trade.params.max_runup = max(trade.params.max_runup, pnl) # Highest PnL Value During Trade
            trade.params.max_runup_perc = (trade.params.max_runup / (trade.entry_price * trade.size)) * 100 # Highest PnL Value During Trade / (Entry Price x Quantity) * 100
            trade.params.max_drawdown = min(trade.params.max_drawdown, pnl) # Lowest PnL Value During Trade
            trade.params.max_drawdown_perc = (trade.params.max_drawdown / (trade.entry_price * trade.size)) * 100 # Lowest PnL Value During Trade / (Entry Price x Quantity) * 100

            trade.params.pnl = pnl
            trade.params.pnl_perc = (trade.params.pnl / self.portfolio.loc[bar.index, 'balance']) * 100
            

    def cancel_order(self, order:Order):
        '''
        Adds Cancelled/Rejected Order to History
        '''
        self.history['orders'].append(order)


    def execute_order(self, order:Order, bar:Bar):
        '''
        Generates Trade Objects from Order Object
        '''
        self.trade_id += 1
        new_trade = None

        # Check If Order Contains Child Orders
        if order.children_orders:
            self.family_id += 1
            # Send the child orders to the engine
            if order.direction == Order.Direction.Long:
                # Exit Profit Order
                self.sell(bar, stoplimit_price=None, parent_id=self.trade_id, 
                          exit_profit=None, exit_loss=None, trailing_percent=None, 
                          expiry_date=None, family_id=self.family_id,
                          **order.children_orders['exit_profit'])
                # Exit Loss Order
                self.sell(bar, stoplimit_price=None, parent_id=self.trade_id, 
                          exit_profit=None, exit_loss=None, trailing_percent=None, 
                          expiry_date=None, family_id=self.family_id,
                          **order.children_orders['exit_loss'])
            
            else:
                # Exit Profit Order
                self.buy(bar, stoplimit_price=None, parent_id=self.trade_id, 
                          exit_profit=None, exit_loss=None, trailing_percent=None, 
                          expiry_date=None, family_id=self.family_id,
                          **order.children_orders['exit_profit'])
                # Exit Loss Order
                self.buy(bar, stoplimit_price=None, parent_id=self.trade_id, 
                          exit_profit=None, exit_loss=None, trailing_percent=None, 
                          expiry_date=None, family_id=self.family_id,
                          **order.children_orders['exit_loss'])
                
            # Execute the parent order
            new_trade = Trade(self.trade_id, order, timestamp=bar.timestamp, 
                  exit_price=None, family_id=self.family_id)
        
        else:
            # Execute the Trade
            new_trade = Trade(self.trade_id, order, timestamp=bar.timestamp)
        
        # Add Trade to self.trades
        self.trades[self.trade_id] = new_trade
        log.info(f'Trade {new_trade.id} Executed.')     


    def manage_orders(self, cancelled_orders:List[Order], rejected_orders:List[Order], filled_orders:List[Order]):
        removed_orders = cancelled_orders + rejected_orders + filled_orders

        # Remove these orders from self.orders
        self.orders = list(set(self.orders) - set(removed_orders))

        # Adds the orders in the history
        for order in cancelled_orders:
            order.cancel()
            self.cancel_order(order)
            
        for order in rejected_orders:
            order.reject()
            self.cancel_order(order)

        for order in filled_orders:
            order.complete()
            self.cancel_order(order)        


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



if __name__ == '__main__':
    tickers = 'AAPL TSLA GOOG'.split(' ')
    ticker_path = [f'source/alpha/{ticker}.parquet' for ticker in tickers]

    # Read Data
    dataframes = dict(zip(tickers, [pd.read_parquet(path) for path in ticker_path]))

    alpha = Engine(tickers, dataframes, '1d', '2020-01-02', '2023-12-31')

    alpha.run_backtest()