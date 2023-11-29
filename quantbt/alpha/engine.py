import logging
import os
from bisect import bisect_left, bisect_right
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytz
from dateutil.parser import parse
from orders import Order
from trades import Trade
from utils import Bar  # noqa: F401
from utils import clear_terminal, debug, sorted_index  # noqa: F401

logging.basicConfig(filename='logs.log', level=logging.INFO)

exectypes = Order.ExecType

class Engine:
    TZ = pytz.timezone('UTC')

    DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
    CAPITAL = 100000
    PYRAMIDING = 1
    PRECISION = 2

    # FEES
    SWAP = 0
    COMMISSION = 0
    SPREAD = 0

    # PARAMETERS
    order_id = 0 # Initialize Order Counter
    trade_id = 0 # Initialize Trade Counter

    trade_count = 0

    def __set_data(self, dataframes:dict[str,pd.DataFrame], date_range):
        def check_values(df, columns):
                df_shifted = df[columns].shift(1)
        
                # Use the pandas DataFrame method 'ne' to create a new DataFrame where each cell is either True (if the corresponding cell in 'df' is not equal to the corresponding cell in 'df_shifted') or False.
                df_different = df[columns].ne(df_shifted)
                
                # Use the pandas DataFrame method 'any' to create a Series where each element is True if any cell in the corresponding row in 'df_different' is True, and False otherwise.
                return df_different.any(axis=1)

        # Modify Dataframes to Desired Format
        for ticker, data in dataframes.items():
            df = pd.DataFrame(index=date_range)
    
            data.index = pd.to_datetime(data.index.strftime(self.DATE_FORMAT)).tz_localize(self.TZ)
            data.rename(columns={old_column : old_column.lower() for old_column in list(data.columns)}, inplace=True)
            data = data[['open', 'high', 'low', 'close', 'volume']]

            df = df.join(data, how='left').ffill().bfill()
            df['price_change'] = df['close'] - df['close'].shift().fillna(0)
            dataframes[ticker] = df

            df['market_open'] = check_values(
                    df, 
                    ['open', 'high', 'low', 'close', 'volume']
                )
            df['market_open'] = df['market_open'].astype(int)

        # TODO : Initial Universe Filter : Select the tickers to be included in the backtest
        # Example : For Volume Based Strategies, exclude Forex data, because they don't aaply

        return dataframes


    def __init__(self, tickers:list[str], dataframes:dict[str,pd.DataFrame], resolution:str, start_date, end_date) -> None:
        self.resolution = resolution
        self.tickers = tickers
        self.date_range = pd.date_range(start=parse(start_date, ignoretz=True), 
                                        end=parse(end_date, ignoretz=True), 
                                        freq=self.resolution, tz=self.TZ)

        self.dataframes = self.__set_data(dataframes, self.date_range)

        self.portfolio : pd.DataFrame = None
        self.orders : List[Order] = []
        self.trades : Dict[str, Dict[int, Order]] = {key : {} for key in self.tickers} # Syntax : Dict[ticker, Dict [ trade_id : Trade()]]
        self.history : Dict[str , List[Order]|List[Trade]] = {
            'orders' : [],
            'trades' : []
        }

        # This would contain informations about sizing for each asset
        # Equal weighting : All assets take the same share of the assets
        self.tickers_weight = {ticker : 1 / len(self.tickers)  for ticker in self.tickers}
    
    
    def init_portfolio(self, date_range:pd.DatetimeIndex):
        # Initialize Portfolio Dataframe: this would contain all the portfolio attributes
        portfolio =  pd.DataFrame({'timestamp': date_range}) # Initialize the full date range for the backtest
        portfolio.loc[0, 'balance'] = self.CAPITAL  # Initialize the backtest capital (initial balance)
        portfolio.loc[0, 'equity'] =  self.CAPITAL  # Initialize the equity
        portfolio.loc[0, 'open_pnl'] = 0

        for ticker in self.tickers:
            portfolio.loc[0, f'{ticker} units'] = 0
            portfolio.loc[0, f'{ticker} open_pnl'] = 0
            portfolio.loc[0, f'{ticker} closed_pnl'] = 0

        return portfolio


    def filter_eligible_assets(self, date) -> None:
        '''
        Filter Eligible Assets to Trade (apply Universe Filtering Rules) on a specific day.\n
        Asset Eligibility can include asset tradable days (crypto -> 24/7, forex -> 24/5, futures -> Market Hours), holidays, etc.
        In summary, this will check which data is available for trading in what days, and mark them as eligible.
        '''
        # for ticker in self.tickers:
        #     self.dataframes[ticker]['market_open'] = check_values(
        #             self.dataframes[ticker], 
        #             ['open', 'high', 'low', 'close', 'volume']
        #         ).fillna(value=0)
        #     self.dataframes[ticker]['market_open'] = self.dataframes[ticker]['market_open'].astype(int)
        
        #     self.dataframes[ticker]['Day'] = self.dataframes[ticker].index.strftime('%A')    

        # return dataframes
        eligible_assets = [ticker for ticker in self.tickers if self.dataframes[ticker].loc[date, 'market_open']]
        non_eligible_assets = list(set(self.tickers) - set(eligible_assets))

        return eligible_assets, non_eligible_assets


    def run_backtest(self):
        print('Initiating Backtest')

        # Set Backtest Range
        self.portfolio= self.init_portfolio(self.date_range)
        
        # Iterate through each bar/index (timestamp) in the backtest range
        for bar_index in self.portfolio.index:
            date = self.portfolio.loc[bar_index, 'timestamp']

            # Create the bar objects for easier access to data
            bars = {}
            for ticker in self.tickers:

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

            # If Date is not the first date
            if bar_index > 0:
                # Update Portfolio
                self.compute_portfolio_stats(bar_index)

                # Process Orders
                self.compute_orders(bars)

                # Process Active Trades
                self.compute_trades_stats(bars)
            
            # Filter Universe for Assets Eligible for Trading in a specific date
            eligible_assets, non_eligible_assets = self.filter_eligible_assets(date)

            # Decision-making / Signal-generating Algorithm
            alpha_long, alpha_short = self.signal_generator(eligible_assets)
            
            non_eligible_assets = list(set(non_eligible_assets).union((set(eligible_assets) - set(alpha_long + alpha_short)))  )
            eligible_assets = list(set(alpha_long + alpha_short))

            # Executing Signals
            for ticker in non_eligible_assets:
                # Units of asset in holding (Set to zero)
                self.portfolio.loc[bar_index, f'{ticker} units'] = 0
                self.portfolio.loc[bar_index, f'{ticker} open_pnl'] = 0
            
            for ticker in eligible_assets:

                # Calculate Allocation for Each Symbols (Equal Allocation)
                risk_dollars = self.portfolio.loc[bar_index, 'balance'] * self.tickers_weight[ticker] 

                # Tickers to Long
                if ticker in alpha_long:
                    entry_price = bars[ticker].close

                    position_size = risk_dollars / entry_price

                    exit_tp = entry_price * 1.1
                    exit_sl = entry_price * 0.95
                    
                    # Create and Send Long Order
                    self.buy(bars[ticker], entry_price, position_size, exectypes.Market, exit_profit=exit_tp, exit_loss=exit_sl)                    

                 # Tickers to Short
                elif ticker in alpha_short:
                    entry_price = bars[ticker].close
                    position_size = -1 * risk_dollars / entry_price

                    exit_tp = entry_price * 0.9
                    exit_sl = entry_price * 1.05
                    
                    # Create and Send Short Order
                    self.sell(bars[ticker], entry_price, position_size, exectypes.Market, exit_profit=exit_tp, exit_loss=exit_sl)

            # logging.info(f'{self.portfolio.loc[bar.index]}')
        
        # Get the Equity Curve
        # self.plot_results({
        #     'Equity' : self.portfolio['equity'].to_numpy(),
        #     'Balance' : self.portfolio['balance'].to_numpy(),
        # })

        report_trades, report_metrics = self.compute_reports()
        report_trades.to_csv('data/results/report_trades.csv', index=True)
        report_metrics.to_csv('data/results/report_metrics.csv', index=True)

        print('Backtest Complete')



    def buy(self, bar, price, size:float, exectype:Order.ExecType, 
            stoplimit_price:float=None, parent_id:str=None,
            exit_profit:float=None, exit_loss:float=None,
            exit_profit_percent:float=None, exit_loss_percent:float=None,
            trailing_percent:float=None, family_role=None, 
            expiry_date=None) -> Order:
        
        
        if size or (exectype == exectypes.Market):
            self.order_id += 1
            order = Order( # Create New Order Object 
                self.order_id, bar, Order.Direction.Long, price, exectype, size,
                stoplimit_price, parent_id, exit_profit, exit_loss,
                exit_profit_percent, exit_loss_percent, trailing_percent, family_role, 
                expiry_date
            )
            
            # Add order into self.orders collection
            self._add_order(order)

            return order

        return None 
    

    def sell(self, bar, price, size:float, exectype:Order.ExecType, 
            stoplimit_price:float=None, parent_id:str=None,
            exit_profit:float=None, exit_loss:float=None,
            exit_profit_percent:float=None, exit_loss_percent:float=None,
            trailing_percent:float=None, family_role=None, 
            expiry_date=None) -> Order:
        
        if size or (exectype == exectypes.Market):
            self.order_id += 1
            order = Order( # Create New Order Object 
                self.order_id, bar, Order.Direction.Short, price, exectype, size,
                stoplimit_price, parent_id, exit_profit, exit_loss,
                exit_profit_percent, exit_loss_percent, trailing_percent, family_role, 
                expiry_date
            ) 

            # Add order into self.orders collection
            self._add_order(order)

            return order

        return None 
        

    def signal_generator(self, eligibles:list[str]) -> tuple:
        # import numpy as np

        # alpha_scores = { key : np.random.rand() for key in eligibles}

        # alpha_scores = {key : value for key, value in sorted(alpha_scores.items(), key=lambda item : item[1])} # Sorts the dictionary
        # list_scores = list(alpha_scores.keys())
        
        # alpha_long = [asset for asset in list_scores if alpha_scores[asset] >= .8]
        # alpha_short = [asset for asset in list_scores if alpha_scores[asset] <= .2]

        # return alpha_long, alpha_short

        return ['AAPL', 'GOOG'], []
             
  

    def compute_portfolio_stats(self, bar_index, exclude_closed:bool=False):

        # Iterate through active trade for open_pnl and equity
        total_open_pnl, total_closed_pnl = 0, 0
        
        # Get Units and PnL Values for Each Ticker
        for ticker in self.tickers:
            # Get list of active trades
            trades = self.trades[ticker]

            units, pnl = 0, 0
            for trade in trades.values():
                units += trade.size
                pnl += trade.params.pnl

            self.portfolio.loc[bar_index, f'{ticker} units'] = units
            self.portfolio.loc[bar_index, f'{ticker} open_pnl'] = pnl

            if not exclude_closed:
                self.portfolio.loc[bar_index, f'{ticker} closed_pnl'] = self.portfolio.loc[bar_index - 1, f'{ticker} closed_pnl']

            total_open_pnl += pnl
            total_closed_pnl += self.portfolio.loc[bar_index, f'{ticker} closed_pnl']

        # Update General Portfolio Stats
        self.portfolio.loc[bar_index, 'balance'] = self.CAPITAL + total_closed_pnl
        self.portfolio.loc[bar_index, 'equity'] = self.portfolio.loc[bar_index, 'balance'] + total_open_pnl
        self.portfolio.loc[bar_index, 'open_pnl'] = total_open_pnl # Total Unrealized PnL


    def compute_orders(self, bars : dict[Bar]):
        # TODO : Simulate realtime order execution (sorted)

        # Process Orders
        for order in self.orders:
            bar = bars[order.ticker]

            # Don't Update Orders when market is closed
            if not self.dataframes[ticker].loc[bar.timestamp, 'market_open']:
                continue

            rerun = self._process_order(order, bar)

            if rerun:
                return self.compute_orders(bars)


    def compute_trades_stats(self, bars : dict[Bar]) -> float:
        open_pnl = 0
        for ticker in self.tickers:
            ticker_trades = self.trades[ticker]
            bar = bars[ticker]

            # Don't Update Trades when market is closed
            if not self.dataframes[ticker].loc[bar.timestamp, 'market_open']:
                continue

            for trade in ticker_trades.values():
                # Update the trade
                open_pnl += self._update_trade(trade, bar)

        return open_pnl 



    # METHODS FOR ORDER PROCESSING
    def _add_order(self, order:Order | List[Order]):    
# If the new order is a market  
        # If the order is a market order, and 
        # The number of active trades for that ticker is at the maximum, reject the order
        if (not order.parent_id) and (order.exectype == exectypes.Market) and \
            (self.count_active_trades(order.ticker) >= self.PYRAMIDING):
            # Reject the order
            self._reject_order(order, f'Maximum Actice Trades for {order.ticker} Reached.')
            return
        
        # If Market is not open on that day
        if not self.dataframes[ticker].loc[order.timestamp, 'market_open']:
            # Reject the order
            self._reject_order(order, f'{order.ticker} Market is closed.')
            return
        
        # Accept Order
        self._accept_order(order)
    

    def _process_order(self,  order:Order, bar:Bar):

        # Checks if Order is Expired
        if order.expired(bar.timestamp):
            # Add the order to cancelled orders
            return self._expire_order(order)

        # If order is a child order (order.parent_id is set)
        # Check if the parent order is active, skip if not
        if (order.parent_id is not None) and  (order.parent_id in self.trades[order.ticker].keys()):
            # If the parent order is active, Handle Child Orders (Take Profit, Stop Loss, Trailing)
            # Check if order is filled

            filled, fill_price = order.filled(bar)
            if filled:
                order.price = fill_price

                # Get the parent order, and other children orders where applicable
                parent = self.trades[order.ticker][order.parent_id]

                # Execute The Appropriate Action For the Differnet Types of Children Orders
                
                # ChildExit : # Close Parent Trade, at the order.price
                if order.family_role == Order.FamilyRole.ChildExit:
                    self._close_trade(parent, bar, price=order.price)
                    
                    # Find orders with the same parent_id
                    children = [child for child in self.orders if (child.parent_id == order.parent_id)] 

                    # Cancel the other children orders
                    self._cancel_order(children)


                # TODO ChildReduce : Reduce the parent (as in partial exits, trailing)
                if order.family_role == Order.FamilyRole.ChildReduce:
                    pass           
                
                return None

        # Checks if order.price is filled and if number of open trades is less than the maximum allowed
        filled, fill_price = order.filled(bar)

        if filled:
            # For other exectypes, modify order.price to fill_price, without modifying the size
            if not (order.price == fill_price):
                order.price = fill_price
            
            # Check if trade is active
            if self.count_active_trades(order.ticker) < self.PYRAMIDING:                                
                
                # Insufficient Balance for the Trade
                if (self.portfolio.loc[bar.index, 'balance'] < (order.size * fill_price)):

                    # Resize the order, if it is a Market Order
                    if order.exectype == exectypes.Market:
                        order.size = self._recalculate_market_order_size(order, bar, fill_price)
                        order.price = fill_price

                    # For Pending Orders, cancel it
                    else: 
                        # Not Enough Cash to take the Trade
                        # Add the order to rejected orders
                        return self._cancel_order(order, f'Insufficient margin/balance for this position. {order.size * fill_price})')
                
                if (self.portfolio.loc[bar.index, 'balance'] >= (order.size * fill_price)):
                    # Add order to filled orders
                    self._fill_order(order)

                    # Execute the order
                    self._execute_trade(order, bar)

                    # Return True, as a signal to run through all orders again
                    return True

            else:
                return self._cancel_order(order, f'Maximum Open Trades reached. Order {order.id} has been skipped.')
            

    def _accept_order(self, order:Order | List[Order]):
        # If a list is passed, recursively accept each order
        if isinstance(order, list):
            for _order in order:
                self._accept_order(_order)
        
        # Base Condition (order is Order instant)
        else:
            # Accept the order
            order.accept()

            # Add it to self.orders
            # index = sorted_index(self.orders, order)
            # self.orders.insert(index, order)
            self.orders.append(order)

            # logging.info(f'Order {order.id} Accepted Successfully.')


    def _expire_order(self, order:Order | List[Order]):   
        # If a list is passed, recursively expire each order
        if isinstance(order, list):
            for _order in order:
                self._expire_order(_order)
        
        # Base Condition (order is Order instant)
        else:
            # Remove the order from self.orders
            self.orders.remove(order)

            # Expire the order
            order.cancel()

            # Add the order to orders history
            self.history['orders'].append(order)
            # logging.info(f'Order {order.id} Expired.')


    def _cancel_order(self, order:Order | List[Order], message:str=None):   
        # If a list is passed, recursively cancel each order
        if isinstance(order, list):
            for _order in order:
                self._cancel_order(_order)
        
        # Base Condition (order is Order instant)
        else:
            # Remove the order from self.orders
            self.orders.remove(order)

            # Cancel the order
            order.cancel()

            # Add the order to orders history
            message = f'\nReason : {message}' if message else ''
            self.history['orders'].append(order)
            # logging.info(f'Order {order.id} Cancelled.' + message)


    def _fill_order(self, order:Order | List[Order]):
        # If a list is passed, recursively fill each order
        if isinstance(order, list):
            for _order in order:
                self._fill_order(_order)
        
        # Base Condition (order is Order instant)
        else:
            # Remove the order from self.orders
            self.orders.remove(order)

            # Fill the order
            order.fill()

            # Add the order to orders history
            self.history['orders'].append(order)
            # logging.info(f'Order {order.id} Filled Successfully.')


    def _reject_order(self, order:Order | List[Order], message:str=None):
         # If a list is passed, recursively reject each order
        if isinstance(order, list):
            for _order in order:
                self._reject_order(_order)
        
        # Base Condition (order is Order instant)
        else:
            # Reject the order
            order.reject()

            # Add the order to orders history
            message = f'\nReason : {message}' if message else ''
            self.history['orders'].append(order)
            # logging.info(f'Order {order.id} Rejected.' + message)

    
    def _recalculate_market_order_size(self, order:Order, bar:Bar, fill_price:float):
        '''
        For market orders, the order size should be recalculated based on the fill price, to account for gaps in the market.
        '''
        
        # Get Ticker Weight
        weight = self.tickers_weight[order.ticker]

        # Calculate the risk in cash
        risk_amount = self.portfolio.loc[bar.index, 'balance'] * weight

        return order.direction.value * (risk_amount / fill_price)


    def _sort_for_execution(self, bar:Bar):
        price = bar.open

        # Split Orders Array by the bar.open
        below_price = self.orders[:bisect_left(self.orders, price)]
        above_price = self.orders[bisect_right(self.orders, price):]
        eq_price = [order for order in self.orders if order.price == price]
        below_price.reverse()
        
        # For Green Bars, execution is OLHC
        if (bar.open <= bar.close):
            return eq_price + below_price + above_price
        
        # For Red Bars, execution is OHLC
        return eq_price + above_price + below_price



    # METHODS FOR TRADE PROCESSING
    def _update_trade(self, trade:Trade, bar:Bar, price:float=None):
        # If a price is not passed, use the bar's close price
        price = price or bar.close

        pnl = (price - trade.entry_price) * trade.size
        trade.params.commission = 0 # TODO: Model Commissions

        trade.params.max_runup = max(trade.params.max_runup, pnl) # Highest PnL Value During Trade
        trade.params.max_runup_perc = (trade.params.max_runup / (trade.entry_price * trade.size)) * 100 # Highest PnL Value During Trade / (Entry Price x Quantity) * 100
        trade.params.max_drawdown = min(trade.params.max_drawdown, pnl) # Lowest PnL Value During Trade
        trade.params.max_drawdown_perc = (trade.params.max_drawdown / (trade.entry_price * trade.size)) * 100 # Lowest PnL Value During Trade / (Entry Price x Quantity) * 100

        trade.params.pnl = pnl
        trade.params.pnl_perc = (trade.params.pnl / self.portfolio.loc[bar.index, 'balance']) * 100

        return pnl    


    def _execute_trade(self, order:Order, bar:Bar):
        '''
        Generates Trade Objects from Order Object
        '''
        self.trade_id += 1
        new_trade = None
        _, fill_price = order.filled(bar)

        order_above = None
        order_below = None

        # Check If Order Contains Child Orders
        if order.children_orders:

            # If exit_profit_percent is not None, calculate the exit_profit price
            if order.children_orders['exit_profit']['exit_profit_percent'] is not None:
                price = order.price * order.children_orders['exit_profit']['exit_profit_percent']
                order.children_orders['exit_profit'].update(price=price, exit_profit_percent=None)
            
            if order.children_orders['exit_loss']['exit_loss_percent'] is not None:
                price = order.price * order.children_orders['exit_loss']['exit_loss_percent']
                order.children_orders['exit_loss'].update(price=price, exit_loss_percent=None)

            # Update the children order.size
            order.children_orders['exit_profit'].update(size=order.size)
            order.children_orders['exit_loss'].update(size=order.size)

            # Send the child orders to the engine
            if order.direction == Order.Direction.Long:
                # Exit Profit Order
                order_above = self.sell(bar, stoplimit_price=None, parent_id=self.trade_id, 
                          exit_profit=None, exit_loss=None, trailing_percent=None, 
                          expiry_date=None,
                          **order.children_orders['exit_profit'])
                # Exit Loss Order
                order_below = self.sell(bar, stoplimit_price=None, parent_id=self.trade_id, 
                          exit_profit=None, exit_loss=None, trailing_percent=None, 
                          expiry_date=None,
                          **order.children_orders['exit_loss'])
            
            else:
                # Exit Profit Order
                order_below = self.buy(bar, stoplimit_price=None, parent_id=self.trade_id, 
                          exit_profit=None, exit_loss=None, trailing_percent=None, 
                          expiry_date=None,
                          **order.children_orders['exit_profit'])
                # Exit Loss Order
                order_above = self.buy(bar, stoplimit_price=None, parent_id=self.trade_id, 
                          exit_profit=None, exit_loss=None, trailing_percent=None, 
                          expiry_date=None,
                          **order.children_orders['exit_loss'])

            # Execute the parent order
            new_trade = Trade(self.trade_id, order, timestamp=bar.timestamp)
        
        else:
            # Execute the Trade
            new_trade = Trade(self.trade_id, order, timestamp=bar.timestamp)
        
        # Add Trade to self.trades
        self.trades[bar.ticker][self.trade_id] = new_trade

        # Update Ticker Units in Portfolio
        self.compute_portfolio_stats(bar.index, exclude_closed=False)

        logging.info(f'\n\n\n\n\n\n<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< TRADE {new_trade.id} OPENED >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        logging.info(f'Trade {new_trade.id} Executed. (Entry Price : {order.price})')
        logging.info(f'{self.portfolio.loc[bar.index]}')

        # If there are children orders:
        if order.children_orders:
            # Check if Children Orders would be filled in that bar
            if bar.open <= bar.close:
                # For Green Bars, check the order above if it is filled on the same bar
                self._process_order(order_above, bar)
            else:
                # For Red Bars, check the order below if it is filled on the same bar
                self._process_order(order_below, bar)


    def _close_trade(self, trade:Trade, bar:Bar, price:float):

        # Update the Trade
        self._update_trade(trade, bar, price)
        trade.close(bar, price)

        # Update Portfolio Balance
        self.portfolio.loc[bar.index, 'balance'] += trade.params.pnl

        # Mark the trade as closed
        trade.Status = Trade.Status.Closed

        # Remove the trade from self.trade dictionary (key = trade_id)
        self.trades[bar.ticker].pop(trade.id)

        # Add trade trade history
        self.history['trades'].append(trade)

        # Update Portfolio Value for that ticker with Exit Price, Then Reduce Them
        self.portfolio.loc[bar.index, f'{bar.ticker} closed_pnl'] += trade.params.pnl 
        self.compute_portfolio_stats(bar.index, exclude_closed=True)
        
        # For debugging purposes
        self.trade_count += 1
        logging.info(f'TRADE CLOSED : (Entry : {trade.entry_price}, Exit : ({price}))')
        logging.info(f'{self.portfolio.loc[bar.index]}')
        logging.info(f'\n\n<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< TRADE {trade.id} CLOSED >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')

    
    def count_active_trades(self, ticker:str=None):  
        # If ticker is passed       
        if ticker is not None:
            if ticker not in self.tickers:
                return None
        
            return len(self.trades[ticker])
        
        count = {}
        for ticker in self.tickers:
            count[ticker] = len(self.trades[ticker])

        return sum(count.values())
        

    # PLOTTING METHODS
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



    # FOR REPORTS
    def compute_reports(self):
        '''
        Generate Reports for the Backtest
        '''

        # Compute the Trades History
        report_trades, _ = self.compute_trades_report()


        report_trades = self._process_trade_history(report_trades)

        # Compute the metrics
        report_metrics = self.compute_metrics_report(report_trades)

        return report_trades, report_metrics
    

    def _process_trade_history(self, history):
        history['cumm_profit'] = history['profit'].cumsum()
        history['cumm_profit_perc'] = history['profit_percent'].cumsum()
        history['entry_timestamp'] = pd.to_datetime(history['entry_timestamp'])
        history['exit_timestamp'] = pd.to_datetime(history['exit_timestamp'])
        history['duration'] = history['exit_timestamp'] - history['entry_timestamp']

        history.index = history.entry_timestamp
        history.index.name = 'date'
        history = history.sort_values(by='date')

        return history


    def report_max_drawdown(self, data : pd.DataFrame):
        max_dd = data['cumm_profit'].cummax() - data['cumm_profit']
        max_dd_percent = data['cumm_profit_perc'].cummax() - data['cumm_profit_perc']
        return max_dd.max(), max_dd_percent.max()


    def report_max_runup(self, data : pd.DataFrame):
        max_runup = data['cumm_profit'] - data['cumm_profit'].cummin()
        max_runup_percent = data['cumm_profit_perc'] - data['cumm_profit_perc'].cummin()
        return max_runup.max(), max_runup_percent.max()


    def compute_trades_report(self):
        """
        Compute and return a trades report including all trades and trades per ticker.

        Returns:
        - all_trades (pd.DataFrame): DataFrame containing properties of all trades.
        - trades_per_ticker (dict): Dictionary with tickers as keys and trade DataFrames as values.
        """
        # Initialize an empty list to store trade properties for all trades
        trades_list = []

        # Extract the 'trades' history from the object
        history = self.history['trades']

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
        all_trades = pd.DataFrame(trades_list)

        # Convert data types of DataFrame columns for consistency
        all_trades = all_trades.astype({
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

        # Create a list of traded tickers
        traded_tickers = all_trades['ticker'].unique().tolist()

        # Create a dictionary of DataFrames, where each key is a ticker and the value is a DataFrame of trades for that ticker
        trades_per_ticker = {ticker: all_trades[all_trades['ticker'] == ticker] for ticker in traded_tickers}

        # Return the overall trades DataFrame and the dictionary of trades per ticker
        return all_trades, trades_per_ticker


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
        max_runup, max_runup_percent = self.report_max_runup(trades) # Max Run-up
        max_drawdown, max_drawdown_percent = self.report_max_drawdown(trades) # Max Drawdown

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


    def compute_full_history(self, portfolio):
        '''
        Compiles the portfolio/trade history for each backtest day
        '''
        return


    def compute_params_report(self):
        '''
        This would compile all the parameters from the strategy applied. 
        It should be able to communicate with the strategy object used in the backtest
        '''

        return



if __name__ == '__main__':
    import yfinance as yf
    start_date = '2020-01-02'
    end_date = '2023-12-31'

    clear_terminal()
    with open('logs.log', 'w'):
        pass

    tickers = ['AAPL', 'GOOG', 'TSLA']
    ticker_path = [f'data/prices/{ticker}.csv' for ticker in tickers]

    dfs = []

    for ticker in tickers:
        file_name = f'data/prices/{ticker}.csv'

        if os.path.exists(file_name):
            df = pd.read_csv(file_name, index_col='Date', parse_dates=True)
        else:
            df = yf.download(ticker, start=start_date, end=end_date)
            df.to_csv(file_name)
            
        dfs.append(df)

    dataframes = dict(zip(tickers, dfs))

    alpha = Engine(tickers, dataframes, '1d', start_date, end_date)
    alpha.run_backtest()