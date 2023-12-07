import logging

from bisect import bisect_left, bisect_right
from typing import Dict, List

from dataloader import DataLoader
from portfolio import Portfolio
from observers import Observer

from orders import Order
from trades import Trade
from utils import Bar, ObservableList as olist, ObservableDict as odict  # noqa: F401
from utils import debug  # noqa: F401


logging.basicConfig(filename='logs.log', level=logging.INFO)

exectypes = Order.ExecType

class Engine:
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

    def __init__(self, dataloader:DataLoader) -> None:
        self.tickers = dataloader.tickers
        self.dataframes = dataloader.dataframes

        self.portfolio = Portfolio(dataloader, self.CAPITAL)

        # Store Pending Orders
        self.orders : olist[Order] = olist(callback=self.onOrder)

        self.trades : Dict[str, odict[int, Order]] = {key : odict(callback=self.onTrade) for key in self.tickers} # Syntax : Dict[ticker, Dict [ trade_id : Trade()]]
        self.history : List[Trade] = []

        # Keep Observers
        self.observers : List[Observer] = []


    # METHODS FOR COMPUTING PORTFOLIO
    def filter_eligible_assets(self, date) -> None:
        '''
        Filter Eligible Assets to Trade (apply Universe Filtering Rules) on a specific day.\n
        Asset Eligibility can include asset tradable days (crypto -> 24/7, forex -> 24/5, futures -> Market Hours), holidays, etc.
        In summary, this will check which data is available for trading in what days, and mark them as eligible.
        '''   

        # Return Eligible Assets
        eligible_assets = [ticker for ticker in self.tickers if self.dataframes[ticker].loc[date, 'market_open']]
        non_eligible_assets = list(set(self.tickers) - set(eligible_assets))

        return eligible_assets, non_eligible_assets


    def compute_orders(self, bars : dict[Bar]):
        # TODO : Simulate realtime order execution (sorted)

        # Process Orders
        for order in self.orders:
            bar = bars[order.ticker]

            # Don't Update Orders when market is closed
            if not self.dataframes[order.ticker].loc[bar.timestamp, 'market_open']:
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


    def compute_portfolio_stats(self, bar_index, exclude_closed:bool=False):
        # If this is called on a new day
        # Store Previous Day's Values
        last_record_date = self.portfolio.get_record(0).date
        previous_date = self.portfolio.dataframe.loc[bar_index, 'timestamp']
        
        if (not last_record_date) or (last_record_date and not (previous_date == last_record_date)):
            self.portfolio.add_record(
                bar_index - 1, 
                previous_date,
                self.portfolio.dataframe.loc[bar_index - 1, 'balance'],
                self.portfolio.dataframe.loc[bar_index - 1, 'equity'],
                self.portfolio.dataframe.loc[bar_index - 1, 'open_pnl']
            )

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

            self.portfolio.dataframe.loc[bar_index, f'{ticker} units'] = units
            self.portfolio.dataframe.loc[bar_index, f'{ticker} open_pnl'] = pnl

            if not exclude_closed:
                self.portfolio.dataframe.loc[bar_index, f'{ticker} closed_pnl'] = self.portfolio.dataframe.loc[bar_index - 1, f'{ticker} closed_pnl']

            total_open_pnl += pnl
            total_closed_pnl += self.portfolio.dataframe.loc[bar_index, f'{ticker} closed_pnl']

        # Update General Portfolio Stats
        self.portfolio.dataframe.loc[bar_index, 'balance'] = self.CAPITAL + total_closed_pnl
        self.portfolio.dataframe.loc[bar_index, 'equity'] = self.portfolio.dataframe.loc[bar_index, 'balance'] + total_open_pnl
        self.portfolio.dataframe.loc[bar_index, 'open_pnl'] = total_open_pnl # Total Unrealized PnL


    def add_observer(self, observer:Observer):
        self.observers.append(observer)




    # METHODS FOR ORDER PROCESSING
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
        if not self.dataframes[order.ticker].loc[order.timestamp, 'market_open']:
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
            # If the parent order is active, Handle Child Orders (Take loss, Stop Loss, Trailing)
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
                if (self.portfolio.dataframe.loc[bar.index, 'balance'] < (order.size * fill_price)):

                    # Resize the order, if it is a Market Order
                    if order.exectype == exectypes.Market:
                        order.size = self._recalculate_market_order_size(order, bar, fill_price)
                        order.price = fill_price

                    # For Pending Orders, cancel it
                    else: 
                        # Not Enough Cash to take the Trade
                        # Add the order to rejected orders
                        return self._cancel_order(order, f'Insufficient margin/balance for this position. {order.size * fill_price})')
                
                if (self.portfolio.dataframe.loc[bar.index, 'balance'] >= (order.size * fill_price)):
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

            # logging.info(f'Order {order.id} Expired.')


    def _cancel_order(self, order:Order | List[Order], message:str=None):   
        # If a list is passed, recursively cancel each order
        if isinstance(order, list):
            for _order in order:
                self._cancel_order(_order)
        
        # Base Condition (order is Order instant)
        else:
            if order in self.orders:
                # Remove the order from self.orders
                self.orders.remove(order)

                # Cancel the order
                order.cancel()

                message = f'\nReason : {message}' if message else ''
                # logging.info(f'Order {order.id} Cancelled.' + message)

            # else:
                # raise ValueError(f"Order {order.id} is not found in the orders list.")


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
            
            message = f'\nReason : {message}' if message else ''
            # logging.info(f'Order {order.id} Rejected.' + message)

    
    def _recalculate_market_order_size(self, order:Order, bar:Bar, fill_price:float):
        '''
        For market orders, the order size should be recalculated based on the fill price, to account for gaps in the market.
        '''

        # Calculate the risk in cash
        risk_amount = self.portfolio.dataframe.loc[bar.index, 'balance']

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


    def onOrder(self, order):
        '''
        Use this method to monitor changes to self.orders
        '''
        
        # Update all observers
        for observer in self.observers:
            observer.update(order)




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
        trade.params.pnl_perc = (trade.params.pnl / self.portfolio.dataframe.loc[bar.index, 'balance']) * 100

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
        # TODO : Update observers 

        # Update Ticker Units in Portfolio
        self.compute_portfolio_stats(bar.index, exclude_closed=False)

        logging.info(f'\n\n\n\n\n\n<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< TRADE {new_trade.id} OPENED >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        logging.info(f'Trade {new_trade.id} Executed. (Entry Price : {order.price})')
        logging.info(f'{self.portfolio.dataframe.loc[bar.index]}')

        # If there are children orders:
        if order.children_orders:

            if not (order_above and order_below):
                debug('What the fuck is happening')
            # Check if Children Orders would be filled in that bar
            if bar.open <= bar.close:
                # For Green Bars, check the order above if it is filled on the same bar
                self._process_order(order_above, bar)
            else:
                # For Red Bars, check the order below if it is filled on the same bar
                self._process_order(order_below, bar)
        
        return new_trade


    def _close_trade(self, trade:Trade, bar:Bar, price:float):

        # Update the Trade
        self._update_trade(trade, bar, price)
        trade.close(bar, price)

        # Update Portfolio Balance
        self.portfolio.dataframe.loc[bar.index, 'balance'] += trade.params.pnl

        # Mark the trade as closed
        trade.Status = Trade.Status.Closed

        # Remove the trade from self.trade dictionary (key = trade_id)
        self.trades[bar.ticker].pop(trade.id)

        # Add trade trade history
        self.history.append(trade)

        # Update Portfolio Value for that ticker with Exit Price, Then Reduce Them
        self.portfolio.dataframe.loc[bar.index, f'{bar.ticker} closed_pnl'] += trade.params.pnl 
        self.compute_portfolio_stats(bar.index, exclude_closed=True)
        
        # For debugging purposes
        self.trade_count += 1
        logging.info(f'TRADE CLOSED : (Entry : {trade.entry_price}, Exit : ({price}))')
        logging.info(f'{self.portfolio.dataframe.loc[bar.index]}')
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


    def onTrade(self, trade:Trade):
        '''
        Use the method to monitor changes to self.trades
        '''

        # Update all observers
        for observer in self.observers:
            observer.update(trade)
