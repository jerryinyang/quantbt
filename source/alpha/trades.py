from enum import Enum
from orders import Order
from utils import Bar

class Trade(object):

    def __init__(self, id, order:Order, timestamp, exit_price:float=None) -> None:
    
        self.id = id
        self.order_id = order.order_id
        self.ticker = order.ticker
        self.direction = order.direction
        self.timestamp = timestamp
        self.size = order.size
        
        self.entry_price = order.price
        self.exit_price = exit_price
        
        self.parent_id = order.parent_id
        self.bracket_orders = order.bracket_orders # TODO : Use these to create order objects

        self.status = Trade.Status.Active

        self.params = self.Params()


    class Status(Enum):
        Active = 'Active'     # Order object created
        Closed = 'Closed'   # Order has been added to engine's lists of orders
        
    class Params:
        def __init__(self, entry_bar:Bar, entry_order:Order) -> None:
            self.pnl = None
            self.pnl_perc = None
            self.commission = None
            self.entry_bar_index = entry_bar.index
            self.entry_order_id = entry_order.id
            self.exit_bar_index = None
            self.exit_order_id = None
            self.exit_time = None
            self.max_runup = None # Highest PnL Value During Trade
            self.max_runup_perc = None # Highest PnL Value During Trade / (Entry Price x Quantity) * 100
            self.max_drawdown = None # Lowest PnL Value During Trade
            self.max_drawdown_perc = None # Lowest PnL Value During Trade / (Entry Price x Quantity) * 100

    def update(self, bar):
        pass
