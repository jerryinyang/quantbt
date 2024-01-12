from enum import Enum

from orders import Order
from utils import Bar


class Trade:
    def __init__(self, id, order: Order, timestamp) -> None:
        """
        Initializes a Trade object.

        Parameters:
        - id: Unique identifier for the trade.
        - order: The Order object associated with this trade.
        - timestamp: Timestamp when the trade is initiated.
        """
        self.id = id
        self.ticker = order.ticker
        self.direction = order.direction
        self.size = order.size

        self.entry_price = order.price
        self.entry_timestamp = timestamp

        self.exit_price = order.price
        self.exit_timestamp = timestamp

        self.parent_id = order.parent_id

        self.status = Trade.Status.Active
        self.params = self.Params()
        self.alpha_name  = order.alpha_name

    class Status(Enum):
        Active = "Active"  # Order object created
        Closed = "Closed"  # Order has been added to engine's lists of orders

    class Params:
        def __init__(self) -> None:
            """
            Initializes parameters for tracking trade performance.
            """
            self.pnl = 0
            self.pnl_perc = 0
            self.commission = 0
            self.max_runup = 0  # Highest PnL Value During Trade
            self.max_runup_perc = 0  # Highest PnL Value During Trade / (Entry Price x Quantity) * 100
            self.max_drawdown = 0  # Lowest PnL Value During Trade
            self.max_drawdown_perc = 0  # Lowest PnL Value During Trade / (Entry Price x Quantity) * 100


    def close(self, bar: Bar, price: float) -> None:
        """
        Closes the trade with the given closing price and updates relevant information.

        Parameters:
        - bar: The Bar object representing the closing bar.
        - price: The closing price of the trade.
        """
        self.exit_price = price
        self.exit_timestamp = bar.timestamp
        self.status = Trade.Status.Closed

        return
    

    # PICKLE-COMPATIBILITY
    def __getstate__(self):
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        # Customize the object reconstruction
        self.__dict__.update(state)
