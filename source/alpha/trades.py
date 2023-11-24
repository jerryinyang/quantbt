from enum import Enum
from orders import Order
from utils import Bar


class Trade:
    def __init__(self, id, order: Order, timestamp, family_id: int = None) -> None:
        self.id = id
        self.ticker = order.ticker
        self.direction = order.direction
        self.size = order.size

        self.entry_price = order.price
        self.entry_timestamp = timestamp
        self.entry_order_id = order.id

        self.exit_price = order.price
        self.exit_timestamp = timestamp
        self.exit_order_id = order.id

        self.parent_id = order.parent_id
        self.family_id = family_id

        self.status = Trade.Status.Active
        self.params = self.Params()

    class Status(Enum):
        Active = "Active"  # Order object created
        Closed = "Closed"  # Order has been added to engine's lists of orders

    class Params:
        def __init__(self) -> None:
            self.pnl = 0
            self.pnl_perc = 0
            self.commission = 0
            self.max_runup = 0  # Highest PnL Value During Trade
            self.max_runup_perc = (
                0  # Highest PnL Value During Trade / (Entry Price x Quantity) * 100
            )
            self.max_drawdown = 0  # Lowest PnL Value During Trade
            self.max_drawdown_perc = (
                0  # Lowest PnL Value During Trade / (Entry Price x Quantity) * 100
            )

    def close(self, order: Order, bar: Bar):
        pass
