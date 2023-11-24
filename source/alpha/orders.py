from enum import Enum
from typing import Any
from utils import Bar
from dateutil.parser import parse
from datetime import datetime


class Order:
    # TODO: Create __new__, __repr__ methods
    def __repr__(self) -> str:
        return f"Order {self.id}"

    def __init__(
        self,
        id,
        bar: Bar,
        direction,
        price,
        order_type,
        size,
        stoplimit_price: float = None,
        parent_id: int = None,
        exit_profit: float = None,
        exit_loss: float = None,
        trailing_percent: float = None,
        family_role=None,
        family_id: int = None,
        expiry_date=None,
    ) -> None:
        # TODO : Assert the data types for each parameter
        # assert isinstance(bar, Bar), 'bar passed is not a Bar object'

        self.id = id
        self.ticker = bar.ticker
        self.direction = direction
        self.timestamp = bar.timestamp
        self.size = size

        self.price = price
        self.order_type = order_type

        self.stoplimit_price = stoplimit_price  # For Stop Limit Orders
        self.stoplimit_active = False  #
        self.parent_id = parent_id

        self.trailing_percent = (
            trailing_percent if (parent_id and trailing_percent) else None
        )
        self.status = self.Status.Created
        self.expiry_date = expiry_date or parse("9999-01-01").astimezone(
            self.timestamp.tzinfo
        )  # Sets the expiry date for the order

        # If exit_profit or exit_loss is defined; any child order
        self.children_orders = {}
        self.family_id = family_id
        self.family_role = family_role

        if exit_profit:
            self.children_orders["exit_profit"] = {
                # 'direction' : Order.Direction.Short if self.direction is Order.Direction.Long else Order.Direction.Long, # Opposite direction to the parent
                # 'timestamp' : self.timestamp,
                "price": exit_profit,
                "order_type": Order.ExecType.ExitLimit,
                "size": self.size,  # Default size should be the same size of the entry
                "family_role": Order.FamilyRole.ChildExit,
            }
            self.family_role = Order.FamilyRole.Parent

        if exit_loss:
            self.children_orders["exit_loss"] = {
                # 'direction' : Order.Direction.Short if self.direction is Order.Direction.Long else Order.Direction.Long, # Opposite direction to the parent
                # 'timestamp' : self.timestamp,
                "price": exit_loss,
                "order_type": Order.ExecType.ExitStop,
                "size": self.size,  # Default size should be the same size of the entry
                "family_role": Order.FamilyRole.ChildExit,
            }
            self.family_role = Order.FamilyRole.Parent

        # TODO: Add other child orders to family_orders (trailing_percent)

    def __getattr__(self, __name: str) -> Any:
        try:
            return self.__getattribute__(__name)
        except AttributeError as e:
            print(f"Attribute {__name} has not been set")
            raise e

    def __lt__(self, other):
        if isinstance(other, Order):
            return self.price < other.price
        return NotImplemented

    def __le__(self, other):
        if isinstance(other, Order):
            return self.price <= other.price
        return NotImplemented

    def __gt__(self, other):
        if isinstance(other, Order):
            return self.price > other.price
        return NotImplemented

    def __ge__(self, other):
        if isinstance(other, Order):
            return self.price >= other.price
        return NotImplemented

    def __eq__(self, other):
        if isinstance(other, Order):
            return self.id == other.id
        return False

    def __hash__(self):
        # Creating a tuple of relevant attributes for hashing
        hash_tuple = (
            self.id,
            self.ticker,
            self.direction,
            self.timestamp,
            self.size,
            self.price,
            self.order_type,
            self.stoplimit_price,
            self.stoplimit_active,
            self.parent_id,
            self.trailing_percent,
            self.status,
            self.expiry_date,
            self.family_id,
            self.family_role,
        )
        return hash(hash_tuple)

    def filled(self, bar: Bar):
        """
        Determines if an order is filled based on its execution type and the provided bar.

        Parameters:
        - self: The instance of the Order class.
        - bar (Bar): The bar to check for order fill.

        Returns:
        Tuple[bool, Union[float, None]]:
            - If the order is filled:
                - True: Order is filled.
                - If the order type is Market, the second element is the closing price of the bar.
                - If the order type is Limit, Stop, or StopLimit (after activation), the second element is the order price.
            - If the order is not filled:
                - False: Order is not filled.
                - None: The second element is None.

        Note:
        - For StopLimit orders, the method checks for the activation of the StopLimit and subsequent fills at the limit price.
        - If the StopLimit is not active, it checks for fills at the StopLimit price and activates it if fills occur.

        Example Usage:
        ```python
        order = Order()
        bar = Bar(close=100.0, high=105.0, low=95.0, open=98.0, volume=1000)

        is_filled, fill_price = order.filled(bar)
        ```
        """
        if self.order_type == Order.ExecType.Market:
            return True, bar.close

        elif self.order_type in [
            Order.ExecType.Limit,
            Order.ExecType.Stop,
            Order.ExecType.ExitLimit,
            Order.ExecType.ExitStop,
            Order.ExecType.Trailing,
        ]:
            return bar.fills_price(self.price), self.price

        elif self.order_type in Order.ExecType.StopLimit:
            if not self.stoplimit_active:
                if bar.fills_price(self.stoplimit_price):
                    self.stoplimit_active = True
                return False, None

            elif bar.fills_price(self.price):
                return True, self.price

        return False, None

    def expired(self, date: str | datetime):
        return date >= self.expiry_date

    def accept(self):
        self.status = Order.Status.Accepted
        return self

    def cancel(self):
        self.status = Order.Status.Canceled
        return self

    def fill(self):
        self.status = Order.Status.Filled
        return self

    def reject(self):
        self.status = Order.Status.Rejected
        return self

    class ExecType(Enum):
        Market = "Market"
        Limit = "Limit"
        Stop = "Stop"
        StopLimit = "StopLimit"
        ExitLimit = "ExitLimit"
        ExitStop = "ExitStop"
        Trailing = "Trailing"  # TODO: Implement this

    class Direction(Enum):
        Long = 1
        Short = -1

    class Status(Enum):
        Created = "Created"  # Order object created
        Accepted = "Accepted"  # Order has been added to engine's lists of orders
        Filled = "Filled"  # Order object Has Been Filled
        Canceled = "Canceled"  # Order has been cancelled
        Cancelled = Canceled  # Creating an Alias
        Rejected = "Rejected"  # Order failed to be added to engine's lists of orders

    class FamilyRole(Enum):
        Parent = "Parent"
        ChildExit = "ChildExit"
        ChildReduce = "ChildReduce"
