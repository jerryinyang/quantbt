from enum import Enum
from typing import Any
from utils import Bar
from dateutil.parser import parse
from datetime import datetime

class Order:
    # TODO: Create __new__, __repr__ methods
    
    def __init__(self, id, bar:Bar, direction, price, order_type, size,
                 stoplimit_price:float=None, parent_id:str=None, 
                 exit_profit:float=None, exit_loss:float=None,
                 trailing_percent:float=None, bracket_role=None, expiry_date=None) -> None:
        
        # TODO : Assert the data types for each parameter

        self.id = id
        self.ticker = bar.ticker
        self.direction = direction
        self.timestamp = bar.timestamp
        self.size = size
        
        self.price = price
        self.order_type = order_type
        
        self.stoplimit_price = stoplimit_price # For Stop Limit Orders 
        self.stoplimit_active = False # 
        self.parent_id = parent_id

        self.trailing_percent = trailing_percent if (parent_id and trailing_percent) else None
        self.status = self.Status.Created
        self.expiry_date = expiry_date or parse('9999-01-01') # Sets the expiry date for the order


        # If exit_profit and exit_loss are defined
        self.bracket_orders = None
        self.bracket_role = bracket_role
        if (exit_profit and exit_loss): # if Both are specified
            self.bracket_orders = {
                'exit_profit' : {
                    'symbol' : self.ticker,
                    'direction' : Order.Direction.Short if self.direction is Order.Direction.Long else Order.Direction.Long, # Opposite direction to the parent
                    'timestamp' : self.timestamp,
                    'price' : exit_profit,
                    'order_type' : Order.ExecType.Limit,
                    'size' : self.size, # Default size should be the same size of the entry
                    'parent_id' : self.id,
                    'bracket_role' : Order.BrackerRole.ChildProfit
                },
                'exit_loss' : {
                    'symbol' : self.ticker,
                    'direction' : self.direction, # Opposite direction to the parent
                    'timestamp' : self.timestamp,
                    'price' : exit_loss,
                    'order_type' : Order.ExecType.Stop,
                    'size' : self.size, # Default size should be the same size of the entry
                    'parent_id' : self.id,
                    'bracket_role' : Order.BrackerRole.ChildLoss
                }
            }

            self.bracket_role = Order.BrackerRole.Parent


    def __getattr__(self, __name: str) -> Any:
        try:
            return self.__getattribute__(__name)
        except AttributeError as e:
            print (f"Attribute {__name} has not been set")
            raise e
    

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

        elif self.order_type in [Order.ExecType.Limit, Order.ExecType.Stop]:
            return bar.fills_price(self.price), self.price

        elif self.order_type == Order.ExecType.StopLimit:
            if not self.stoplimit_active:
                if bar.fills_price(self.stoplimit_price):
                    self.stoplimit_active = True
                return False, None

            elif bar.fills_price(self.price):
                return True, self.price

        return False, None


    def expired(self, date:str|datetime):
        return date >= self.expiry_date


    def accept(self):
        self.status = Order.Status.Accepted
        return self


    def cancel(self):
        self.status = Order.Status.Canceled
        return self
    

    def complete(self):
        self.status = Order.Status.Completed
        return self
     

    def reject(self):
        self.status = Order.Status.Rejected
        return self

    class ExecType(Enum):
        Market = 'Market'
        Limit = 'Limit'
        Stop = 'Stop'
        StopLimit = 'StopLimit'
        Exit = 'Exit'

    class Direction(Enum):
        Long = 'Long'
        Short = 'Short'

    class Status(Enum):
        Created = 'Created'     # Order object created
        Accepted = 'Accepted'   # Order has been added to engine's lists of orders
        Executed = 'Executed'   # Order object Has Been Filled  
        Canceled = 'Canceled'   # Order has been cancelled
        Cancelled = Canceled    # Creating an Alias
        Rejected = 'Rejected'   # Order failed to be added to engine's lists of orders

    class BrackerRole(Enum):
        Parent = 'Parent'
        ChildProfit = 'Child_profit'
        ChildLoss = 'Child_loss'


xx = Order(1, 'EURUSD', Order.Direction.Long, '0', 2345, Order.ExecType.Market, 10)
