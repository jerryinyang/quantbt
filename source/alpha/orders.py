from enum import Enum

class Order:

    def __init__(self, id, symbol, direction, timestamp, price, order_type, size, trigger_price:float=None, parent_id:str=None, bracket_name:str=None, bracket_role:str=None) -> None:
        self.id = id
        self.symbol = symbol
        self.direction = direction
        self.timestamp = timestamp
        self.size = size
        
        self.price = price
        self.order_type = order_type
        
        self.trigger_price = trigger_price # For Stop Limit Orders 
        self.parent_id = None
        self.bracket = self.BracketOrder(bracket_name, bracket_role) if (bracket_name and bracket_role) else None

        self.status = None

    class BracketOrder:
        def __init__(self, name, role):
            self.name = name # Group name or ID
            self.role = role # Parent order, ChildLimit, ChildStop order 

    class ExecType(Enum):
        Market = 'Market'
        Limit = 'Limit'
        Stop = 'Stop'
        StopLimit = 'StopLimit'

    class Direction(Enum):
        Long = 'Long'
        Short = 'Short'

    class Status(Enum):
        Created = 'Created'
        Executed = 'Executed'
        Canceled = 'Canceled'
        Cancelled = Canceled  # Creating an Alias
        Rejected = 'Rejected'
