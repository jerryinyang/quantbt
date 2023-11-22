class Trade(object):

    def __init__(self, id, symbol, direction, timestamp, size, entry_price, exit_price:float=None, parent_id:str=None, bracket_name:str=None, bracket_role:str=None) -> None:
        self.id = id
        self.symbol = symbol
        self.direction = direction
        self.timestamp = timestamp
        self.size = size
        
        self.entry_price = entry_price
        self.exit_price = exit_price
        
        self.parent_id = None
        self.bracket = self.BracketOrder(bracket_name, bracket_role) if (bracket_name and bracket_role) else None

        # params = (('pnl', None)
        # ('pnl_perc', None)

        # ('commission', None)

        # ('entry_bar_index', None)
        # ('exit_bar_index', None)

        # ('entry_order_id', None)
        # ('exit_order_id', None)

        # ('exit_time', None)
        # ('max_runup', None) # Highest PnL Value During Trade
        # ('max_runup_perc', None) # Highest PnL Value During Trade / (Entry Price x Quantity) * 100
        # ('max_drawdown', None) # Lowest PnL Value During Trade
        # ('max_drawdown_perc', None) # Lowest PnL Value During Trade / (Entry Price x Quantity) * 100
        # )

