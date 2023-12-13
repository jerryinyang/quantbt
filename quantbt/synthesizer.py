import pandas as pd
import numpy as np
from utils import clear_terminal, debug # noqa


class GBM:
    def __init__(self, data:pd.DataFrame, add_jump:bool=True, jump_probability:float=0.005, jump_size:float=0.00001):
        """
        Generate synthetic OHLC data using Geometric Brownian Motion and Jump Diffusion model.

        Parameters:
        - data (pd.DataFrame): Pandas DataFrame with columns 'Open', 'High', 'Low', 'Close', and 'Volume'.
        - add_jump (bool): Flag to include an additional jump factor for extra random price spikes (default is True).
        - jump_probability (float): Intensity of jumps (probability of jump at each time step).
        - jump_size (float): Relative size of jumps.
        """
        
        # Create a copy of the dataframe, convert the columns into lowercase
        data = data.copy()
        data.columns = data.columns.str.lower()

        # Ensure the DataFrame has the required columns
        assert set(['open', 'high', 'low', 'close']).issubset(data.columns), "Input DataFrame must contain 'Open', 'High', 'Low', 'Close' columns."

        self.data = data
        self.length = len(data)

        self.add_jump = add_jump
        self.jump_probability = jump_probability
        self.jump_size = jump_size


    def synthesize(self):
        # Assign default data, if not assigned
        data = self.data['close']
        
        # Get Initial Price
        initial_price = data.iloc[0]
        
        # Set length
        length = self.length

        # Calculate returns from the input data
        returns = data.pct_change()
        returns = returns.fillna(0)

        # Calculate drift (mu) and volatility (sigma) from historical returns
        mu = returns.mean()
        sigma = returns.std()

        # Simulate future prices using Jump Diffusion model
        new_prices = [initial_price]

        # Fill data with new values 
        for _ in range(1, length):
            # Calculate the new price
            next_price = self.next_value(new_prices[-1], mu, sigma)

            # Add the new price
            new_prices.append(next_price)

        # Create a DataFrame for synthetic OHLC data
        new_data = pd.DataFrame(index=self.data.index,
                                columns=['open', 'high', 'low', 'close', 'volume'])
    
        # Set initial values
        new_data['close'] = new_prices
        new_data['open'] = new_data['close'].shift().fillna(new_data['close']) 
        new_data['high'] = new_data[['open', 'close']].max(axis=1)
        new_data['low'] = new_data[['open', 'close']].min(axis=1)

        if 'volume' in self.data.columns:
            new_data['volume'] = self.data['volume']

        return new_data
    

    def forecast(self, length, lookback):
        # Assign default data, if not assigned
        data = self.data['close']
        data = data.tail(lookback)
        
        # Get Initial Price
        initial_price = data.iloc[-1]

        # Calculate returns from the input data
        returns = data.pct_change()
        returns = returns.fillna(0)

        # Calculate drift (mu) and volatility (sigma) from historical returns
        mu = returns.mean()
        sigma = returns.std()

        # Simulate future prices using Jump Diffusion model
        new_prices = [initial_price]

        # Fill data with new values 
        for _ in range(1, length):
            # Calculate the new price
            next_price = self.next_value(new_prices[-1], mu, sigma)

            # Add the new price
            new_prices.append(next_price)

        # Create a DataFrame for synthetic OHLC data
        new_data = pd.DataFrame(index=data.index[-1] + pd.to_timedelta(np.arange(1, length + 1), unit='D').days,
                                columns=['open', 'high', 'low', 'close', 'volume'])
    
        # Set initial values
        new_data['close'] = new_prices
        new_data['open'] = new_data['close'].shift().fillna(new_data['close'])
        new_data['high'] = new_data[['open', 'close']].max(axis=1)
        new_data['low'] = new_data[['open', 'close']].min(axis=1)

        if 'volume' in self.data.columns:
            new_data['volume'] = self.data['volume']

        return new_data
    

    def next_value(self, value, mu, sigma, delta=0.1):
        delta_t = delta
        Z_t = np.random.normal(0, 1)
        
        # Add Jump Factor, for extra random price spikes
        jump = self.jump_diffusion()

        next_price = value * np.exp((mu - 0.5 * sigma**2) * delta_t + sigma * np.sqrt(delta_t) * Z_t + jump)

        return next_price
        

    def jump_diffusion(self):
        '''
        Generates a random jump factor.

        Parameters:
        - jump_probability: Intensity of jumps (probability of jump at each time step).
        - jump_size: Relative size of jumps.
        '''

        # Simulate jump
        jump_prob = np.random.uniform(0, 1)
        if (jump_prob < self.jump_probability) and (self.add_jump):

            jump = np.random.normal(0, self.jump_size)
        else:
            jump = 0

        return jump
    

class Noise:
    def __init__(self) -> None:
        pass








if __name__ == '__main__':

    clear_terminal()

    data = pd.read_csv('/Users/jerryinyang/Code/quantbt/data/prices/AAPL.csv')

    np.random.seed(42)
    gbm = GBM(data, add_jump=False)
    x = gbm.synthesize()

    x.to_csv('/Users/jerryinyang/Code/quantbt/quantbt/x.csv')