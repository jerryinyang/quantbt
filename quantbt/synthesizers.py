import numpy as np
import pandas as pd
from utils import clear_terminal, debug  # noqa


class GBM:
    def __init__(self, add_jump:bool=True, jump_probability:float=0.005, jump_size:float=0.00001):
        """
        Generate synthetic OHLC data using Geometric Brownian Motion and Jump Diffusion model.

        Parameters:
        - add_jump (bool): Flag to include an additional jump factor for extra random price spikes (default is True).
        - jump_probability (float): Intensity of jumps (probability of jump at each time step).
        - jump_size (float): Relative size of jumps.
        """
        self.add_jump = add_jump
        self.jump_probability = jump_probability
        self.jump_size = jump_size


    def synthesize(self, data:pd.DataFrame):
        # Create a copy of the dataframe, convert the columns into lowercase
        data = data.copy()
        data.columns = data.columns.str.lower()

        # Ensure the DataFrame has the required columns
        assert set(['open', 'high', 'low', 'close']).issubset(data.columns), "Input DataFrame must contain 'open', 'high', 'low', 'close' columns."

        # Assign default data, if not assigned
        prices = data['close']
        
        # Get Initial Price
        initial_price = prices.iloc[0]
        
        # Set length
        length = len(prices)

        # Calculate returns from the input data
        returns = prices.pct_change()
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

        # Create a copy of the orginal dataframe
        new_data = data.copy()
    
        # Modify OHLC values
        new_data['close'] = new_prices
        new_data['open'] = new_data['close'].shift().fillna(new_data['close']) 
        new_data['high'] = new_data[['open', 'close']].max(axis=1)
        new_data['low'] = new_data[['open', 'close']].min(axis=1)

        return new_data[['open', 'high', 'low', 'close', 'volume']]
    

    def forecast(self, data:pd.DataFrame, length, lookback):
        # Create a copy of the dataframe, convert the columns into lowercase
        data = data.copy()
        data.columns = data.columns.str.lower()

        # Ensure the DataFrame has the required columns
        assert set(['open', 'high', 'low', 'close']).issubset(data.columns), "Input DataFrame must contain 'open', 'high', 'low', 'close' columns."

        # Assign default data, if not assigned
        prices = data['close']
        prices = prices.tail(lookback)
        
        # Get Initial Price
        initial_price = prices.iloc[-1]

        # Calculate returns from the input data
        returns = prices.pct_change()
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
        new_data['volume'] = data['volume'].sample(length).values

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
    def __init__(self, noise_factor: float = 0.2, mu: float = 0, sigma: float = 0.01, normalize: bool = True) -> None:
        """
        Apply Gaussian noise to a DataFrame of OHLCV data.

        Parameters:
        - noise_factor (float): Scaling factor for adjusting the intensity of the applied noise.
        - mu (float): Mean of the Gaussian distribution used for generating noise.
        - sigma (float): Standard deviation of the Gaussian distribution used for generating noise.
        - normalize (bool): If True, dynamically calculates mu and sigma based on column percentage changes.

        Returns:
        - pd.DataFrame: DataFrame with added noise.

        Raises:
        - AssertionError: If the DataFrame does not contain the necessary 'open', 'high', 'low', 'close', 'volume' columns.
        """
        
        self.noise_factor = noise_factor
        self.mu = mu
        self.sigma = sigma
        self.normalize = normalize


    def synthesize(self, data: pd.DataFrame) -> pd.DataFrame:
        # Ensure the DataFrame has the necessary columns
        assert set(['open', 'high', 'low', 'close', 'volume']).issubset(data.columns), \
            "DataFrame must contain 'open', 'high', 'low', 'close', 'volume' columns."

        # Get Parameters
        noise_factor = self.noise_factor
        mu = self.mu
        sigma = self.sigma
        normalize = self.normalize

        # Create a copy of the DataFrame to avoid modifying the original data
        new_data = data.copy()

        # Apply noise to each OHLCV column
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if normalize:
                mu = new_data[col].pct_change().fillna(0).mean() * noise_factor
                sigma = new_data[col].pct_change().fillna(0).std() * noise_factor

            noise = np.random.normal(mu, sigma, size=len(data))
            new_data[col] = new_data[col] * (1 + noise)

        return new_data


if __name__ == '__main__':

    clear_terminal()

    data = pd.read_csv('/Users/jerryinyang/Code/quantbt/data/prices/AAPL.csv')

    np.random.seed(42)
    gbm = GBM(add_jump=False)
    x = gbm.forecast(data, 23, 14)

    x.to_csv('/Users/jerryinyang/Code/quantbt/quantbt/x.csv')