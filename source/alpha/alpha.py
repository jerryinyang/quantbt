import pandas as pd
import yfinance as yf
from dateutil.parser import parse
import pytz


class Alpha:
    TZ = pytz.utc
    DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

    def __init__(self, tickers:list[str], dataframes:dict[str,pd.DataFrame], resolution:str, start_date, end_date) -> None:
        self.resolution = resolution
        self.tickers = tickers
        self.dataframes = self.__set_dataframe_timezone(dataframes)
        self.start_date = parse(start_date, tzinfos={'UTC': self.TZ})
        self.end_date = parse(end_date, tzinfos={'UTC': self.TZ})

        self.capital = 10000

    
    def init_portfolio(self, date_range:pd.DatetimeIndex):
        # Initialize Portfolio Dataframe: this would contain all the portfolio attributes

        portfolio =  pd.DataFrame({'timestamp': date_range}) # Initialize the full date range for the backtest
        portfolio.loc[0, 'capital'] = 10000 # Initialize the backtest capital

        return portfolio


    def __set_dataframe_timezone(self, dataframes:dict[str,pd.DataFrame]):

        for ticker, df in dataframes.items():

            df.index = pd.to_datetime(df.index.strftime(self.DATE_FORMAT)).tz_localize(self.TZ)
            df.rename(columns={old_column : old_column.lower() for old_column in list(df.columns)}, inplace=True)
            df = df[['open', 'high', 'low', 'close', 'volume']]
            dataframes[ticker] = df

        return dataframes


    def filter_asset_universe(self, date_range:pd.DatetimeIndex) -> None:
        '''
        Compute Available/ / Eligible Assets to Trade (apply Universe Filtering Rules).\n
        Asset Eligibility can include asset tradable days (crypto -> 24/7, forex -> 24/5, futures -> Market Hours), holidays, etc.
        In summary, this will check which data is available for trading in what days, and mark them as eligible.
        '''
        def check_changes(row, columns):
            changes = all(row[columns] != row[columns].shift(1))
            return changes
        
        for ticker in self.tickers:
            df = pd.DataFrame(index=date_range)
    
            self.dataframes[ticker] = df.join(self.dataframes[ticker], how='left').ffill() #.bfill()
            self.dataframes[ticker]['return'] = self.dataframes[ticker]['close'].pct_change()
            self.dataframes[ticker]['eligibility'] = check_changes(self.dataframes[ticker], ['open', 'high', 'low', 'close', 'volume'])
            self.dataframes[ticker]['eligibility'] = self.dataframes[ticker]['eligibility'].astype(int)
            
        return          

    
    def run_backtest(self):
        print('Initiating Backtest')

        # Set Backtest Range
        backtest_range = pd.date_range(start=self.start_date, end=self.end_date, freq=self.resolution, tz=self.TZ) # Full Backtest Date Range
        portfolio = self.init_portfolio(backtest_range)

        # Iterate through each bar/index (timestamp) in the backtest range
        count = 0
        for bar_index in portfolio.index:
            date = portfolio.loc[bar_index, 'timestamp']

            if bar_index > 0:
                # Manage Pending Orders
                # Update Open Orders
                # Update Portfolio Attributes (capital)

                # Compute the PnL
                previous_date = portfolio.loc[bar_index - 1, 'timestamp']
                pnl, capital_return = self.compute_pnl_stats(portfolio, bar_index,  date, previous_date)

            # Filter Asset Universe
            self.filter_asset_universe(backtest_range)
            
            eligible_assets = [ticker for ticker in self.tickers if self.dataframes[ticker].loc[date, 'eligibility']]
            non_eligible_assets = list(set(self.tickers) - set(eligible_assets))

            # Decision-making / Signal-generating Algorithm
            alpha_long, alpha_short = self.signal_generator(eligible_assets)

            for ticker in non_eligible_assets:
                # Units of asset in holding (Set to zero)
                portfolio.loc[bar_index, f"{ticker} weight"] = 0
                portfolio.loc[bar_index, f"{ticker} units"] = 0
            
            total_nominal_value = 0
            for ticker in eligible_assets:
                direction = 1 if ticker in alpha_long else -1 if ticker in alpha_short else 0

                risk_dollars = portfolio.loc[bar_index, 'capital'] / (len(alpha_long) + len(alpha_short))
                position_size = direction * risk_dollars / self.dataframes[ticker].loc[date, 'close']
                portfolio.loc[bar_index, f"{ticker} units"] = position_size

                total_nominal_value += abs(position_size * self.dataframes[ticker].loc[date, 'close'])
            
            for ticker in eligible_assets:
                units = portfolio.loc[bar_index, f"{ticker} units"]
                ticker_nominal_value = units * self.dataframes[ticker].loc[date, 'close']
                instrument_weight = ticker_nominal_value / total_nominal_value 

                portfolio.loc[bar_index, f"{ticker} weight"] = instrument_weight

            portfolio.loc[bar_index, 'total nominal value'] = total_nominal_value
            portfolio.loc[bar_index, 'leverage'] = total_nominal_value / portfolio.loc[bar_index, 'capital']


            
            count += 1
            if count % 50 == 0:
                input(portfolio.loc[bar_index])


    def signal_generator(self, eligibles:list[str]):
        import numpy as np

        alpha_scores = { key : np.random.rand() for key in eligibles}

        alpha_scores = {key : value for key, value in sorted(alpha_scores.items(), key=lambda item : item[1])} # Sorts the dictionary
        list_scores = list(alpha_scores.keys())
        
        alpha_long = [asset for asset in list_scores if alpha_scores[asset] > .5]
        alpha_short = [asset for asset in list_scores if alpha_scores[asset] <= .5]

        return alpha_long, alpha_short
             
  
    def compute_pnl_stats(self, portfolio, bar_index,  date, prev_date):
        pnl = 0
        nominal_return = 0

        for ticker in self.tickers:
            data = self.dataframes[ticker]
            units_traded = portfolio.loc[bar_index - 1, f"{ticker} units"]
            if units_traded == 0:
                continue

            # TODO : Use pct_change already calculated in filter_universe
            delta = data.loc[date, 'close'] - data.loc[prev_date, 'close']
            ticker_pnl = delta * units_traded

            pnl += ticker_pnl
            nominal_return += portfolio.loc[bar_index - 1, f"{ticker} weight"] * data.loc[date, 'return']

        capital_return = nominal_return * portfolio.loc[bar_index - 1, 'leverage']

        portfolio.loc[bar_index, 'capital'] = portfolio.loc[bar_index - 1, 'capital'] + pnl
        portfolio.loc[bar_index, 'pnl'] = pnl
        portfolio.loc[bar_index, 'nominal_return'] = nominal_return
        portfolio.loc[bar_index, 'capital_return'] = capital_return
        
        return pnl, capital_return


tickers = 'AAPL TSLA GOOG'.split(' ')
ticker_path = [f'source/alpha/{ticker}.parquet' for ticker in tickers]

# Read Data
dataframes = dict(zip(tickers, [pd.read_parquet(path) for path in ticker_path]))

alpha = Alpha(tickers, dataframes, '1d', '2020-01-01', '2023-12-31')

alpha.run_backtest()