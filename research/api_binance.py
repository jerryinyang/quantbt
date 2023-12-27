import datetime
from dotenv import load_dotenv, dotenv_values
import requests
import json
import time
import pandas as pd

# Load Environment Variables
load_dotenv()
config = dotenv_values(".env")

# Import Environment Variables
binance_key = config.get("BINANCE_API_KEY")
binance_secret = config.get("BINANCE_SECRET_KEY")

def fetch_crypto_data(symbol, resolution, start_time = datetime.datetime.now() - datetime.timedelta(days=30), end_time=datetime.datetime.now()):

    # Convert the times to Unix timestamps in milliseconds
    start_timestamp = int(start_time.timestamp() * 1000)
    end_timestamp = int(end_time.timestamp() * 1000)

    # Define the Binance API endpoint for K-line data
    endpoint = 'https://api.binance.com/api/v3/klines'

    # Define the parameters for the API request
    limit = 1000
    params = {'symbol': symbol, 'interval': resolution, 'startTime': start_timestamp, 'endTime': end_timestamp, 'limit': limit}

    # Send the API request and store the response data in a list
    data = []
    while True:
        response = requests.get(endpoint, params=params)
        klines = json.loads(response.text)
        data += klines
        if len(klines) < limit:
            break
        params['startTime'] = int(klines[-1][0]) + 1
        time.sleep(0.1)

    # Create a pandas dataframe with the OHLC data and timestamps
    ohlc_data = [[float(kline[1]), float(kline[2]), float(kline[3]), float(kline[4]), float(kline[5])] for kline in data]
    df = pd.DataFrame(ohlc_data, columns=['open', 'high', 'low', 'close', 'volume'])
    timestamps = [datetime.datetime.fromtimestamp(int(kline[0]) / 1000) for kline in data]
    df['date'] = timestamps
    df.set_index('date', inplace=True)

    df.to_parquet(f'/Users/jerryinyang/Code/quantbt/data/prices/{symbol}_1D.parquet')

    return df

if __name__ == '__main__':
    # Download Bulk Data
    symbols = ['BTCUSDT']#, 'ETHUSDT', 'GMTUSDT', 'CELOUSDT', 'DOGEUSDT', 'SOLUSDT']

    for symbol in symbols:

        # Define the start and end times for the data
        end_time = datetime.datetime.now()
        start_time = end_time - datetime.timedelta(days=365 * 6)

        _ = fetch_crypto_data(symbol, '1d', start_time, end_time)