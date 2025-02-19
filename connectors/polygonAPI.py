import pandas as pd
from datetime import datetime, timedelta
import os
import sys
current_dir = os.path.dirname(__file__)
root_dir = os.path.abspath(os.path.join(current_dir, '../'))
sys.path.append(root_dir)

from cache import cache_df,loadTickerCache,getCachedTikerData

from polygon import RESTClient

def polygonGet(t,s,e,i):
    # print(f"exporting {s} to {e} for {t} interval:{i}")
    df = getCachedTikerData(t,s, e,i) 
    if not df.empty:
        return df

    key = "gmsq1BH4TorTJXfKQ0FiV87bhoG3suCz"
    
    # RESTClient can be used as a context manager to facilitate closing the underlying http session
    client = RESTClient(key)
    res = list(client.list_aggs(ticker=t, multiplier=1, timespan=i, from_=s, to=e, limit=50000))
    df = pd.DataFrame([vars(r) for r in res])  # Convert the list to a pandas DataFrame
    # Rename columns
    df.rename(columns={'open': 'Open', 'high': 'High', 'close': 'Adj Close', 'low': 'Low', 'volume': 'Volume'}, inplace=True)
    
    # Convert timestamp to datetime and set as index
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    
    # Drop unnecessary columns
    df.drop(columns=['transactions', 'vwap', 'otc'], inplace=True)
    loadTickerCache(df,t,s, e,i)

    return df

if __name__ == '__main__':
    polygonGet("SPY","2022-01-01","2022-06-01","minute")