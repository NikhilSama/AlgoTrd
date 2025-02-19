import requests
import pandas as pd
import time

api_key = 'SMNNO8O470J0R6VW'

def avGet(t,m,i):
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={t}&interval={i}&month={m}&outputsize=full&apikey={api_key}'
    # Make the API request
    try:
        r = requests.get(url)
        data = r.json()
    except Exception as e:
        print(f"Error occurred while making API request: {e}")
        return None
    print (data['Meta Data'])
    # Check if the 'Time Series (5min)' key exists in the response
    if 'Time Series (5min)' in data:
        # Extract the intraday data
        intraday_data = data['Time Series (5min)']
        
        # Convert the data to a pandas DataFrame
        df = pd.DataFrame.from_dict(intraday_data, orient='index')
        df.index = pd.to_datetime(df.index)  # Convert the index to datetime
        # Filter df to 9:30 am to 4:30 pm only
        df = df.between_time('9:30', '16:30')
                # Sort df in ascending index
        df.sort_index(inplace=True)
                
        df.columns = ['Open', 'High', 'Low', 'Adj Close', 'Volume']  # Rename columns
        df.dropna(inplace=True)

        # Convert column data types as needed
        df = df.astype({'Open': float, 'High': float, 'Low': float, 'Adj Close': float, 'Volume': int})
        print (df)
        return df

