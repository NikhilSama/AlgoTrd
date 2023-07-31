import requests

def getURL(ticker,type='TIME_SERIES_DAILY_ADJUSTED',csv=False):
    apikey = 'SMNNO8O470J0R6VW'
    # replace the "demo" apikey below with your own key from https://www.alphavantage.co/support/#api-key
    url = f'https://www.alphavantage.co/query?function={type}&symbol={ticker}&ouputsize=full&apikey={apikey}'
    url += '&datatype=csv' if csv else ''
    return url

csv = True

url = getURL('HDFCBANK.BSE',csv=csv)
r = requests.get(url)
data = r.json() if not csv else r.text
print(data)



