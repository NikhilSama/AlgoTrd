# %%
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import date,timedelta
import talib as ta
import time
import os
from IPython.display import clear_output
from plotly.subplots import make_subplots
from ta.volume import VolumeWeightedAveragePrice
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima_model import ARIMA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import mplfinance as mpf
import matplotlib.pyplot as plt
import datetime as dt

# %%
# Functions used in the code

def vwap(dataframe, label='vwap', window=10, fillna=True):
        dataframe[label] = VolumeWeightedAveragePrice(high=dataframe['High'], low=dataframe['Low'], close=dataframe["Adj Close"], volume=dataframe['Volume'], window=window, fillna=fillna).volume_weighted_average_price()
        return dataframe
def na_rows(df):
  '''
  This functions returns rows 
  where NaN values are present
  '''
  return df[df.isna().any(axis=1)]
def check_adfuller(df):
    # Dickey-Fuller test
    print("NULL HYPOTHESIS: time series is NOT stationary")
    print("ALTERNATE HYPOTHESIS: time series is stationary")
    result = adfuller(df, autolag = 'AIC')
    print('Test statistic:', result[0])
    print('p-value:', result[1])
    print('Critical Values:' ,result[4])
    if (result[0] > result[4]["5%"]):
        print("Test statistic is greater than 5% critical value, ACCEPT NULL HYPOTHESIS")
    else:
        print("Test statistic is less than 5% critical value, REJECT NULL HYPOTHESIS")

    if result[1] < 0.05:
        print("p-value is less than 0.05, REJECT NULL HYPOTHESIS")
    else:
        print("p-value is greater than 0.05, ACCEPT NULL HYPOTHESIS")
        
def fn_keltner_channels(df,kc_lookback_period=7,KC_mult_high= 1): 
        
        df["KC_basis"]        = ta.SMA(df.Close, kc_lookback_period)
        df["devKC"]           = ta.SMA(ta.TRANGE(df.High,df.Low,df.Close),kc_lookback_period)
        df["KC_upper_high"]   = df.KC_basis + df.devKC * KC_mult_high
        df["KC_lower_high"]   = df.KC_basis - df.devKC * KC_mult_high

        return df
def fn_relative_strength(prices, n=14):

        deltas = np.diff(prices)
        seed = deltas[:n + 1]
        up = seed[seed >= 0].sum() / n
        down = -seed[seed < 0].sum() / n
        rs = up / down
        rsi = np.zeros_like(prices)
        rsi[:n] = 100. - 100. / (1. + rs)

        for i in range(n, len(prices)):
            delta = deltas[i - 1]  # cause the diff is 1 shorter

            if delta > 0:
                upval = delta
                downval = 0.
            else:
                upval = 0.
                downval = -delta

            up = (up * (n - 1) + upval) / n
            down = (down * (n - 1) + downval) / n

            rs = up / down
            rsi[i] = 100. - 100. / (1. + rs)

        return rsi

# %% [markdown]
# ### FII DII Data

# %%
encodings_to_try = ['utf-8', 'latin1', 'ISO-8859-1']
dataDir = '/Users/nikhilsama/Dropbox/Coding/AlgoTrading/Analysis/Hrishikesh/Data/'

for encoding in encodings_to_try:
    try:
        FII_DII_Data = pd.read_csv(f'{dataDir}Fii_Dii.csv', encoding=encoding)
        break  # Stop trying encodings if successful
    except UnicodeDecodeError:
        pass
FII_DII_Data = pd.read_csv(f'{dataDir}Fii_Dii.csv', encoding=encoding)
FII_DII_Data['Date'] = pd.to_datetime(FII_DII_Data['Date'])

FII_DII_Data.set_index('Date', inplace=True)
print(FII_DII_Data.head())


# %% [markdown]
# ### OHLC Data with technical indicators

# %%
end_date = dt.datetime.now()
start_date = '2013-01-01'
df1=yf.download('^NSEI',start_date,end_date)
Nifty_OHLC=fn_keltner_channels(df1,kc_lookback_period=20,KC_mult_high= 1)
Nifty_OHLC['rsi'] =fn_relative_strength(Nifty_OHLC['Adj Close'], n=14)
vwap(Nifty_OHLC, label='vwap', window=3, fillna=True)
Nifty_OHLC

# %%
check_adfuller(Nifty_OHLC['Adj Close'])

# %% [markdown]
# #### PE,PB,DIV ratio data

# %%
from nsepython import *
import pandas as pd
symbol = "NIFTY 50"
start_date = "01-Jan-2013"
end_date = "11-Aug-2023"
pe_pb_data=pd.DataFrame(index_pe_pb_div(symbol,start_date,end_date))
pe_pb_data['Date'] = pd.to_datetime(pe_pb_data['DATE'])
pe_pb_data.set_index('Date', inplace=True)
#pe_pb_data.to_csv('pb_pe.csv')
pe_pb_data

# %% [markdown]
# ### Crude oil prices

# %%
end_date = dt.datetime.now()
start_date = '2013-01-01'
crude_prices=yf.download('CL=F',start_date,end_date)
vwap(crude_prices, label='vwap', window=3, fillna=True)
new_column_names = {'Open': 'crude_Open', 'High': 'crude_High','Low': 'crude_Low', 'Close': 'crude_Close','Adj Close': 'crude_Adj Close','Volume': 'crude_Volume','vwap':'crude_VWAP'}
crude_prices.rename(columns=new_column_names, inplace=True)
crude_prices

# %% [markdown]
# ### USD-INR Prices

# %%
end_date = dt.datetime.now()
start_date = '2013-01-01'
USD_INR=yf.download('USDINR=X',start_date,end_date)
new_column_names = {'Open': 'USD_INR_Open', 'High': 'USD_INR_High','Low': 'USD_INR_Low', 'Close': 'USD_INR_Close','Adj Close': 'USD_INR_Adj Close','Volume': 'USD_INR_Volume'}
USD_INR.rename(columns=new_column_names, inplace=True)
USD_INR

# %% [markdown]
# ### Gold prices

# %%
end_date = dt.datetime.now()
start_date = '2013-01-01'
Gold_price=yf.download('GC=F',start_date,end_date)
vwap(Gold_price, label='vwap', window=3, fillna=True)
new_column_names = {'Open': 'Gold_price_Open', 'High': 'Gold_price_High','Low': 'Gold_price_Low', 'Close': 'Gold_price_Close','Adj Close': 'Gold_price_Adj Close','Volume': 'Gold_price_Volume','vwap':'Gold_price_VWAP'}
Gold_price.rename(columns=new_column_names, inplace=True)
Gold_price

# %% [markdown]
# ### S&P 500

# %%
end_date = dt.datetime.now()
start_date = '2013-01-01'
SP500=yf.download('^GSPC',start_date,end_date)
vwap(SP500, label='vwap', window=3, fillna=True)
new_column_names = {'Open': 'SP500_Open', 'High': 'SP500_High','Low': 'SP500_Low', 'Close': 'SP500_Close','Adj Close': 'SP500_Adj Close','Volume': 'SP500_Volume','vwap':'SP500_VWAP'}
SP500.rename(columns=new_column_names, inplace=True)
SP500

# %%
end_date = dt.datetime.now()
start_date = '2013-01-01'
SGXNIFTY=yf.download('^SGXNIFTY',start_date,end_date)
vwap(SGXNIFTY, label='vwap', window=3, fillna=True)
#new_column_names = {'Open': 'SP500_Open', 'High': 'SP500_High','Low': 'SP500_Low', 'Close': 'SP500_Close','Adj Close': 'SP500_Adj Close','Volume': 'SP500_Volume','vwap':'SP500_VWAP'}
#SP500.rename(columns=new_column_names, inplace=True)
SGXNIFTY

# %% [markdown]
# ### Top 10 stocks

# %%
ticker_list=['HDFCBANK.NS','RELIANCE.NS','ICICIBANK.NS','INFY.NS','ITC.NS','TCS.NS','LT.NS','KOTAKBANK.NS','AXISBANK.NS','SBIN.NS']
df_Top_stocks=pd.DataFrame()
for ticker in ticker_list:
    end_date = dt.datetime.now()
    start_date = '2013-01-01'
    df=yf.download(ticker,start_date,end_date)
    vwap(df, label='vwap', window=3, fillna=True)
    df_Top_stocks[f'{ticker}_VWAP']=df['vwap']
    clear_output(wait=False)
df_Top_stocks

# %%
# all data combined in one dataframe
concatenated_df = pd.concat([Nifty_OHLC, FII_DII_Data,pe_pb_data,crude_prices,USD_INR,Gold_price,SP500,df_Top_stocks], axis=1)
concatenated_df.dropna(inplace=True,axis=0)

# Previous day values used to predict current day price

#columns_to_shift = ['Adj Close','Volume','vwap','KC_basis','devKC','rsi','FII_Net','DII_Net','pe','pb','crude_Adj Close','USD_INR_Adj Close','Gold_price_Adj Close','SP500_Adj Close','HDFCBANK.NS', 'RELIANCE.NS','ICICIBANK.NS','INFY.NS','ITC.NS','TCS.NS','LT.NS','KOTAKBANK.NS','AXISBANK.NS','SBIN.NS']
columns_to_shift = ['Adj Close','Volume','vwap','KC_basis','devKC','rsi','FII_Net','DII_Net','pe','pb','crude_VWAP','USD_INR_Adj Close','Gold_price_VWAP','SP500_VWAP','HDFCBANK.NS_VWAP', 'RELIANCE.NS_VWAP','ICICIBANK.NS_VWAP','INFY.NS_VWAP','ITC.NS_VWAP','TCS.NS_VWAP','LT.NS_VWAP','KOTAKBANK.NS_VWAP','AXISBANK.NS_VWAP','SBIN.NS_VWAP']

# Number of previous days to shift
num_days = 1
new_columns_list=[]
# Create previous day columns using a loop
for column in columns_to_shift:
    for i in range(1, num_days + 1):
        concatenated_df[f'{column}_prev_{i}'] = concatenated_df[column].shift(i)
        new_columns_list.append(f'{column}_prev_{i}')

concatenated_df.to_csv('x.csv')

# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

data =concatenated_df
data.dropna(inplace=True,axis=0)
# Splitting the data into features (input factors) and target variable
X = data[['Adj Close_prev_1','KC_basis_prev_1','rsi_prev_1','FII_Net_prev_1','DII_Net_prev_1','pe_prev_1','USD_INR_Adj Close_prev_1']]

y = data['Adj Close'] 

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Creating a linear regression model
model = LinearRegression()

# Training the model
model.fit(X_train, y_train)

# Making predictions
predictions = model.predict(X_test)

coefficients = model.coef_

# Printing the coefficients
for i, coef in enumerate(coefficients):
    print(f'\n Coefficient for {X.columns[i]}:', coef)

r_squared = model.score(X, y)
n=len(y)
k=7
adjusted_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - k - 1)

print("\n R-squared:", r_squared) 
print("\n Adj R-squared:", adjusted_r_squared) 
    
# Calculating the Mean Squared Error
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
print("\n Mean Squared Error:", mse)
print("\n Root Mean Squared Error:", rmse)
plt.figure(figsize=(10, 6))
plt.scatter(y_test, predictions, color='blue', label='Predicted vs. Actual')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='--', color='red', label='Perfect Prediction')
plt.xlabel('Actual Nifty Price')
plt.ylabel('Predicted Nifty Price')
plt.title('Predicted vs. Actual Nifty Price')
plt.legend()
plt.show()

# %%
# prediction for tomorrow's Nifty price

Prev_Day_Factors = np.array([[19428.30078,19656.9927734375,47.5080721330983,-3073.28,500.35,22.63,82.79989624]])
predicted_nifty_price = model.predict(Prev_Day_Factors)

print("Predicted Nifty Price for Tomorrow:", predicted_nifty_price)


