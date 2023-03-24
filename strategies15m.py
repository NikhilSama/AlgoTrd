import signals

import numpy as np
import pandas as pd
from datetime import date,timedelta,timezone
import datetime
import pytz
# set timezone to IST
ist = pytz.timezone('Asia/Kolkata')

# Algorithmic trading strategy that uses timeseiies data to generate buy and sell signals
# for a given stock. The strategy is based on the following indicators:
# 1. Long Term Trend - 200 period moving average: If the 200 period moving average is sloping up, the trend is bullish
# if the 200 period moving average is sloping down, the trend is bearish
# 2. Bollinger Bands - If the trend is bullish, and the price is below the lower band, the signal is buy
# If the trend is bearish, and the price is above the upper band, the signal is sell
# In either case, exit the position when the prices crosses the middle band
### DOES NOT WORK WELL
def meanReversionStrategy (df, startTime=0):
    if startTime == 0:
        startTime = datetime.datetime(2000,1,1,10,0,0) #Long ago :-)
        startTime = ist.localize(startTime)

    df = signals.addBBStats(df)
    
    # fill df['signal'] with nan
    df['signal'] = np.nan
    
    # BUY condition
    # 1) Trading Hours, 2) Price crossing under lower band
    # 3) Super trend below super lower band, or if it is higher then at least it is 
    # trending downs
    df['signal'] = np.where((df.index >= startTime) &
                            (df.index.hour <15) &  
                            (df['Adj Close'] > df['lower_band']) &
                            (df['Adj Close'].shift(1) <= df['lower_band'])
                            ,1,df['signal'])
    
    # SELL condition
    # 1) Trading Hours, 2) Price crossing under upper band
    # 3) Super trend below super upper band, or if it is higher then at least it is 
    # trending down

    df['signal'] = np.where((df.index >= startTime) &
                            (df.index.hour <15) & 
                            (df['Adj Close'] < df['upper_band']) &
                            (df['Adj Close'].shift(1) >= df['upper_band'])
                            ,-1,df['signal'])
    
    # EXIT condition    
    # # 1) Trading Hours, 2) Price crossing over middle band or price crossing under middle band \
    # or price crossing over upper band or price crossing under lower band

    df['signal'] = np.where((df.index >= startTime) &
                            (((df['Adj Close'] < df['ma20']) &
                            (df['Adj Close'].shift(1) >= df['ma20'])) |
                            ((df['Adj Close'] > df['ma20']) &
                            (df['Adj Close'].shift(1) <= df['ma20'])) |
                            ((df['Adj Close'] < df['lower_band']) &
                            (df['Adj Close'].shift(1) >= df['lower_band'])) |
                            ((df['Adj Close'] > df['upper_band']) &
                            (df['Adj Close'].shift(1) <= df['upper_band'])))
                            ,0,df['signal'])
    return df

                            
