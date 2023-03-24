#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  5 14:26:36 2023

@author: nikhilsama
"""
import numpy as np
import pandas as pd
import math 
from datetime import date,timedelta,timezone
import datetime
import pytz

# set timezone to IST
ist = pytz.timezone('Asia/Kolkata')

def tenAMToday (now=0):
    if (now == 0):
        now = datetime.datetime.now(ist)

    # Set the target time to 8:00 AM
    tenAM = datetime.time(hour=10, minute=0, second=0, tzinfo=ist)
    
    # Combine the current date with the target time
    tenAMToday = datetime.datetime.combine(now.date(), tenAM)
    return tenAMToday

def addATR(df,n=14):
    dfc = df.copy()
    dfc['H-L'] = df['High'] - df['Low']
    dfc['H-PC'] = df['High'] - df['Adj Close'].shift(1)
    dfc['L-PC'] = df['Low'] - df['Adj Close'].shift(1)

    dfc['TR'] = df[['H-L','H-PC','L-PC']].max(axis=1)
    dfc['ATR'] = df['TR'].ewm(com=n,min_periods=n).mean()
    return dfc['ATR']

def ADX(DF, n=20):
    "function to calculate ADX"
    df = DF.copy()
    df["ATR"] = addATR(DF, n)
    df["upmove"] = df["High"] - df["High"].shift(1)
    df["downmove"] = df["Low"].shift(1) - df["Low"]
    df["+dm"] = np.where((df["upmove"]>df["downmove"]) & (df["upmove"] >0), df["upmove"], 0)
    df["-dm"] = np.where((df["downmove"]>df["upmove"]) & (df["downmove"] >0), df["downmove"], 0)
    df["+di"] = 100 * (df["+dm"]/df["ATR"]).ewm(alpha=1/n, min_periods=n).mean()
    df["-di"] = 100 * (df["-dm"]/df["ATR"]).ewm(alpha=1/n, min_periods=n).mean()
    df["ADX"] = 100* abs((df["+di"] - df["-di"])/(df["+di"] + df["-di"])).ewm(alpha=1/n, min_periods=n).mean()
    return df["ADX"]



def RSI(DF, n=14):
    "function to calculate RSI"
    df = DF.copy()
    df["change"] = df["Adj Close"] - df["Adj Close"].shift(1)
    df["gain"] = np.where(df["change"]>=0, df["change"], 0)
    df["loss"] = np.where(df["change"]<0, -1*df["change"], 0)
    df["avgGain"] = df["gain"].ewm(alpha=1/n, min_periods=n).mean()
    df["avgLoss"] = df["loss"].ewm(alpha=1/n, min_periods=n).mean()
    df["rs"] = df["avgGain"]/df["avgLoss"]
    df["rsi"] = 100 - (100/ (1 + df["rs"]))
    return df["rsi"]

def addSMA(df,fast=8,slow=26,superTrend=200):
    
    period = df.index[1]- df.index[0]
    period_hours = period.total_seconds() / 3600
    
    df['ma_superTrend'] = df['Adj Close'].ewm(com=superTrend, min_periods=superTrend).mean()
    df['ma_superTrend_pct_change'] = df['ma_superTrend'].pct_change()
    df['ma_superTrend_pct_change_ma'] = df['ma_superTrend_pct_change'].ewm(com=fast, min_periods=5).mean()
    df['ma_superTrend_pct_change_ma_per_hr'] = df['ma_superTrend_pct_change_ma']/period_hours
    df['superTrend'] = np.where(df['ma_superTrend_pct_change_ma_per_hr'] > .002,1,0)
    df['superTrend'] = np.where(df['ma_superTrend_pct_change_ma_per_hr'] < -.002,-1, df['superTrend'] )

    
    df['ma_slow'] = df['Adj Close'].ewm(com=slow, min_periods=slow).mean()
    df['ma_fast'] = df['Adj Close'].ewm(com=fast, min_periods=fast).mean()

def addBBStats(df,superLen=200,maLen=20,bandWidth=2,superBandWidth=2.5):
    # creating bollinger band indicators
    df['ma_superTrend'] = df['Adj Close'].ewm(com=superLen, min_periods=superLen).mean()
    df['ma_superTrend_pct_change'] = 10000*df['ma_superTrend'].pct_change()
    df['ma20'] = df['Adj Close'].rolling(window=maLen).mean()
    df['ma20_pct_change'] = 10000*df['ma20'].pct_change()
    df['ma20_pct_change_ma'] = df['ma20_pct_change'].ewm(com=maLen, min_periods=maLen).mean()
    df['std'] = df['Adj Close'].rolling(window=maLen).std()
    df['upper_band'] = df['ma20'] + (bandWidth * df['std'])
    df['lower_band'] = df['ma20'] - (bandWidth * df['std'])
    df['super_upper_band'] = df['ma20'] + (superBandWidth * df['std'])
    df['super_lower_band'] = df['ma20'] - (superBandWidth * df['std'])
    #df.drop(['Open','High','Low'],axis=1,inplace=True,errors='ignore')
    #df.tail(5)
    return df
    

def eom_effect(df):
    # BUY condition
    df['signal'] = np.where( (df.index.day > 25),1,0)
    df['signal'] = np.where( (df.index.day < 5),1,0)


def sma50(df): 
    df['sma50'] = df['Adj Close'].rolling(window=50).mean()
    
def sma50_bullish(df):
    sma50(df)
    entry_price = 0
    stop_loss = .99 #1%

    #df['signal'] = np.where(True,0,0)
    
    for i in range(len(df)):

        if (i == 0 or math.isnan(df['sma50'][i-1])):
            continue

        date = df.index[i]
        #print(date)

        #start fresh at beginning of the day
        #exit all trades at market close prev day
        if(date.hour == 9):
            df.loc[date,'signal'] = 0
            entry_price = 0

        if ((df['Adj Close'][i-1] > df['sma50'][i-1]) and 
                 (df['Adj Close'][i-2] <= df['sma50'][i-1])):
            print("Entering")
            print(date)
            print(i)
            df.loc[date,'signal'] = 1
            entry_price = df['Open'][i]
        
        #stop loss
        if (entry_price > 0):
            #print("entry price >0")
            if ((stop_loss * entry_price) > df['Low'][i]):
                df.loc[date,'signal'] = 0
                entry_price = 0

        #print (df.index[i])
    
def sma_cx (df,f=12,s=24):
    
    # creating bollinger band indicators
    df['slow_ma'] = df['Adj Close'].rolling(window=s).mean()
    df['fast_ma'] = df['Adj Close'].rolling(window=f).mean()
    cx_multiplier = 1
    
    # BUY condition
    df['signal'] = np.where( (df['fast_ma'] > (cx_multiplier*df['slow_ma'])) &
                              (df['fast_ma'].shift(-1) <= (cx_multiplier*(df['slow_ma'].shift(-1)))),1, 10)
    
    # SELL condition
    df['signal'] = np.where( (df['slow_ma'] > df['fast_ma']) &
                              (df['slow_ma'].shift(-1) <= df['fast_ma'].shift(-1)),-1,df['signal'])

    
def ema_cx (df,f=9,s=24):
    
    # creating bollinger band indicators
    df['slow_ma'] = df['Adj Close'].ewm(com=s, min_periods=s).mean()
    df['fast_ma'] = df['Adj Close'].ewm(com=f, min_periods=f).mean()
    
    # BUY condition
    df['signal'] = np.where( (df['fast_ma'] > (df['slow_ma'])) &
                              (df['fast_ma'].shift(1) <= ((df['slow_ma'].shift(1)))),1,float("nan"))
    
    # SELL condition
    df['signal'] = np.where( (df['slow_ma'] > df['fast_ma']) &
                              (df['slow_ma'].shift(1) <= df['fast_ma'].shift(1)),-1,df['signal'])
    # CLOSE all positions at EOD
    df['signal'] = np.where(((df.index.hour == 15) &
                               (df.index.minute == 25)),0,df['signal'])
    
def setInitialExitSignalonBBandBreach(df):
    df['signal'] = np.where((df.index.hour <15) &  
                            (((df['Adj Close'] < df['lower_band']) &
                            (df['Adj Close'].shift(1) >= df['lower_band'])) |
                            ((df['Adj Close'] > df['upper_band']) &
                             (df['Adj Close'].shift(1) <= df['upper_band']))),
                            0,float("nan"))
def closePositionsEODandSOD(df):
    # CLOSE all positions at EOD
    df['signal'] = np.where(((df.index.hour == 15) &
                               (df.index.minute == 1)),0,df['signal'])
    # Dont Open Positions before 10 am, so warm up ma in the morning
    df['signal'] = np.where(((df.index.hour == 9)),0,df['signal'])

def exitExtremeTrendConditions(df):
    ## IMPORTANT CALL THIS FUNCTIN IN THE BEGINNING, so that other
    ## MEAN REVERSION SIGNALS OVERRIDE THIS ONE

    ## EXIT MA 20 extreme slope conditions override BB signals.  Heavy slope up, 
    # just buy, and heavy slope down, just sell.
    df['signal'] = np.where((df.index.hour <15) &  
                            (df['ma20_pct_change'] < 1) &
                            (df['ma20_pct_change'].shift(1) > 1), 
                            0,df['signal'])

    df['signal'] = np.where((df.index.hour <15) &  
                            (df['ma20_pct_change'] > -1) &
                            (df['ma20_pct_change'].shift(1) < -1),
                            0,df['signal'])

def enterExtremeTrendPositions(df):
    ## IMPORTANT CALL THIS FUNCTIN IN THE END, so that it overrides other
    ## MEAN REVERSION SIGNALS
    ## MA 20 extreme slope conditions override BB signals.  Heavy slope up, 
    # just buy, and heavy slope down, just sell.
    df['signal'] = np.where((df.index.hour <15) &  
                            (df['ma20_pct_change'] > 1),
                            1,df['signal'])

    df['signal'] = np.where((df.index.hour <15) &  
                            (df['ma20_pct_change'] < -1),
                            -1,df['signal'])

def bollinger_band_cx(df,superLen=200,maLen=20,bandWidth=2,superBandWidth=2.5,
                      startTime = 0):
    if startTime == 0:
        startTime = datetime.datetime(2000,1,1,10,0,0) #Long ago :-)
        startTime = ist.localize(startTime)
        
    addBBStats(df,superLen,maLen,bandWidth,superBandWidth)
    
    #EXIT CONDITIONS -- Spefically put nan in the signal column 
    setInitialExitSignalonBBandBreach(df)
    
    # BUY condition
    # 1) Trading Hours, 2) Price crossing under lower band
    # 3) Super trend below super lower band, or if it is higher then at least it is 
    # trending downs
    df['signal'] = np.where((df.index >= startTime) &
                            (df.index.hour <15) &  
                            (df['Adj Close'] < df['lower_band']) #&
                            #(df['Adj Close'].shift(1) >= df['lower_band'])
                            ,1,df['signal'])
    
    # SELL condition
    # 1) Trading Hours, 2) Price crossing over upper band
    # 3) Super trend below super upper band, or if it is higher then at least it is 
    # trending down

    df['signal'] = np.where((df.index >= startTime) &
                            (df.index.hour <15) & 
                            (df['Adj Close'] > df['upper_band']) #&
#                            (df['Adj Close'].shift(1) <= df['upper_band'])
                            ,-1,df['signal'])
    
    closePositionsEODandSOD(df)
    
    return df
def bollinger_band_cx2(df,superLen=200,maLen=20,bandWidth=2,superBandWidth=2.5,
                      startTime = 0):
    if startTime == 0:
        startTime = datetime.datetime(2000,1,1,10,0,0) #Long ago :-)
        startTime = ist.localize(startTime)
        
    addBBStats(df,superLen,maLen,bandWidth,superBandWidth)
    
    #EXIT CONDITIONS
    setInitialExitSignalonBBandBreach(df)
    
    # BUY condition
    # 1) Trading Hours, 2) Price crossing under lower band
    # 3) Super trend below super lower band, or if it is higher then at least it is 
    # trending downs
    df['signal'] = np.where((df.index >= startTime) &
                            (df.index.hour <15) &  
                            (df['Adj Close'] < df['lower_band']) #&
                            #(df['Adj Close'].shift(1) >= df['lower_band'])
                            ,1,df['signal'])
    
    # SELL condition
    # 1) Trading Hours, 2) Price crossing over upper band
    # 3) Super trend below super upper band, or if it is higher then at least it is 
    # trending down

    df['signal'] = np.where((df.index >= startTime) &
                            (df.index.hour <15) & 
                            (df['Adj Close'] > df['upper_band']) #&
#                            (df['Adj Close'].shift(1) <= df['upper_band'])
                            ,-1,df['signal'])
    # EXIT condition
    df['signal'] = np.where((df.index >= startTime) &
                            (df.index.hour <15) &  
                            (((df['Adj Close'] < df['ma20']) &
                            (df['Adj Close'].shift(1) >= df['ma20'])) |
                             ((df['Adj Close'] > df['ma20']) &
                             (df['Adj Close'].shift(1) <= df['ma20'])))
                            ,0,df['signal'])

    closePositionsEODandSOD(df)
    
    return df

def bollinger_band_cx_w_basis_breakout (df,superLen=200,maLen=20,bandWidth=2,superBandWidth=2.5):
    addBBStats(df,superLen,maLen,bandWidth,superBandWidth)
    
    #EXIT CONDITIONS
    setInitialExitSignalonBBandBreach(df)
    
    ## IMPORTANT CALL THIS FUNCTIN IN THE BEGINNING, so that other
    ## MEAN REVERSION SIGNALS OVERRIDE THIS ONE

    ## EXIT MA 20 extreme slope conditions override BB signals.  Heavy slope up, 
    # just buy, and heavy slope down, just sell.
    exitExtremeTrendConditions(df)
    
    # BUY condition
    # 1) Trading Hours, 2) Price crossing under lower band
    # 3) Super trend below super lower band, or if it is higher then at least it is 
    # trending down
    df['signal'] = np.where((df.index.hour <15) &  
                            (df['Adj Close'] < df['lower_band']) &
                            (df['Adj Close'].shift(1) >= df['lower_band']),
                            1,df['signal'])
    # SELL condition
    # 1) Trading Hours, 2) Price crossing over upper band
    # 3) Super trend below super upper band, or if it is higher then at least it is 
    # trending down

    df['signal'] = np.where((df.index.hour <15) & 
                            (df['Adj Close'] > df['upper_band']) &
                            (df['Adj Close'].shift(1) <= df['upper_band']),
                            -1,df['signal'])
    
    
    ## IMPORTANT CALL THIS FUNCTIN IN THE END, so that it overrides other
    ## MEAN REVERSION SIGNALS
    ## MA 20 extreme slope conditions override BB signals.  Heavy slope up, 
    # just buy, and heavy slope down, just sell.
    enterExtremeTrendPositions(df)
    
    closePositionsEODandSOD(df)
    
    return df

def bollinger_band_cx_w_flat_superTrend (df,superLen=200,maLen=20,
                                         bandWidth=2,superBandWidth=2.5,
                                         startTime = 0, exitAtMA=False):
    if startTime == 0:
        startTime = datetime.datetime(2000,1,1,10,0,0) #Long ago :-)
        startTime = ist.localize(startTime)

    addBBStats(df,superLen,maLen,bandWidth,superBandWidth)
    
    #EXIT CONDITIONS
    #EXIT CONDITIONS
    setInitialExitSignalonBBandBreach(df)

    # BUY condition
    # 1) Trading Hours, 2) Price crossing under lower band
    # 3) Super trend below super lower band, or if it is higher then at least it is 
    # trending down
    df['signal'] = np.where((df.index.hour <15) &  
                            (df['Adj Close'] < df['lower_band']) &
                            #(df['Adj Close'].shift(1) >= df['lower_band']) &
                            ((df['ma_superTrend'] < df['super_upper_band'])),
                            1,df['signal'])
    
    # SELL condition
    # 1) Trading Hours, 2) Price crossing over upper band
    # 3) Super trend below super upper band, or if it is higher then at least it is 
    # trending down

    df['signal'] = np.where((df.index.hour <15) & 
                            (df['Adj Close'] > df['upper_band']) &
                            #(df['Adj Close'].shift(1) <= df['upper_band']) &
                            ((df['ma_superTrend'] > df['super_lower_band'])),
                            -1,df['signal'])
    
    if (exitAtMA):
        # EXIT condition
        df['signal'] = np.where((df.index >= startTime) &
                                (df.index.hour <15) &  
                                (((df['Adj Close'] < df['ma20']) &
                                (df['Adj Close'].shift(1) >= df['ma20'])) |
                                 ((df['Adj Close'] > df['ma20']) &
                                 (df['Adj Close'].shift(1) <= df['ma20'])))
                                ,0,df['signal'])

    
    closePositionsEODandSOD(df)
    return df

def bollinger_band_cx_w_flat_superTrend2 (df,superLen=200,maLen=20,
                                         bandWidth=2,superBandWidth=2.5,
                                         startTime=0, exitAtMA=True):
    return bollinger_band_cx_w_flat_superTrend (df,superLen,maLen,bandWidth,
                                                superBandWidth,exitAtMA)


def bollinger_band_cx_w_flat_superTrend_wHurdle (df,superLen=200,maLen=20,
                                          bandWidth=2,superBandWidth=2.5,
                                          stSlopeHurdle=0.001):
    addBBStats(df,superLen,maLen,bandWidth,superBandWidth)
    
    #EXIT CONDITIONS
    #EXIT CONDITIONS
    setInitialExitSignalonBBandBreach(df)

    # BUY condition
    # 1) Trading Hours, 2) Price crossing under lower band
    # 3) Super trend below super lower band, or if it is higher then at least it is 
    # trending down
    df['signal'] = np.where((df.index.hour <15) &  
                            (df['Adj Close'] < df['lower_band']) &
                            (df['Adj Close'].shift(1) >= df['lower_band']) &
                            ((df['ma_superTrend'] < df['super_upper_band']) |
                             (df['ma_superTrend_pct_change'] > stSlopeHurdle)),
                            1,df['signal'])
    
    # SELL condition
    # 1) Trading Hours, 2) Price crossing over upper band
    # 3) Super trend below super upper band, or if it is higher then at least it is 
    # trending down

    df['signal'] = np.where((df.index.hour <15) & 
                            (df['Adj Close'] > df['upper_band']) &
                            (df['Adj Close'].shift(1) <= df['upper_band']) &
                            ((df['ma_superTrend'] > df['super_lower_band']) |
                             (df['ma_superTrend_pct_change'] < stSlopeHurdle)),
                            -1,df['signal'])
    
    closePositionsEODandSOD(df)
    return df

def macd(df,f=20,s=50):
    df = df.copy()
    df["ma_fast"] = df["Adj Close"].ewm(span=f,min_periods=f).mean()
    df["ma_slow"] = df["Adj Close"].ewm(span=s,min_periods=s).mean()
    df["macd"] = df["ma_fast"] - df["ma_slow"]
    df["macd_signal"] = df["macd"].ewm(span=c, min_periods=c).mean()
    df['macd_his'] = df['macd'] = df['macd_signal']

def obv(df):
    # calculate the OBV column
    df['change'] = df['Adj Close'] - df['Open']
    df['direction'] = df['change'].apply(lambda x: 1 if x > 0 else -1 if x < 0 else 0)
    df['obv'] = df['direction'] * df['Volume']
    df['obv'] = df['obv'].cumsum()
    df['ma_obv'] = df['obv'].ewm(com=30, min_periods=5).mean()
    df['ma_obv_diff'] = df['ma_obv'].diff(5)
    df['obv_osc'] = df['ma_obv_diff'] / (df['ma_obv_diff'].max() - df['ma_obv_diff'].min())
    df['obv_trend'] = np.where(df['obv_osc'] > .1,1,0)
    df['obv_trend'] = np.where(df['obv_osc'] < -.1,-1,df['obv_trend'])

def mystrat(df,lenSuperTrend=200,lenSlow=26,lenFast=8):
    addSMA(df)
    obv(df)
    
    #BUY condition
    df['signal'] = np.where( (df['superTrend'] > 0) &
                              (df['obv_trend'] > 0) &
                              (df['ma_fast'] >= df['ma_slow']), 1, float("nan"))
    
    #SELL condition
    df['signal'] = np.where( (df['superTrend'] < 0) &
                              (df['obv_trend'] < 0) &
                              (df['ma_fast'] <= df['ma_slow']),-1,df['signal'])
      
    #EXIT condition
    df['signal'] = np.where(df['superTrend'] == 0,0,df['signal'])
    
    