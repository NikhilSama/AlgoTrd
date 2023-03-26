#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  5 14:26:36 2023

@author: nikhilsama
"""
import log_setup
import numpy as np
import pandas as pd
import math 
from datetime import date,timedelta,timezone
import datetime
import pytz
import performance as perf
import logging

# set timezone to IST
ist = pytz.timezone('Asia/Kolkata')

def MACD(DF,f=20,s=50):
    df = DF.copy()
    df["ma_fast"] = df["Adj Close"].ewm(span=f,min_periods=f).mean()
    df["ma_slow"] = df["Adj Close"].ewm(span=s,min_periods=s).mean()
    df["macd"] = df["ma_fast"] - df["ma_slow"]
    df["macd_signal"] = df["macd"].ewm(span=c, min_periods=c).mean()
    df['macd_his'] = df['macd'] = df['macd_signal']
    return(df['macd_signal'],df['macd_his'])

def OBV(df,obvOscThresh=.1):
    df = df.copy()
    # calculate the OBV column
    df['change'] = df['Adj Close'] - df['Open']
    df['direction'] = df['change'].apply(lambda x: 1 if x > 0 else -1 if x < 0 else 0)
    df['obv'] = df['direction'] * df['Volume']
    df['obv'] = df['obv'].cumsum()
    df['ma_obv'] = df['obv'].ewm(com=30, min_periods=5).mean()
    df['ma_obv_diff'] = df['ma_obv'].diff(5)
    df['obv_osc'] = df['ma_obv_diff'] / (df['ma_obv_diff'].max() - df['ma_obv_diff'].min())
    df['obv_trend'] = np.where(df['obv_osc'] > obvOscThresh,1,0)
    df['obv_trend'] = np.where(df['obv_osc'] < -obvOscThresh,-1,df['obv_trend'])
    return (df['ma_obv'],df['obv_osc'],df['obv_trend'])

def tenAMToday (now=0):
    if (now == 0):
        now = datetime.datetime.now(ist)

    # Set the target time to 8:00 AM
    tenAM = datetime.time(hour=10, minute=0, second=0, tzinfo=ist)
    
    # Combine the current date with the target time
    tenAMToday = datetime.datetime.combine(now.date(), tenAM)
    return tenAMToday

def ATR(DF,n=14):
    df = DF.copy()
    df['H-L'] = df['High'] - df['Low']
    df['H-PC'] = df['High'] - df['Adj Close'].shift(1)
    df['L-PC'] = df['Low'] - df['Adj Close'].shift(1)

    df['TR'] = df[['H-L','H-PC','L-PC']].max(axis=1)
    df['ATR'] = df['TR'].ewm(com=n,min_periods=n).mean()
    return df['ATR']

def ADX(DF, n=20):
    "function to calculate ADX"
    df = DF.copy()
    df["ATR"] = ATR(DF, n)
    df["upmove"] = df["High"] - df["High"].shift(1)
    df["downmove"] = df["Low"].shift(1) - df["Low"]
    df["+dm"] = np.where((df["upmove"]>df["downmove"]) & (df["upmove"] >0), df["upmove"], 0)
    df["-dm"] = np.where((df["downmove"]>df["upmove"]) & (df["downmove"] >0), df["downmove"], 0)
    df["+di"] = 100 * (df["+dm"]/df["ATR"]).ewm(alpha=1/n, min_periods=n).mean()
    df["-di"] = 100 * (df["-dm"]/df["ATR"]).ewm(alpha=1/n, min_periods=n).mean()
    df["ADX"] = 100* abs((df["+di"] - df["-di"])/(df["+di"] + df["-di"])).ewm(alpha=1/n, min_periods=n).mean()
    return df["ADX"]


def RSI(DF, n=14):
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
    df['ma_superTrend_pct_change'] = 10000*df['ma_superTrend'].pct_change(periods=1)
    df['ma20'] = df['Adj Close'].rolling(window=maLen).mean()
    df['ma20_pct_change'] = 10000*df['ma20'].pct_change(periods=1)
    df['ma20_pct_change_ma'] = df['ma20_pct_change'].ewm(com=maLen, min_periods=maLen).mean()
    df['ma20_pct_change_ma_sq'] = 10000*df['ma20_pct_change_ma'].pct_change()
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


### UTILTIY FUNCTIONS

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


#####END OF UTILTIY FUNCTIONS#########

####### POPULATE FUNCTIONS #######
# Functions that populate the dataframe with the indicators
def populateBB (df,superLen=200,maLen=20,bandWidth=2,superBandWidth=2.5,
                      adxThresh = 30,maThresh = 1,obvOscThresh=.1):
    addBBStats(df,superLen,maLen,bandWidth,superBandWidth)

def populateADX (df,superLen=200,maLen=20,bandWidth=2,superBandWidth=2.5,
                      adxThresh = 30,maThresh = 1,obvOscThresh=.1):
    df['ADX'] = ADX(df,maLen)

def populateOBV (df,superLen=200,maLen=20,bandWidth=2,superBandWidth=2.5,
                      adxThresh = 30,maThresh = 1,obvOscThresh=.1):
    (df['OBV'],df['OBV-OSC'],df['OBV-TREND']) = OBV(df,obvOscThresh)
########### END OF POPULATE FUNCTIONS ###########

## SIGNAL GENERATION functions
# Functions that generate the signals based on the indicators
# type is 1 for entry or exit, 0 for exit only time frame close
# to end of trading day
def getSig_BB_CX(type,row,superLen=200,maLen=20,bandWidth=2,superBandWidth=2.5,
                      adxThresh = 30,maThresh = 1,obvOscThresh=.1):
    # First get the original signal
    signal = row['signal']
    # BUY condition
    # 1) Trading Hours, 2) Price crossing under lower band
    # 3) Super trend below super lower band, or if it is higher then at least it is 
    # trending downs
    if row['Adj Close'] <= row['lower_band']:
        signal = 1*type
    # SELL condition
    # 1) Trading Hours, 2) Price crossing over upper band
    # 3) Super trend below super upper band, or if it is higher then at least it is 
    # trending down
    if row['Adj Close'] >= row['upper_band']:
        signal = -1*type

    return signal

def getSig_ADX_FILTER (type,row,superLen=200,maLen=20,bandWidth=2,superBandWidth=2.5,
                      adxThresh = 30, maThresh = 1,obvOscThresh=.1):
    # First get the original signal
    signal = row['signal']

    # Since this is a FILTER, we only negate long and short signals
    # on extreme ADX.
    # for nan or 0, we just return the signal
    if row['ADX'] >= adxThresh:
        if signal == 1:
            if row['ma20_pct_change_ma'] < 0:
                logging.debug(f"ADX FILTERERD SIGNL TO 0: {row.symbol}@{row.name}")
                signal = 0
        elif signal == -1:
            if row['ma20_pct_change_ma'] > 0:
                logging.debug(f"ADX FILTERERD SIGNL TO 0: {row.symbol}@{row.name}")
                signal = 0

    return signal

def getSig_MASLOPE_FILTER (type,row,superLen=200,maLen=20,bandWidth=2,superBandWidth=2.5,
                      adxThresh = 30, maThresh = 1,obvOscThresh=.1):
    # First get the original signal
    signal = row['signal']

    # Since this is a FILTER, we only negate long and short signals
    # on extreme MSSLOPE.
    # for nan or 0, we just return the signal
    if signal == 1 and row['ma20_pct_change_ma'] >= maThresh:
        logging.debug(f"MASLOPE FILTERERD SIGNL TO 0: {row.symbol}@{row.name}")
        signal = 0
    elif signal == -1 and row['ma20_pct_change_ma'] <= -maThresh:
        logging.debug(f"MASLOPE FILTERERD SIGNL TO 0: {row.symbol}@{row.name}")
        signal = 0
        
    return signal


def getSig_OBV_FILTER (type,row,superLen=200,maLen=20,bandWidth=2,superBandWidth=2.5,
                      adxThresh = 30, maThresh = 1,obvOscThresh=.1):
    # First get the original signal
    signal = row['signal']
    
    # Since this is a FILTER, we only negate long and short signals
    # on extreme OBV.
    # for nan or 0, we just return the signal
    if signal == 1 and row['OBV-OSC'] <= -obvOscThresh:
        logging.debug(f"OBV FILTERERD SIGNL TO 0: {row.symbol}@index:{row.i} obv is:{row['OBV-OSC']} threshod is:{obvOscThresh} ")
        signal = 0
    elif signal == -1 and row['OBV-OSC'] >= obvOscThresh:
        logging.debug(f"OBV FILTERERD SIGNL TO 0: {row.symbol}@index:{row.i} obv is:{row['OBV-OSC']} threshod is:{obvOscThresh}")
        signal = 0
    return signal

### OVERRIDE SIGNAL GENERATORS
# These are the signal generators that override the other signals
# They are caleld with other signal generators have already come up
# with a signal.  These can override in extreme cases such as 
# sharpe declines or rises in prices or extreme ADX
def getSig_Extreme_ADX_OBV_MA20_OVERRIDE (row,last_signal,
        superLen=200,maLen=20,bandWidth=2,superBandWidth=2.5,
        adxThresh = 30, maThresh = 1,obvOscThresh=.1,overridMultiplier=1.2):
    # First get the original signal
    if abs(row['OBV-OSC']) >= obvOscThresh*overridMultiplier or \
        abs(row['ma20_pct_change_ma']) >= maThresh*overridMultiplier or \
        row['ADX'] >= adxThresh*overridMultiplier:
            print(f"Extreme ADX/OBV/MA20 OVERRIDE: {row.symbol}@{row.name}")
            logging.debug(f"Extreme ADX/OBV/MA20 OVERRIDE: {row.symbol}@{row.name}")
            return 0
    return last_signal

######### END OF SIGNAL GENERATION FUNCTIONS #########
def getSignal(row,sigGen, superLen,maLen,bandWidth,
            superBandWidth,startTime,startHour,endHour,exitHour,
            adxThresh,maThresh,obvOscThresh):
    s = float("nan")

    #Return nan if its not within trading hours
    if(row.name >= startTime) & \
        (row.name.hour > startHour):
        if row.name.hour < endHour:
            #Generate entry and exit signals
            return sigGen(1,row,superLen,maLen,bandWidth,
                        superBandWidth,adxThresh,maThresh,
                        obvOscThresh)
                          
        elif row.name.hour < exitHour:
            #Generate exit signals only
            return sigGen(0,row,superLen,maLen,bandWidth,
                        superBandWidth,adxThresh,maThresh,
                        obvOscThresh)
        else:
            return s  
    else:
        return s
          
    return s
## MAIN APPLY STRATEGY FUNCTION
def applyIntraDayStrategy(df,analyticsGenerators=[populateBB], signalGenerators=[getSig_BB_CX],
                        overrideSignalGenerators=[], superLen=200,maLen=20,bandWidth=2,superBandWidth=2.5,
                        startTime = 0, startHour = 10, endHour = 14, exitHour= 15, 
                        adxThresh = 30, maThresh = 1,obvOscThresh=.1):
    if startTime == 0:
        startTime = datetime.datetime(2000,1,1,10,0,0) #Long ago :-)
        startTime = ist.localize(startTime)
    df['signal'] = float("nan")
    
    for analGen in analyticsGenerators:
        analGen(df,superLen=superLen,maLen=maLen,bandWidth=bandWidth,
                    superBandWidth=superBandWidth, 
                    adxThresh=adxThresh,maThresh=maThresh,obvOscThresh=obvOscThresh)
            
    for sigGen in signalGenerators:
        # apply the condition function to each row of the DataFrame
        df['signal'] = df.apply(getSignal, 
            args=(sigGen, superLen,maLen,bandWidth,
                  superBandWidth,startTime,startHour,endHour,exitHour,
                  adxThresh,maThresh,obvOscThresh), axis=1)
    
    # If we have exitSignalGenerators and have a non 0 net position
    # at the end, then run them
    last_signal = df['signal'].loc[df['signal'].last_valid_index()]
    if len(overrideSignalGenerators) and \
        (last_signal == 1 or last_signal == -1):
        for ovSigGen in overrideSignalGenerators:
            df['signal'] = df.apply(ovSigGen, 
                args=(last_signal, superLen,maLen,bandWidth,
                      superBandWidth,startTime,startHour,endHour,exitHour,
                      adxThresh,maThresh,obvOscThresh), axis=1)

