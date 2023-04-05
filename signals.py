#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  5 14:26:36 2023

@author: nikhilsama
"""
import os
import subprocess
import log_setup
import numpy as np
import pandas as pd
import math 
from datetime import date,timedelta,timezone
import datetime
import pytz
import performance as perf
import logging
import pickle
#cfg has all the config parameters make them all globals here
import cfg
globals().update(vars(cfg))

# set timezone to IST
ist = pytz.timezone('Asia/Kolkata')

tickerStatus = {}

def updateCFG(ma_slope_thresh, ma_slope_thresh_yellow_multiplier, \
                         ma_slope_slope_thresh, obv_osc_thresh, \
                         obv_osc_thresh_yellow_multiplier, ovc_osc_slope_thresh, \
                         override_multiplier):
    global maSlopeThresh,maSlopeThreshYellowMultiplier, \
        maSlopeSlopeThresh,obvOscThresh, \
            obvOscThreshYellowMultiplier, \
                obvOscSlopeThresh,overrideMultiplier
    maSlopeThresh = ma_slope_thresh
    maSlopeThreshYellowMultiplier = ma_slope_thresh_yellow_multiplier
    maSlopeSlopeThresh = ma_slope_slope_thresh
    obvOscThresh =  obv_osc_thresh
    obvOscThreshYellowMultiplier = obv_osc_thresh_yellow_multiplier
    obvOscSlopeThresh = ovc_osc_slope_thresh
    overrideMultiplier = override_multiplier
    
def printCFG():
    print(f"maSlopeThresh: {maSlopeThresh}")
    print(f"maSlopeThreshYellowMultiplier: {maSlopeThreshYellowMultiplier}")
    print(f"maSlopeSlopeThresh: {maSlopeSlopeThresh}")
    print(f"obvOscThresh: {obvOscThresh}")
    print(f"obvOscThreshYellowMultiplier: {obvOscThreshYellowMultiplier}")
    print(f"obvOscSlopeThresh: {obvOscSlopeThresh}")
    print(f"overrideMultiplier: {overrideMultiplier}")
    
def MACD(DF,f=20,s=50):
    df = DF.copy()
    df["ma_fast"] = df["Adj Close"].ewm(span=f,min_periods=f).mean()
    df["ma_slow"] = df["Adj Close"].ewm(span=s,min_periods=s).mean()
    df["macd"] = df["ma_fast"] - df["ma_slow"]
    df["macd_signal"] = df["macd"].ewm(span=c, min_periods=c).mean()
    df['macd_his'] = df['macd'] = df['macd_signal']
    return(df['macd_signal'],df['macd_his'])

def OBV(df):
    df = df.copy()
    # calculate the OBV column
    df['change'] = df['Adj Close'] - df['Open']
    df['direction'] = df['change'].apply(lambda x: 1 if x > 0 else -1 if x < 0 else 0)
    df['obv'] = df['direction'] * df['Volume']
    df['obv'] = df['obv'].cumsum()
    df['ma_obv'] = df['obv'].ewm(com=20, min_periods=5).mean()
    df['ma_obv_diff'] = df['ma_obv'].diff(5)
    
    #OBV-Diff Max/Min diff should only look at previous candles, not future candles
    #Also restrict the lookback to cfgObvMaxMinDiff_MaxLookbackCandles, to keep backtest results consistent
    #apples to apples with live trading
    
    df['ma_obv_diff_max'] = df['ma_obv_diff'].rolling(window=cfgObvMaxMinDiff_MaxLookbackCandles, min_periods=1).max()
    df['ma_obv_diff_min'] = df['ma_obv_diff'].rolling(window=cfgObvMaxMinDiff_MaxLookbackCandles, min_periods=1).min()
    df['obv_osc'] = df['ma_obv_diff'] / (df['ma_obv_diff_max'] - df['ma_obv_diff_min'])
    df['obv_osc_pct_change'] = df['obv_osc'].diff(2)/2
    df['obv_trend'] = np.where(df['obv_osc'] > obvOscThresh,1,0)
    df['obv_trend'] = np.where(df['obv_osc'] < -obvOscThresh,-1,df['obv_trend'])
    
    # CLIP extreme
    df['obv_osc'] = df['obv_osc'].clip(lower=-1, upper=1)

    return (df['ma_obv'],df['obv_osc'],df['obv_trend'],df['obv_osc_pct_change'])

def tenAMToday (now=0):
    if (now == 0):
        now = datetime.datetime.now(ist)

    # Set the target time to 8:00 AM
    tenAM = datetime.time(hour=10, minute=0, second=0, tzinfo=ist)
    
    # Combine the current date with the target time
    tenAMToday = datetime.datetime.combine(now.date(), tenAM)
    return tenAMToday

def ATR(DF,n=atrLen):
    df = DF.copy()
    df['H-L'] = df['High'] - df['Low']
    df['H-PC'] = df['High'] - df['Adj Close'].shift(1)
    df['L-PC'] = df['Low'] - df['Adj Close'].shift(1)

    df['TR'] = df[['H-L','H-PC','L-PC']].max(axis=1)
    df['ATR'] = df['TR'].ewm(com=n,min_periods=n).mean()
    return df['ATR']

def ADX(DF, n=adxLen):
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
    df['ADX-PCT-CHNG'] = df['ADX'].diff(2)/2
    df['ADX-PCT-CHNG'] = df['ADX-PCT-CHNG'].clip(lower=-1, upper=1)

    return (df["ADX"],df['ADX-PCT-CHNG'])


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

def addBBStats(df):
    # creating bollinger band indicators
    df['ma_superTrend'] = df['Adj Close'].ewm(com=superLen, min_periods=superLen).mean()
    df['ma_superTrend_pct_change'] = 10000*df['ma_superTrend'].pct_change(periods=1)
    df['ma20'] = df['Adj Close'].rolling(window=maLen).mean()
    df['MA-FAST'] = df['Adj Close'].rolling(window=fastMALen).mean()
    df['ma20_pct_change'] = df['ma20'].pct_change(periods=1)
    df['ma20_pct_change_ma'] = df['ma20_pct_change'].ewm(com=5, min_periods=1).mean()
    df['ma20_pct_change_ma_sq'] = df['ma20_pct_change_ma'].pct_change()
    df['ma20_pct_change_ma_sq'] = df['ma20_pct_change_ma_sq'].ewm(com=maLen, min_periods=maLen).mean()
    df['ma20_pct_change_ma_sq'] = df['ma20_pct_change_ma_sq'].clip(lower=-0.5, upper=0.5)
    df['std'] = df['Adj Close'].rolling(window=maLen).std()
    df['upper_band'] = df['ma20'] + (bandWidth * df['std'])
    df['lower_band'] = df['ma20'] - (bandWidth * df['std'])
    df['super_upper_band'] = df['ma20'] + (superBandWidth * df['std'])
    df['super_lower_band'] = df['ma20'] - (superBandWidth * df['std'])
    #df.drop(['Open','High','Low'],axis=1,inplace=True,errors='ignore')
    #df.tail(5)
    slopeStdDev = df['ma20_pct_change_ma'].std()
    slopeMean = df['ma20_pct_change_ma'].mean()
    df['SLOPE-OSC'] = (df['ma20_pct_change_ma'] - slopeMean)/slopeStdDev
    df['SLOPE-OSC-SLOPE'] = df['SLOPE-OSC'].diff(2)/2
    return df
        
def eom_effect(df):
    # BUY condition
    df['signal'] = np.where( (df.index.day > 25),1,0)
    df['signal'] = np.where( (df.index.day < 5),1,0)


### UTILTIY FUNCTIONS

def isGettingHigher(slope,threshold):
    return slope >= threshold
def isGettingLower(slope,threshold):
    return slope <= -threshold
def isNotChanging(slope,threshold):
    return not (isGettingHigher(slope,threshold) 
                or isGettingLower(slope,threshold))
def isSuperHigh(value,threshold,multiplier):
    return value >= threshold/multiplier
def isHigh(value,threshold,multiplier):
    return (not isSuperHigh(value,threshold,multiplier)) and \
        value >= threshold
def isAlmostHigh(value,threshold,multiplier):
    return (not (isHigh(value,threshold,multiplier) 
                 or isSuperHigh(value,threshold,multiplier))) and \
        (value >= threshold*multiplier)
def isSuperLow(value,threshold,multiplier):
    return value <= -threshold/multiplier
def isLow(value,threshold,multiplier):
    return (not isSuperLow(value,threshold,multiplier)) and \
        value <= -threshold
def isAlmostLow(value,threshold,multiplier):
    return (not (isLow(value,threshold,multiplier) 
                 or isSuperLow(value,threshold,multiplier))) and \
        (value <= -threshold*multiplier)

def valueBreachedThreshold (value,threshold,slope,
                            slopeThreshold,multiplier,type='H'):
    superHigh = value >= threshold/multiplier
    High = (not superHigh) and value >= threshold
    almostHigh = (not (High or superHigh)) and \
        (value >= threshold*multiplier)

    superLow = value <= -threshold/multiplier
    Low = (not superLow) and value <= -threshold
    almostLow = (not (Low or superLow)) and \
        (value <= -threshold*multiplier)


    gettingHigher = slope >= slopeThreshold
    gettingLower = slope <= -slopeThreshold
    notchanging = not (gettingHigher or gettingLower)
    
    highBreached = superHigh or \
            (High and (not gettingLower)) or \
            (almostHigh and gettingHigher)  
            
    lowBreached = superLow or \
            (Low and (not gettingHigher)) or \
            (almostLow and gettingLower)

    if type == 'H':
        return highBreached
    elif type == 'L':
        return lowBreached
    elif type == 'H_OR_L':
        if highBreached:
            return 'H'
        elif lowBreached:
            return 'L'
        else:
            return False
    else:
        logging.error(f"unknown type {type}")
        return False

def projectedValueBreachedThreshold(value,threshold,slope,
                                    slopeThreshold,multiplier,type):
    #Calc projectedValue is where the value will be after {candlesToProject}
    #if it starts at value and continues at current slope levels
    for counter in range(numCandlesForSlopeProjection):
        #logging.debug(f"counter is {counter} and value is {value} and slope is {slope} and threshold is {threshold}")
        value = (value + slope) # calculate new value of x after one candle
    return valueBreachedThreshold(value,threshold,0,0.1,multiplier,type)

def valueOrProjectedValueBreachedThreshold(value,threshold,slope,
                                    slopeThreshold,multiplier,type):
    p = projectedValueBreachedThreshold(value,threshold,slope,
                                    slopeThreshold,multiplier,type)
    if p == False:
        return valueBreachedThreshold(value,threshold,slope,
                                    slopeThreshold,multiplier,type)
    else:
        return p
def reversalWillComplete(value,threshold,slope,
                         slopeThreshold,multiplier):
    superHigh = value >= threshold/multiplier
    High = (not superHigh) and value >= threshold
    almostHigh = (not (High or superHigh)) and \
        (value >= threshold*multiplier)

    superLow = value <= -threshold/multiplier
    Low = (not superLow) and value <= -threshold
    almostLow = (not (Low or superLow)) and \
        (value <= -threshold*multiplier)
    
    
def crossOver(fast,slow,lastFast,lastSlow):
    return fast > slow and lastFast <= lastSlow
def crossUnder(fast,slow,lastFast,lastSlow):
    return fast < slow and lastFast >= lastSlow

def getLastSignal(signal):
    last_signal_index = signal.last_valid_index()
    if(last_signal_index):
        last_signal = signal.loc[last_signal_index]
    else:
        last_signal = 0
    return last_signal
def initTickerStatus(ticker):
    tickerStatus[ticker] = {'position':float("nan"), 
                                    'trendSignal':float("nan")} 
    loadTickerStatusFromCache()
    
def loadTickerStatusFromCache():
    tickerStatusCacheFileName = "Data/tickerStatusCache.pickle"
    if os.path.isfile(tickerStatusCacheFileName):
        with open(tickerStatusCacheFileName, "rb") as f:
            tickerStatus = pickle.load(f)
            logging.info(f"loaded tickerStatus from cache: {tickerStatus}")

def cacheTickerStatus():
    tickerStatusCacheFileName = "Data/tickerStatusCache.pickle"
    with open(tickerStatusCacheFileName,"wb") as f:
        pickle.dump(tickerStatus,f)

def getTickerPosition(ticker):
    if ticker not in tickerStatus:
        return float ('nan')
    elif 'position' not in tickerStatus[ticker]:
        return float ('nan')
    else:
        return tickerStatus[ticker]['position']

def setTickerPosition(ticker,p):
    if ticker not in tickerStatus:
        tickerStatus[ticker] = {}
    tickerStatus[ticker]['position'] = p
    cacheTickerStatus()

def getTickerTrend(ticker):
    if ticker not in tickerStatus:
        return float ('nan')
    elif 'trendSignal' not in tickerStatus[ticker]:
        return float ('nan')
    else:
        return tickerStatus[ticker]['trendSignal']

def setTickerTrend(ticker,trend):
    if ticker not in tickerStatus:
        tickerStatus[ticker] = {}
    tickerStatus[ticker]['trendSignal'] = trend

def tickerIsTrending(ticker):
    t = getTickerTrend(ticker)
    return True if (t == 1 or t == -1) else False

def tickerIsTrendingUp(ticker):
    t = getTickerTrend(ticker)
    return True if (t == 1) else False

def tickerIsTrendingDn(ticker):
    t = getTickerTrend(ticker)
    return True if (t == -1) else False

def isLongSignal(s):
    return s == 1

def isShortSignal(s):
    return s == -1

def isExitSignal(s):
    return s == 0

def isSignal(s):
    return np.isnan(s) == False

def isLongOrShortSignal(s):
    return isLongSignal(s) or isShortSignal(s)

def signalChanged(s,lastS):
    #Oddly s!=lastS is True if both are nan, because nan's cant compare
    return s != lastS and (not math.isnan(s))

def logSignal(msg,reqData,signal,s,row,window,isLastRow,extra='',logWithNoSignalChange=False):
    rowInfo = f'{row.symbol}:{row.i} '
    rowPrice = f'p:{round(row["Adj Close"],1)} '
    sigInfo = f'sig:{"-" if np.isnan(signal) else signal} s:{"-" if np.isnan(s) else s} {"E" if window == 1 else "X"} '
    dataStrings = {
        "adxData" : f"ADX:{round(row['ADX'],1)} > {adxThresh}(*{round(adxThreshYellowMultiplier,1)}) adxSLP:{round(row['ADX-PCT-CHNG'],2)}>{adxSlopeThresh} ",
        "maSlpData" : f"maSlp:{round(row['SLOPE-OSC'],2)} >= {maSlopeThresh}(*{maSlopeThreshYellowMultiplier}) maSlpChng:{round(row['SLOPE-OSC-SLOPE'],2)}>{maSlopeSlopeThresh} ",
        "obvData" : f"OBV:{round(row['OBV-OSC'],2)} > {obvOscThresh}(*{obvOscThreshYellowMultiplier}) obvSLP:{round(row['OBV-OSC-PCT-CHNG'],2)}>{obvOscSlopeThresh} "
    }
    dataString = ''
    for key in reqData:
        dataString = dataString+dataStrings[key]
    if isLastRow and (signalChanged(s,signal) or logWithNoSignalChange):
        logging.info(rowInfo+' => '+rowPrice+' => '+msg+' '+sigInfo+ ' => '+dataString+' '+extra)
    elif signalChanged(s,signal) or logWithNoSignalChange:
        rowTime = row.name.strftime("%d/%m %I:%M")
        rowInfo = rowInfo+f':{rowTime} ' #backtest needs date
        logging.debug(rowInfo+' => '+rowPrice+' => '+msg+' '+sigInfo+ ' => '+dataString+' '+extra)


#####END OF UTILTIY FUNCTIONS#########

####### POPULATE FUNCTIONS #######
# Functions that populate the dataframe with the indicators
def populateBB (df):
    addBBStats(df)

def populateADX (df):
    (df['ADX'],df['ADX-PCT-CHNG']) = ADX(df,maLen)

def populateOBV (df):
    if (df['Volume'].max() == 0):
        return False # Index has no volume data so skip it
    (df['OBV'],df['OBV-OSC'],df['OBV-TREND'],df['OBV-OSC-PCT-CHNG']) = OBV(df)
########### END OF POPULATE FUNCTIONS ###########

## SIGNAL GENERATION functions
# Functions that generate the signals based on the indicators
# type is 1 for entry or exit, 0 for exit only time frame close
# to end of trading day
def getSig_BB_CX(type,signal, isLastRow, row, df):
    # First get the original signal
    s = signal

    # BUY condition
    # 1) Trading Hours, 2) Price crossing under lower band
    # 3) Super trend below super lower band, or if it is higher then at least it is 
    # trending downs
    if (row['Adj Close'] <= row['lower_band']) and \
        (not tickerIsTrendingDn(row['symbol'])): #Dn trending tickers dance around the lower band, dont use that as an exit signal:
        s = 1*type
        
    # SELL condition
    # 1) Trading Hours, 2) Price crossing over upper band
    # 3) Super trend below super upper band, or if it is higher then at least it is 
    # trending down
    if (row['Adj Close'] >= row['upper_band']) and \
        (not tickerIsTrendingUp(row['symbol'])): #Up trending tickers dance around the upper band, dont use that as an exit signal
        s = -1*type

    logSignal('BB-X-CX',["adxData"],signal,s,row,type,isLastRow)
    # if isLastRow and signalChanged(s,signal):
    #     logging.info(f'{row.symbol}:{row.i} => BB-CX => signal={signal} s={s} type={type}')
    # elif signalChanged(s,signal):
    #     logging.debug(f'{row.symbol}:{row.i} => BB-CX => signal={signal} s={s} type={type}')

    return s

# START OF BB FILTERS
# Filters only negate BB signals to 0 under certain trending conditions
# Never give buy or sell signals if BB is not already signalling it 

# ADX is best for strength of trend, catches even
# gentle sloping ma, with low OBV, as long as it is long lived
def getSig_ADX_FILTER (type,signal, isLastRow,row,df):
    # Since this is a FILTER, we only negate long and short signals
    # on extreme ADX, with MA SLOPE pointing the opposite direction of signal
    # for nan or 0, we just return the signal
    if not isLongOrShortSignal(signal):
        return signal
    
    s = signal
    
    if valueOrProjectedValueBreachedThreshold(row['ADX'],adxThresh,row['ADX-PCT-CHNG'],
                                              adxSlopeThresh,adxThreshYellowMultiplier,"H"):
        
        i = df.index.get_loc(row.name) # get index of current row
        rollbackCandles = round(adxLen*.6) # how many candles to look back
        # Since ADX is based on last 14 candles, we should look
        # at slope in the median of that period to understand 
        # the direction of slope that created this ADX trend
        # if the trend reverses, ADX may stay high, but slope may
        # reverse.  Therefore we need to rollback and look at 
        # old slops to relate it to ADX value
        oldSlope = df.iloc[i - rollbackCandles,
                          df.columns.get_loc('SLOPE-OSC')]  
        if s == 1 and oldSlope < 0:
            s = 0
        elif s == -1 and oldSlope > 0:
            s = 0
    
    logSignal('ADX-FLTR',["adxData","maSlpData"],signal,s,row,type,isLastRow)

    # if isLastRow and signalChanged(s,signal):
    #     logging.info(f"{row.symbol}:{row.i}  => ADX FILTER => ADX:{row['ADX']} > Thresh:{adxThresh} ADXSLOPE:{row['ADX-PCT-CHNG']} MASLOPE:{row['SLOPE-OSC']} Mult: {adxThreshYellowMultiplier} signal={signal} s={s} type={type}")
    # elif signalChanged(s,signal):
    #     logging.debug(f"{row.symbol}:{row.i}:{row.name} => ADX FILTER => ADX:{row['ADX']} > Thresh:{adxThresh}  ADXSLOPE:{row['ADX-PCT-CHNG']} MASLOPE:{row['SLOPE-OSC']} Mult: {adxThreshYellowMultiplier} signal={signal} s={s} type={type}")
        
    return s

# MA SLOPE FILTER is to catch spikes, get out dont get caught in them
def getSig_MASLOPE_FILTER (type,signal, isLastRow,row,df):
    if not isLongOrShortSignal(signal):
        return signal
    
    s=signal
    breached = valueOrProjectedValueBreachedThreshold(row['SLOPE-OSC'],
                                      maSlopeThresh,row['SLOPE-OSC-SLOPE'],
                                      maSlopeSlopeThresh, maSlopeThreshYellowMultiplier, "H_OR_L")

    # Since this is a FILTER, we only negate long and short signals
    # on extreme MSSLOPE.
    # for nan or 0, we just return the signal
    
    if breached == False:
        return s
    else:
        if s == 1 and breached == 'L':
            # We want to go long, but the ticker is diving down, ma pct change is too low
            # so we filter this signal out
            s = 0
        elif s == -1 and breached == 'H':
            s = 0
    
    logSignal('SLP-FLTR',["maSlpData"],signal,s,row,type,isLastRow)

    # if isLastRow and signalChanged(s,signal):
    #     logging.info(f"{row.symbol}:{row.i}  => MASLOPE FILTER => Slope:{row['SLOPE-OSC']} >= Threshold {maSlopeThresh} SlopeChange:{row['SLOPE-OSC-SLOPE']} Mult: {maSlopeThreshYellowMultiplier} signal={signal} s={s} type={type}")
    # elif signalChanged(s,signal):
    #     logging.debug(f"{row.symbol}:{row.i}:{row.name}  => MASLOPE FILTER => Slope:{row['SLOPE-OSC']} >= Threshold {maSlopeThresh}  SlopeChange:{row['SLOPE-OSC-SLOPE']} Mult: {maSlopeThreshYellowMultiplier} signal={signal} s={s} type={type}")
        
    return s

#OBV is best as a leading indicator, flashes and spikes before 
#the price moves dramatically and it beomes and trend that shows up
#in MA or ADX etc 
def getSig_OBV_FILTER (type,signal, isLastRow,row, df):
    
    if (not isLongOrShortSignal(signal)) or (not 'OBV-OSC' in row):
        return signal

    s = signal
    
    # Since this is a FILTER, we only negate long and short signals
    # on extreme OBV.
    # for nan or 0, we just return the signal
    
    breached = valueOrProjectedValueBreachedThreshold(row['OBV-OSC'],obvOscThresh,
                    row['OBV-OSC-PCT-CHNG'], obvOscSlopeThresh, 
                    obvOscThreshYellowMultiplier, "H_OR_L")

    if breached == False:
        return s
    else:
        if s == 1 and breached == 'L':
            s = 0
        elif s == -1 and breached == 'H':
            s = 0
        # if isLastRow and signalChanged(s,signal):
        #     logging.info(f"{row.symbol}:{row.i}  => OBV FILTERER => obv is:{row['OBV-OSC']} >= threshod:{obvOscThresh} Mult: {obvOscThreshYellowMultiplier} Slope: {row['OBV-OSC-PCT-CHNG']} >= {obvOscSlopeThresh} signal={signal} s={s} type={type}")
        # elif signalChanged(s,signal):
        #     logging.debug(f"{row.symbol}:{row.i}:{row.name}  => OBV FILTERER =>  obv is:{row['OBV-OSC']} >= threshod:{obvOscThresh} Mult: {obvOscThreshYellowMultiplier} Slope: {row['OBV-OSC-PCT-CHNG']} >= {obvOscSlopeThresh} signal={signal} s={s} type={type}")
    logSignal('OBV-FLTR',["obvData"],signal,s,row,type,isLastRow)

    return s

## END OF FILTERS

### OVERRIDE SIGNAL GENERATORS
# These are the signal generators that override the other signals
# They are caleld with other signal generators have already come up
# with a signal.  These can override in extreme cases such as 
# sharpe declines or rises in prices or extreme ADX, and can 
# provide buy/sell signals that override BB for breakout trends
def getSig_exitAnyExtremeADX_OBV_MA20_OVERRIDE (type, signal, isLastRow, row, df, 
                                                last_signal=float('nan')):    
    
    # Three potential strategies here -- all work
    # 1) Set signal to 0 on extreme conditions regardless of last signal or current signal
    # no need to even look at signal or last signal
    # in this case we will exit even when inside BB band if conditions are extreme 
    # df will have repeated exit signal 0 every time this is the case
    #
    # 2) Set signal to 0 only if current signal is 1 or -1 and last signal is 1 or -1
    # in this case we will exit even when inside BB band if conditions are extreme 
    # contrary to intuition, we will still have repeated exit signal 0 every time this is the case, 
    # because last_signal will not be updated for the signals created by overrideSignals, 
    #
    # 3) Set signal to 0 only if current signal is 1 or -1, overriding BB signals
    # in this case we *WILL NOT* explicitly exit on extreme conditions before BB is hit
    # but this may be ok since bb will usually be hit when conditions are extreme
    #
    # NOTE: Current strategy is 3, achieved by calling this function from signal generator, 
    # and not from overrideSignals, therefore last_signal will be nan
    
    if isExitSignal(signal):
        return signal # nothing to override, we exiting already
    
    if last_signal in [1,-1]:
        if signal in [1,-1]:
            positionToAnalyse = signal
        else:
            positionToAnalyse = last_signal #signal is nan
    elif signal in [1,-1]:
        positionToAnalyse = signal
    else:
        return signal # signal is nan (not -1,0,1) and last signal is nan or 0
                    #we have not current or planned position

    if ('OBV-OSC' in row) and (not np.isnan(row['OBV-OSC'])):
        obvBreached = valueBreachedThreshold(row['OBV-OSC'],obvOscThresh,
                                    row['OBV-OSC-PCT-CHNG'],obvOscSlopeThresh,
                                    obvOscThreshYellowMultiplier, "H_OR_L")
    else:
        obvBreached = False
    
    if not np.isnan(row['ADX']):
        adxBreached = valueBreachedThreshold(row['ADX'],adxThresh,
                                    row['ADX-PCT-CHNG'],adxSlopeThresh,
                                    adxThreshYellowMultiplier, "H")
    else:
        adxBreached = False
    
    if not np.isnan(row['SLOPE-OSC']):
        slopebreached = valueBreachedThreshold(row['SLOPE-OSC'],
                                        maSlopeThresh,row['SLOPE-OSC-SLOPE'],
                                        maSlopeSlopeThresh, 
                                        maSlopeThreshYellowMultiplier, "H_OR_L")
    else:
        slopebreached = False
        
    if obvBreached == False and adxBreached == False and slopebreached == False:
        return signal    
    
    s = signal 
    #logging.info(f"obvBreached:{obvBreached} adxBreached:{adxBreached} slopebreached:{slopebreached}")
    if obvBreached:
        breach = obvBreached
    elif slopebreached:
        breach = slopebreached
    elif adxBreached:
        i = df.index.get_loc(row.name) # get index of current row
        rollbackCandles = round(adxLen*.6) # how many candles to look back
        # Since ADX is based on last 14 candles, we should look
        # at slope in the median of that period to understand 
        # the direction of slope that created this ADX trend
        # if the trend reverses, ADX may stay high, but slope may
        # reverse.  Therefore we need to rollback and look at 
        # old slops to relate it to ADX value
        oldSlope = df.iloc[i - rollbackCandles,
                          df.columns.get_loc('SLOPE-OSC')]  
        if oldSlope < 0:
            breach = 'L'
        else:
            breach = 'H'
            
    if positionToAnalyse == 1 and breach == 'L':
        s = 0
    elif positionToAnalyse == -1 and breach == 'H':
        s = 0
        
    logSignal('EXIT-EXTRME-COND',["obvData","adxData","maSlpData"],signal,s,row,type,isLastRow)

    # if isLastRow and signalChanged(s,signal):
    #     logging.info(f"{row.symbol}:{row.i}  => EXIT => Extreme ADX/OBV/MA20 OVERRIDE SIGNAL({signal})  s={s} / LAST SIGNAL({last_signal}). ADX:{row['ADX']} > {adxThresh*overrideMultiplier} OR OBV:{row['OBV-OSC']}/{row['OBV-OSC-PCT-CHNG']} > {obvOscThresh*overrideMultiplier} OR MA20:{row['SLOPE-OSC']}/{row['SLOPE-OSC-SLOPE']} > {maSlopeThresh*overrideMultiplier}")                
    # elif signalChanged(s,signal):
    #     logging.debug(f"{row.symbol}:{row.i}:{row.name}  => EXIT => Extreme ADX/OBV/MA20 OVERRIDE SIGNAL({signal})   s={s} / LAST SIGNAL({last_signal}). ADX:{row['ADX']} > {adxThresh*overrideMultiplier} OR OBV:{row['OBV-OSC']}/{row['OBV-OSC-PCT-CHNG']} > {obvOscThresh*overrideMultiplier} OR MA20:{row['SLOPE-OSC']}/{row['SLOPE-OSC-SLOPE']} > {maSlopeThresh*overrideMultiplier}")                
                
    return s

def getSig_followAllExtremeADX_OBV_MA20_OVERRIDE (type, signal, isLastRow, row, df, 
                                                  last_signal=float('nan')):
    
    if type == 0:# Dont take new positions when its time to exit only
        return signal
    s = signal 
    adxIsHigh = 1 if row['ADX'] >= adxThresh else 0
    obvIsHigh = 1 if (('OBV-OSC' in row) and (abs(row['OBV-OSC']) >= obvOscThresh)) else 0
    slopeIsHigh = 1 if abs(row['SLOPE-OSC']) >= maSlopeThresh else 0
    
    obvOsc = None if (not 'OBV-OSC' in row) else row['OBV-OSC']
    obvIsPositive = True if ((obvOsc is None) or (obvOsc > 0)) else False
    obvIsNegative = True if ((obvOsc is None) or (obvOsc < 0)) else False
    
    if obvOsc is None:
        num_conditions = cfgNumConditionsForTrendFollow - 1
    else:
        num_conditions = cfgNumConditionsForTrendFollow
    
    maSlopesUp = row['SLOPE-OSC'] > 0
    maSlopesDn = row['SLOPE-OSC'] < 0
    if (adxIsHigh + obvIsHigh + slopeIsHigh) >= num_conditions:
        #We are breaking out Ride the trend
        #print(f"Extreme ADX/OBV/MA20 OVERRIDE FOLLOW TREND: {row.symbol}@{row.name}")
        if obvIsPositive and maSlopesUp and signal != 1:
            if (last_signal != 1 and signal != 1):
                s = 1
        elif obvIsNegative and maSlopesDn and signal != -1:
            if (last_signal != -1 and signal != -1):
                s = -1
        
        if signalChanged(s,signal):
            #print("entering trend following", row.i)
            setTickerTrend(row.symbol, s)
            # if isLastRow:
            #     logging.info(f"{row.symbol}:{row.i} => FOLLOW TREND => Extreme ADX/OBV/MA20 OVERRIDE signal ({signal})  s={s} / last_signal ({last_signal}) TO FOLLOW TREND.ADX:{row['ADX']} > {adxThresh} AND OBV:{obvOsc} > {obvOscThresh} AND MA20:{row['SLOPE-OSC']} > {maSlopeThresh}")
            # else:
            #     logging.debug(f"{row.symbol}:{row.i}:{row.name}  => FOLLOW TREND => Extreme ADX/OBV/MA20 OVERRIDE signal ({signal})  s={s} / last_signal ({last_signal}) TO FOLLOW TREND.ADX:{row['ADX']} > {adxThresh} AND OBV:{obvOsc} > {obvOscThresh} AND MA20:{row['SLOPE-OSC']} > {maSlopeThresh}")
    logSignal('FLW-TRND',["obvData","adxData","maSlpData"],signal,s,row,type,isLastRow)

    return s

def exitTrendFollowing(type, signal, isLastRow, row, df, 
                        last_signal=float('nan')):
    if (not tickerIsTrending(row.symbol)) or \
        ((not np.isnan(signal)) and (getTickerTrend(row.symbol) == signal)):
        return signal
    
    #If We get here then ticker is trending, and trend signal no longer matches the trend
    #Trend may not continue, need to find a good spot to exit 
    s = signal
    i = df.index.get_loc(row.name) # get index of current row
    # oldSlowMA = df.iloc[i - 1,df.columns.get_loc('ma20')]  
    # oldFastMA = df.iloc[i - 1,df.columns.get_loc('MA-FAST')]
    
    adxIsNotChanging = isNotChanging(row['ADX-PCT-CHNG'], adxSlopeThresh)
    adxIsGettingLower = isGettingLower(row['ADX-PCT-CHNG'], adxSlopeThresh)
    
    maIsNotChanging = isNotChanging(row['SLOPE-OSC'], maSlopeThresh)
    maIsGettingLower = isGettingLower(row['SLOPE-OSC'], maSlopeThresh)
    maIsGettingHigher = isGettingHigher(row['SLOPE-OSC'], maSlopeThresh)
    
    fastMACrossedOverSlow = row['MA-FAST'] >= row['ma20']
    fastMACrossedUnderSlow = row['MA-FAST'] <= row['ma20']
    
    if 'OBV-OSC-PCT-CHNG' in row:
        obvIsNotChanging = isNotChanging(row['OBV-OSC-PCT-CHNG'], obvOscSlopeThresh)
        obvIsGettingLower = isGettingLower(row['OBV-OSC-PCT-CHNG'], obvOscSlopeThresh)
        obvIsGettingHigher = isGettingHigher(row['OBV-OSC-PCT-CHNG'], obvOscSlopeThresh)
    else:
        obvIsNotChanging = True
        obvIsGettingLower = False
        obvIsGettingHigher = False
            
    #This ticker is trending, lets see if its time to exit
    trend = getTickerTrend(row.symbol)
    if trend == 1:
        if fastMACrossedUnderSlow and \
            (adxIsGettingLower or adxIsNotChanging) and \
            (maIsGettingLower or maIsNotChanging) and \
            (obvIsGettingLower or obvIsNotChanging):
            s = 0
    elif trend == -1:   
        if fastMACrossedOverSlow and \
            (adxIsGettingLower or adxIsNotChanging) and \
            (maIsGettingHigher or maIsNotChanging) and \
            (obvIsGettingHigher or obvIsNotChanging):
            s = 0
    else:
        logging.error("Wierd ! trend should always be 1 or -1")
        return signal
    
    if signalChanged(s,signal):
        setTickerTrend(row.symbol, 0)
        logSignal('EXT-TRND',["obvData","adxData","maSlpData"],signal,s,row,type,isLastRow)
        #logString = f"{row.symbol}:{row.i}:{{row.name if isLastRow else ''}}  => EXIT TREND({trend}) on fastMA crossover ADX:{row['ADX']} > {adxThresh} AND OBV:{row['OBV-OSC-PCT-CHNG']} > {obvOscThresh} AND MA20:{row['SLOPE-OSC']} > {maSlopeThresh}"
    else:
        logSignal('CNT-TRND',["obvData","adxData","maSlpData"],signal,s,row,type,isLastRow," cxOver:{fastMACrossedOverSlow} cxUndr:{fastMACrossedUnderSlow} ",logWithNoSignalChange=True)
        #logString = f"{row.symbol}:{row.i}:{{row.name if isLastRow else ''}}  => DONT EXIT TREND YET ({trend}) cxOver:{fastMACrossedOverSlow} cxUndr:{fastMACrossedUnderSlow} ADX:{row['ADX']} > {adxThresh} AND OBV:{row['OBV-OSC-PCT-CHNG']} > {obvOscThresh} AND MA20:{row['SLOPE-OSC']} > {maSlopeThresh} "
    # if isLastRow:
    #     logging.info(logString)
    # else:
    #     logging.debug(logString)
    
    return s

def followTrendReversal (type, signal, isLastRow, row, df, 
                        last_signal=float('nan')):
    # SLOPE-OSC below yellow threshold, but coming up fast (REVERSALy)
    # and obv above yellow threshold, adx above yellow
    
    s = signal
    
    maSlopeIsLow = isAlmostLow(row['SLOPE-OSC'], maSlopeThresh, maSlopeThreshYellowMultiplier)
    maSlopeIsGettingHigher = isGettingHigher(row['SLOPE-OSC'], maSlopeThresh)
    slopeOscIsWillCrossOverLowThreshold = projectedValueBreachedThreshold \
        (row['SLOPE-OSC'], -maSlopeThresh, maSlopeThreshYellowMultiplier,
         row['SLOPE-OSC-SLOPE'], \
            maSlopeSlopeThresh, 'H')
    slopeHasReversedUp = maSlopeIsLow and maSlopeIsGettingHigher and slopeOscIsWillCrossOverLowThreshold
        
    maSlopeIsHigh = isAlmostHigh(row['SLOPE-OSC'], maSlopeThresh, maSlopeThreshYellowMultiplier)
    maSlopeIsGettingLower = isGettingLower(row['SLOPE-OSC'], maSlopeThresh)
    slopeOscIsWillCrossUnderHighThreshold = projectedValueBreachedThreshold \
        (row['SLOPE-OSC'], -maSlopeThresh, maSlopeThreshYellowMultiplier,
         row['SLOPE-OSC-SLOPE'], \
            maSlopeSlopeThresh, 'L')
    slopeHasReversedDn = maSlopeIsHigh and maSlopeIsGettingLower and slopeOscIsWillCrossUnderHighThreshold
    
    adxAboveYellow = row['ADX'] > (adxThresh*adxThreshYellowMultiplier)
    
    obvBreached = valueOrProjectedValueBreachedThreshold(row['OBV-OSC'],obvOscThresh,
                    row['OBV-OSC-PCT-CHNG'],obvOscSlopeThresh, 
                    obvOscThreshYellowMultiplier, "H_OR_L")

    if slopeHasReversedUp and adxAboveYellow and (obvBreached == 'H'):
        s = 1
    elif slopeHasReversedDn and adxAboveYellow and (obvBreached == 'L'):
        s = -1
    if signalChanged(s,signal):
        #print("entering trend following", row.i)
        setTickerTrend(row.symbol, s)
        if isLastRow:
            logging.info(f"{row.symbol}:{row.i} => FOLLOW TREND-REVERSAL => Extreme ADX/OBV/MA20 OVERRIDE signal ({signal})  s={s} / last_signal ({last_signal}) TO FOLLOW TREND.ADX:{row['ADX']} > {adxThresh} AND OBV:{obvOsc} > {obvOscThresh} AND MA20:{row['SLOPE-OSC']} > {maSlopeThresh}")
        else:
            logging.debug(f"{row.symbol}:{row.i}:{row.name}  => FOLLOW TREND-REVERSAL => Extreme ADX/OBV/MA20 OVERRIDE signal ({signal})  s={s} / last_signal ({last_signal}) TO FOLLOW TREND.ADX:{row['ADX']} > {adxThresh} AND OBV:{obvOsc} > {obvOscThresh} AND MA20:{row['SLOPE-OSC']} > {maSlopeThresh}")
    return s 

######### END OF SIGNAL GENERATION FUNCTIONS #########
def getOverrideSignal(row,ovSignalGenerators, df):
    
    s = row['signal'] #the signal that non-override sig generators generated
    #Could be nan, diff rom getLastSignal, which returns last valid, non-nan,
    #signal in df
    
    #Return nan if its not within trading hours
    if(row.name >= startTime) & \
        (row.name.hour >= startHour):
            
            if row.name.hour < endHour:
                type = 1 # Entry or Exit
            elif row.name.hour < exitHour:
                # Last time period before intraday exit; only exit positions
                # No new psitions will be entered
                type = 0 
            else:
                return s # Outside of trading hours
            isLastRow = row.name == df.index[-1]
            last_signal = getLastSignal(df['signal'])
            for sigGen in ovSignalGenerators:
                    #Note signal last_signal is a snapshot from the last traversal of entire DF
                    #s is initialized to last_signal, but is updated by each signal generator
                    #and passed in.  So two signals are passed into sigGen, 
                    # signal  (updated by sigGen) is the signal for *THIS* row, 
                    # and last_signal is the last non nan signal from prev rows
                    # calculated from last traversal of entire DF by signalGenerators
                    s = sigGen(type,s, isLastRow, row, df, last_signal)                          
    else:
        return s#Should Never happen
          
    return s    
    
def getSignal(row,signalGenerators, df):
    s = float("nan")
    isLastRow = row.name == df.index[-1]

    #Return nan if its not within trading hours
    if(row.name >= startTime) & \
        (row.name.hour >= startHour):
            if row.name.hour < endHour:
                type = 1 # Entry or Exit
            elif row.name.hour < exitHour:
                # Last time period before intraday exit; only exit positions
                # No new psitions will be entered
                type = 0 
            else:
                return 0 # Outside of trading hours EXIT ALL POSITIONS

            for sigGen in signalGenerators:
                # these functions can get the signal for *THIS* row, based on the
                # what signal Generators previous to this have done
                # they cannot get or act on signals generated in previous rows
                # signal s from previous signal generations is passed in as an 
                # argument

                s = sigGen(type,s, isLastRow, row, df)
            setTickerPosition(row.symbol, s)
    else:
        return s#Should Never happen
          
    return s
## MAIN APPLY STRATEGY FUNCTION
def applyIntraDayStrategy(df,analyticsGenerators=[populateBB], signalGenerators=[getSig_BB_CX],
                        overrideSignalGenerators=[]):
    global startTime
    if startTime == 0:
        startTime = datetime.datetime(2000,1,1,10,0,0) #Long ago :-)
        startTime = ist.localize(startTime)
    df['signal'] = float("nan")
    
    for analGen in analyticsGenerators:
        analGen(df)
    
    initTickerStatus(df['symbol'][0])
    # apply the condition function to each row of the DataFrame
    # these functions can get the signal for *THIS* row, based on the
    # what signal Generators previous to this have done
    # they cannot get or act on signals generated in previous rows
    df['signal'] = df.apply(getSignal, 
        args=(signalGenerators, df), axis=1)
    
    #Override signals if any:
    #Override signal generators are similar to other signal Generators
    #The only difference is that they are run on a second traversal of the dataframe
    #therefore the df passed to them is updated from previous traversal signals.
    #Unlike signalGenerators that only get signals generated in *THIS* row,
    #overrideSignalGenerators can get signals from previous rows as well 
    #
    # NOTE: the last_signal we get from previous rows, will only include 
    # the signals generated by sigGenerators, and WILL NOT include signals
    # generated by overrideSignalGenerators.  There is no way to get the
    # signals generated by overrideSignalGenerators in previous rows
    #
    #Avoid using these, unless you really need the last signal from previous rows
    #in the current row, as second traversal will slow the entire app down
    if len(overrideSignalGenerators):
        df['signal'] = df.apply(getOverrideSignal, 
            args=(overrideSignalGenerators, df), axis=1)
    startTime = startTime

