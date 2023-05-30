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
import tickerCfg
import utils
import random
import talib as ta
from itertools import compress
from candlerankings import candle_rankings
from signal_generators.svb import svb
from signal_generators.renko import Renko

signalGenerator = Renko(limitExitOrders=False,limitEntryOrders=False,
                        slEntryOrders=True,slExitOrders=True,
                        exitStaticBricks=False,useSVPForEntryExitPrices=False)

# signalGenerator = svb()
#cfg has all the config parameters make them all globals here
import cfg
globals().update(vars(cfg))

# set timezone to IST
ist = pytz.timezone('Asia/Kolkata')

tickerStatus = {}

def updateCFG(ma_slope_thresh, ma_slope_thresh_yellow_multiplier, \
                         obv_osc_thresh, \
                         obv_osc_thresh_yellow_multiplier, \
                         obv_ma_len, ma_slope_period, 
                         RenkoBrickMultiplier, atrlen \
                             , RenkoBrickMultiplierLongTarget, RenkoBrickMultiplierLongSL, \
                                 RenkoBrickMultiplierShortTarget, RenkoBrickMultiplierShortSL):
    global maSlopeThresh,maSlopeThreshYellowMultiplier, \
        obvOscThresh, obvOscThreshYellowMultiplier, overrideMultiplier, \
        cfgObvMaLen, cfgMASlopePeriods, cfgRenkoBrickMultiplier, atrLen \
        , cfgRenkoBrickMultiplierLongTarget, cfgRenkoBrickMultiplierLongSL \
        , cfgRenkoBrickMultiplierShortTarget, cfgRenkoBrickMultiplierShortSL
    maSlopeThresh = ma_slope_thresh
    maSlopeThreshYellowMultiplier = ma_slope_thresh_yellow_multiplier
    obvOscThresh =  obv_osc_thresh
    obvOscThreshYellowMultiplier = obv_osc_thresh_yellow_multiplier
    cfgObvMaLen = obv_ma_len
    cfgMASlopePeriods = ma_slope_period
    cfgRenkoBrickMultiplier = RenkoBrickMultiplier
    atrLen = atrlen
    cfgRenkoBrickMultiplierLongTarget = RenkoBrickMultiplierLongTarget
    cfgRenkoBrickMultiplierLongSL = RenkoBrickMultiplierLongSL
    cfgRenkoBrickMultiplierShortTarget = RenkoBrickMultiplierShortTarget
    cfgRenkoBrickMultiplierShortSL = RenkoBrickMultiplierShortSL
    
def updateCFG2(cfg={}):
    for key, value in cfg.items():
        globals()[key] = value

def applyTickerSpecificCfg(ticker):
    tCfg = utils.getTickerCfg(ticker)    
    for key, value in tCfg.items():
        globals()[key] = value
        #print(f"setting {key} to {value}")
        
def printCFG():
    print(f"\tmaLen: {maLen}")
    print(f"\tbandWidth: {bandWidth}")
    print(f"\tfastMALen: {fastMALen}")
    print(f"\tatrLen: {atrLen}")
    print(f"\tadxLen: {adxLen}")
    print(f"\tadxThresh: {adxThresh}")
    print(f"\tadxThreshYellowMultiplier: {adxThreshYellowMultiplier}")
    print(f"\tnumCandlesForSlopeProjection: {numCandlesForSlopeProjection}")
    print(f"\tmaSlopeThresh: {maSlopeThresh}")
    print(f"\tmaSlopeThreshYellowMultiplier: {maSlopeThreshYellowMultiplier}")
    print(f"\tobvOscThresh: {obvOscThresh}")
    print(f"\tobvOscThreshYellowMultiplier: {obvOscThreshYellowMultiplier}")
    print(f"\tcfgObvMaLen: {cfgObvMaLen}")
    print(f"\tbet_size: {bet_size}")

def candleStickPatterns(df):
    candle_names = ta.get_function_groups()['Pattern Recognition']
    avVol = df.Volume.mean()
    for candle in candle_names:
        # below is same as;
        # df["CDL3LINESTRIKE"] = talib.CDL3LINESTRIKE(op, hi, lo, cl)
        df[candle] = getattr(ta, candle)(df['Open'], df['High'], df['Low'], df['Adj Close'])

    df['candlestick_pattern'] = np.nan
    df['candlestick_match_count'] = np.nan
    df['candlestick_signal'] = np.nan
    for index, row in df.iterrows():

        # no pattern found
        if row.Volume < 1.2*avVol:
            df.loc[index,'candlestick_pattern'] = "NO_VOLUME"
        elif len(row[candle_names]) - sum(row[candle_names] == 0) == 0:
            df.loc[index,'candlestick_pattern'] = "NO_PATTERN"
            df.loc[index, 'candlestick_match_count'] = 0
        # single pattern found
        elif len(row[candle_names]) - sum(row[candle_names] == 0) == 1:
            # bull pattern 100 or 200
            if any(row[candle_names].values > 0):
                pattern = list(compress(row[candle_names].keys(), row[candle_names].values != 0))[0] + '_Bull'
                df.loc[index, 'candlestick_pattern'] = pattern
                df.loc[index, 'candlestick_match_count'] = 1
                df.loc[index, 'candlestick_signal'] = 1
            # bear pattern -100 or -200
            else:
                pattern = list(compress(row[candle_names].keys(), row[candle_names].values != 0))[0] + '_Bear'
                df.loc[index, 'candlestick_pattern'] = pattern
                df.loc[index, 'candlestick_match_count'] = 1
                df.loc[index, 'candlestick_signal'] = -1

        # multiple patterns matched -- select best performance
        else:
            # filter out pattern names from bool list of values
            patterns = list(compress(row[candle_names].keys(), row[candle_names].values != 0))
            container = []
            for pattern in patterns:
                if row[pattern] > 0:
                    container.append(pattern + '_Bull')
                    df.loc[index, 'candlestick_signal'] = 1
                else:
                    container.append(pattern + '_Bear')
                    df.loc[index, 'candlestick_signal'] = -1
            rank_list = [candle_rankings[p] for p in container]
            if len(rank_list) == len(container):
                rank_index_best = rank_list.index(min(rank_list))
                df.loc[index, 'candlestick_pattern'] = container[rank_index_best]
                df.loc[index, 'candlestick_match_count'] = len(container)
    # clean up candle columns
    df.drop(candle_names, axis = 1, inplace = True)

    # hanging_man = ta.CDLHANGINGMAN(df['Open'], df['High'], df['Low'], df['Adj Close'])
    # return (hanging_man)
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
    obv = ta.OBV(df['Adj Close'], df['Volume'])
    obv_pct_chang = obv.pct_change(periods=cfgObvLen).clip(-.1, .1)
    obv_osc = obv/obv.mean() - 1
    obv_osc_pct_chang = obv_osc.diff(cfgObvLen)/cfgObvLen
    return (obv_osc, obv_osc_pct_chang, obv, obv_pct_chang)
    # calculate the OBV column
    df['change'] = df['Adj Close'] - df['Open']
    df['direction'] = df['change'].apply(lambda x: 1 if x > 0 else -1 if x < 0 else 0)
    df['obv'] = df['direction'] * df['Volume']
    df['obv'] = df['obv'].rolling(window=cfgObvLen, min_periods=cfgObvLen).sum() # instead of cumsum; this restricts it to historical candles spec in cfg
    df['ma_obv'] = df['obv'].rolling(window=cfgObvMaLen, min_periods=2).mean()
    df['ma_obv_diff'] = df['ma_obv'].diff(5)
    
    #OBV-Diff Max/Min diff should only look at previous candles, not future candles
    #Also restrict the lookback to cfgMaxLookbackCandles, to keep backtest results consistent
    #apples to apples with live trading
    
    df['ma_obv_diff_max'] = df['ma_obv_diff'].rolling(window=cfgObvLen, min_periods=cfgObvLen).max()
    df['ma_obv_diff_min'] = df['ma_obv_diff'].rolling(window=cfgObvLen, min_periods=cfgObvLen).min()
    df['obv_osc'] = df['ma_obv_diff'] / (df['ma_obv_diff_max'] - df['ma_obv_diff_min'])
    df['obv_osc_pct_change'] = df['obv_osc'].diff(2)/2
    df['obv_trend'] = np.where(df['obv_osc'] > obvOscThresh,1,0)
    df['obv_trend'] = np.where(df['obv_osc'] < -obvOscThresh,-1,df['obv_trend'])
    
    # CLIP extreme
    df['obv_osc'] = df['obv_osc'].clip(lower=-1, upper=1)
    # df.to_csv("obv1.csv")
    # exit(0)
    return (df['ma_obv'],df['obv_osc'],df['obv_trend'],df['obv_osc_pct_change'])
def renko(DF):

    from stocktrends import Renko
    
    "function to convert ohlc data into renko bricks"
    df = DF.copy()
    df.reset_index(inplace=True)

    df.columns = ["date","i","open","high","low","close","volume","symbol","signal","ATR"]
    
    df2 = Renko(df)
    df2.brick_size = cfgRenkoBrickSize
    renko_df = df2.get_ohlc_data() #if using older version of the library please use get_bricks() instead
    renko_df["bar_num"] = np.where(renko_df["uptrend"]==True,1,np.where(renko_df["uptrend"]==False,-1,0))

    for i in range(1,len(renko_df["bar_num"])):
        if renko_df["bar_num"][i]>0 and renko_df["bar_num"][i-1]>0:
            renko_df["bar_num"][i]+=renko_df["bar_num"][i-1]
        elif renko_df["bar_num"][i]<0 and renko_df["bar_num"][i-1]<0:
            renko_df["bar_num"][i]+=renko_df["bar_num"][i-1]
    renko_df.drop_duplicates(subset="date",keep="last",inplace=True)
    renko_df["brick_size"] = cfgRenkoBrickSize
    
    renko_df.to_csv("renko.csv")
    # change leading uptrent to nan until a real trend is found
    mask = renko_df['bar_num'] == 1
    # Find the index where bar_num changes to a value other than 1
    change_idx = np.argmax(renko_df['bar_num'].values != 1)
    # Replace the initial rows with NaN values
    renko_df.loc[mask & (renko_df.index < change_idx), ['bar_num', 'uptrend']] = np.nan
    
    # Find the index where the first non-NaN negative integer appears in renko_brick_num
    first_negative_idx = renko_df[renko_df['bar_num'].notnull() & (renko_df['bar_num'] < 0)].index.min()
    # Reduce non-NaN leading positive integer values of renko_brick_num by 1
    mask = (renko_df.index < first_negative_idx) & renko_df['bar_num'].notnull() & (renko_df['bar_num'] > 0) if not np.isnan(first_negative_idx) else renko_df['bar_num'].notnull()
    renko_df.loc[mask, 'bar_num'] -= 1
    
    # renko_df['uptrend_change'] = renko_df['uptrend'].shift(-1) != renko_df['uptrend']
    # max_min_values = renko_df.loc[renko_df['uptrend_change'], 'bar_num'].tolist()

    # # The first element is not valid because it compares with NaN
    # max_min_values = max_min_values[1:]

    # print(max_min_values)
    # print(renko_df)

    return renko_df

def vwap(df):
    df['VWAP'] = np.cumsum(df['Adj Close'] * df['Volume']) / np.cumsum(df['Volume'])

    # Calculate Standard Deviation (SD) bands
    num_periods = 20  # Number of periods for SD calculation
    df['VWAP_SD'] = df['VWAP'].rolling(num_periods).std()  # Rolling standard deviation
    df['VWAP_upper'] = df['VWAP'] + 1 * df['VWAP_SD']  # Upper band (2 SD above VWAP)
    df['VWAP_lower'] = df['VWAP'] - 1 * df['VWAP_SD']  # Lower band (2 SD below VWAP)
    
# Function to calculate POC, VAH, VAL
def session_volume_profile (DF, type='normal'):
    # calculate volume distribution
    if DF.empty:
        return (np.nan, np.nan, np.nan, np.nan, np.nan)
    df = DF[DF.index.time > datetime.time(9, 31)].copy() # remove pre-market candles as they are huge volume (OI growth) and therefore have an undue influence on SVP
    if df.empty:
        return (np.nan, np.nan, np.nan, np.nan, np.nan)

    df['svp_price'] = df['Adj Close']   
    volume_distribution = df.groupby('svp_price').sum()['Volume']
    # POC: price with max volume
    poc = volume_distribution.idxmax()
    
    # VAH, VAL: prices at 70% of volume
    volume_sorted = volume_distribution.sort_index()
    volume_cumsum = volume_sorted.cumsum()
    volume_total = volume_cumsum.iloc[-1]
    (t1,t2) = (0.15,0.85) if type == 'normal' else (0.1,0.9)
    val = volume_sorted[volume_cumsum <= volume_total * 0.15].index.max()  # 15% from top
    val = volume_sorted.index[0] if np.isnan(val) else val
    vah = volume_sorted[volume_cumsum >= volume_total * 0.85].index.min()  # 15% from bottom
    vah = volume_sorted.index[-1] if np.isnan(vah) else vah

    poc_val = volume_sorted[volume_cumsum <= volume_total * 0.495].index.max()  # 49% from top
    poc_vah = volume_sorted[volume_cumsum >= volume_total * 0.495].index.min()  # 49% from bottom
    poc = (poc_val + poc_vah)/2
    
    dayHigh = volume_sorted.index.max()
    dayLow  = volume_sorted.index.min()
    return (poc, vah, val,dayHigh,dayLow)

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
    df['ATR'] = df['TR'].ewm(com=n,min_periods=cfgMinCandlesForMA).mean()
    return df['ATR']

def supertrend(df, multiplier=3, atr_period=10):
    high = df['High']
    low = df['Low']
    close = df['Adj Close']
    atr = df['ATR']
    # HL2 is simply the average of high and low prices
    hl2 = (high + low) / 2
    # upperband and lowerband calculation
    # notice that final bands are set to be equal to the respective bands
    final_upperband = upperband = hl2 + (multiplier * atr)
    final_lowerband = lowerband = hl2 - (multiplier * atr)
    # initialize Supertrend column to True
    supertrend = [True] * len(df)
    supertrend_signal = np.full(len(df), np.nan)
    
    for i in range(1, len(df.index)):
        curr, prev = i, i-1
                # if current close price crosses above upperband
        if close[curr] > final_upperband[prev]:
            supertrend[curr] = True
        # if current close price crosses below lowerband
        elif close[curr] < final_lowerband[prev]:
            supertrend[curr] = False
        # else, the trend continues
        else:
            supertrend[curr] = supertrend[prev]
            # adjustment to the final bands
            if supertrend[curr] == True and final_lowerband[curr] < final_lowerband[prev]:
                final_lowerband[curr] = final_lowerband[prev]
            if supertrend[curr] == False and final_upperband[curr] > final_upperband[prev]:
                final_upperband[curr] = final_upperband[prev]

        # to remove bands according to the trend direction
        if supertrend[curr] == True:
            final_upperband[curr] = np.nan
        else:
            final_lowerband[curr] = np.nan
        
        if supertrend[curr] and (not supertrend[prev]):
            supertrend_signal[curr] = 1
        elif (not supertrend[curr]) and supertrend[prev]:
            supertrend_signal[curr] = -1 
    return supertrend_signal,supertrend, final_upperband,final_lowerband


def ADX(DF, n=adxLen):
    adx = ta.ADX(DF['High'], DF['Low'], DF['Adj Close'], timeperiod=n)
    adx_pct_change = adx.diff(2)/2
    return (adx,adx_pct_change)

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
    df['ma_superTrend'] = df['Adj Close'].ewm(com=superLen, min_periods=cfgMinCandlesForMA).mean()
    df['ma_superTrend_pct_change'] = 10000*df['ma_superTrend'].pct_change(periods=3)
    df['ma20'] = df['Adj Close'].rolling(window=maLen).mean()
    df['MA-FAST'] = df['Adj Close'].rolling(window=fastMALen).mean()
    df['MA-FAST-SLP'] = df['MA-FAST'].pct_change(periods=3)
    df['MA-FAST-SLP'] = df['MA-FAST-SLP'].clip(lower=-0.1, upper=0.1)
    df['ma20_pct_change'] = df['ma20'].pct_change(periods=cfgMASlopePeriods)
    df['ma20_pct_change_ma'] = df['ma20_pct_change'].ewm(com=5, min_periods=1).mean()
    df['ma20_pct_change_ma_sq'] = df['ma20_pct_change_ma'].pct_change()
    # df['ma20_pct_change_ma_sq'] = df['ma20_pct_change_ma_sq'].ewm(com=maLen, min_periods=maLen).mean()
    # df['ma20_pct_change_ma_sq'] = df['ma20_pct_change_ma_sq'].clip(lower=-0.5, upper=0.5)
    df['std'] = df['Adj Close'].rolling(window=maLen).std()
    df['upper_band'] = df['ma20'] + (bandWidth * df['std'])
    df['lower_band'] = df['ma20'] - (bandWidth * df['std'])
    df['mini_upper_band'] = df['ma20'] + (cfgMiniBandWidthMult*bandWidth * df['std'])
    df['mini_lower_band'] = df['ma20'] - (cfgMiniBandWidthMult*bandWidth * df['std'])
    df['super_upper_band'] = df['ma20'] + (cfgSuperBandWidthMult*bandWidth * df['std'])
    df['super_lower_band'] = df['ma20'] - (cfgSuperBandWidthMult*bandWidth * df['std'])
    #df.drop(['Open','High','Low'],axis=1,inplace=True,errors='ignore')
    #df.tail(5)
    slopeStdDev = df['ma20_pct_change_ma'].rolling(window=cfgMaxLookbackCandles,min_periods=maLen).std()
    slopeMean = df['ma20_pct_change_ma'].rolling(window=cfgMaxLookbackCandles,min_periods=maLen).mean()
    # df['SLOPE-OSC'] = (df['ma20_pct_change_ma'] - slopeMean)/slopeStdDev
    # df['SLOPE-OSC-SLOPE'] = df['SLOPE-OSC'].diff(2)/2
    
    df['SLOPE-OSC'] = df['ma20_pct_change']
    df['SLOPE-OSC-SLOPE'] = df['ma20_pct_change_ma_sq']
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
def adxIsBullish(row):
    return row['+di'] >= row['-di']
def adxIsBearish(row):
    return row['-di'] >= row['+di']
def adxIsHigh(row):
    return valueAndProjectedValueBreachedThreshold(row['ADX'],adxThresh,
                row['ADX-PCT-CHNG'], adxThreshYellowMultiplier,'H')
    return True if row['ADX'] >= adxThresh else False
def adxIsLow(row):
    return row['ADX'] <= adxThresh*cfgAdxThreshExitMultiplier
def adxIsHighAndBullish(row):
    return adxIsHigh(row) and adxIsBullish(row)
def adxIsHighAndBearish(row):
    return adxIsHigh(row) and adxIsBearish(row)
def maSteepSlopeUp(row):
    return row['ma20_pct_change']>=maSlopeThresh
def maSteepSlopeDn(row):
    return row['ma20_pct_change']<=-maSlopeThresh
def maSlopeFlattish(row):
    return valueOrProjectedValueBreachedThreshold(row['SLOPE-OSC'], \
                maSlopeThresh,row['SLOPE-OSC-SLOPE'], maSlopeThreshYellowMultiplier, "H_OR_L") \
            == False
def maDivergesFromADX(row):
    return (adxIsBullish(row) and row['ma20_pct_change'] < 0) \
        or (adxIsBearish(row) and row['ma20_pct_change'] > 0)
def maDivergesFromTrend(row):
    trend = getTickerTrend(row['symbol'])
    if trend == 1 and row['ma20_pct_change'] < 0:
        return True
    elif trend == -1 and row['ma20_pct_change'] > 0:
        return True
    return False
def obvIsBullish(row):
    return row['OBV-PCT-CHNG'] >= cfgObvSlopeThresh 
    return valueOrProjectedValueBreachedThreshold(row['OBV-OSC'],obvOscThresh,
                row['OBV-OSC-PCT-CHNG'],obvOscThreshYellowMultiplier, "H_OR_L") \
            == 'H'
def obvIsBearish(row):
    return row['OBV-PCT-CHNG'] <= -cfgObvSlopeThresh 
    return valueOrProjectedValueBreachedThreshold(row['OBV-OSC'],obvOscThresh,
                row['OBV-OSC-PCT-CHNG'],obvOscThreshYellowMultiplier, "H_OR_L") \
            == 'L'
def obvNoLongerHigh(row):
    obv = row['OBV-OSC']
    slp = row['OBV-OSC-PCT-CHNG']
    projObv = projectedValue(obv,slp)
    return obv < obvOscThresh or projObv < obvOscThresh
def obvNoLongerLow(row):
    obv = row['OBV-OSC']
    slp = row['OBV-OSC-PCT-CHNG']
    projObv = projectedValue(obv,slp)
    return obv > -obvOscThresh or projObv > -obvOscThresh
def valueBreachedThreshold (value,threshold,type='H'):
    if type == 'H':
        return value >= threshold
    elif type == 'L':
        return value <= -threshold
    elif type == 'H_OR_L':
        if value >= threshold:
            return 'H'
        elif value <= -threshold:
            return 'L'
    else:
        logging.warning('valueBreachedThreshold: invalid type')
    return False
def getSVP(row,project=False):
    (vah,poc,val,slpVah,slpVal,slpPoc) = row[['vah','poc','val','slpVah','slpVal','slpPoc']]

    if project:
        (vah,poc,val) = getProjectedSVP(row)
    else:
        vah = vah + 10 if slpVah > 0.1 else vah
        val = val - 10 if slpVal < -0.1 else val
    return (vah,poc,val)
def getProjectedSVP(row):
    (vah,poc,val,slpVah,slpVal,slpPoc) = row[['vah','poc','val','slpVah','slpVal','slpPoc']]
    vah = vah + (slpVah*cfgSVPSlopeProjectionCandles)
    val = val + (slpVal*cfgSVPSlopeProjectionCandles)
    poc = poc + (slpPoc*cfgSVPSlopeProjectionCandles)
    return(vah,poc,val)
def getSVPquadrant(row):
    close = row['Adj Close']
    (vah,poc,val) = getSVP(row)
    if close > vah:
        q = 'High'
    elif close > poc:
        q = 'Upper'
    elif close > val:
        q = 'Lower'
    else:
        q = 'Low'
    return q
         
def longResistance(row):
    close = row['Adj Close']
    if row.i < 100: 
        return close+100 # suppor/resistance bands not formed before 100 candles
    elif row.slpPoc <= -cfgSVPSlopeThreshold:
        return row.poc
    else:
        return max(close,row.vah) if row.slpVah <= cfgSVPSlopeThreshold else close+8
    
    (vah,poc,val) = getSVP(row)
    status = getSVPquadrant(row)
    
    if status == 'High':
        r = close+1
    elif status == 'Upper':
        r = vah
    elif status == 'Lower':
        r = vah
    else:
        r = val
    return r
def longSupport(row):
    close = row['Adj Close']
    if row.i < 100: 
        return close-100 # suppor/resistance bands not formed before 100 candles
    elif row.slpPoc >= cfgSVPSlopeThreshold:
        return row.poc
    else:
        return min(close,row.val) if row.slpVal >= -cfgSVPSlopeThreshold else row.val-8
    (vah,poc,val) = getProjectedSVP(row)
    status = getSVPquadrant(row)
    
    if status == 'High':
        r = vah
    elif status == 'Upper':
        r = poc
    elif status == 'Lower':
        r = val
    else:
        r = close-1
    return r
def shortSupport(row):
    return longResistance(row)
        
def shortResistance(row):
    return longSupport(row)

def svpTrendsDown(row):
    return row['slpPoc'] <= -cfgSVPSlopeThreshold
def svpTrendsUp(row):
    return row['slpPoc'] >= cfgSVPSlopeThreshold
    
def OLD_valueBreachedThreshold (value,threshold,slope,
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
                                    multiplier,type):
    if valueBreachedThreshold(value,threshold*multiplier,type) == False:
        return False # we only project slope if at least close to threshold  
    value = projectedValue(value,slope)
    return valueBreachedThreshold(value,threshold,type)

def valueOrProjectedValueBreachedThreshold(value,threshold,slope,
                                            multiplier,type):
    p = valueBreachedThreshold(value,threshold,type)
    if p == False:
        return projectedValueBreachedThreshold(value,threshold,slope,multiplier,type)
    else:
        return p
def valueAndProjectedValueBreachedThreshold(value,threshold,slope,
                                            multiplier,type):
    p = valueBreachedThreshold(value,threshold,type)
    if p == False:
        return p
    else:
        type = p if (p == 'H' or p == 'L') else type
        return p if projectedValueBreachedThreshold(value,threshold,slope,multiplier,type) else False
def projectedValue(value,slope):
    #Calc projectedValue is where the value will be after {candlesToProject}
    #if it starts at value and continues at current slope levels

    for counter in range(numCandlesForSlopeProjection):
        #print(f"value is {value} and threshold is {threshold} and slope is {slope}")
        #logging.debug(f"counter is {counter} and value is {value} and slope is {slope} and threshold is {threshold}")
        value = (value + slope) # calculate new value of x after one candle
    return value
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
    
    
def crossOver(fast,slow,lastFast=None,lastSlow=None):
    return fast > slow and \
        (lastFast is None or (lastFast <= lastSlow))
def crossUnder(fast,slow,lastFast=None,lastSlow=None):
    return fast < slow and \
        (lastFast is None or (lastFast >= lastSlow))

def getLastSignal(signal):
    last_signal_index = signal.last_valid_index()
    if(last_signal_index):
        last_signal = signal.loc[last_signal_index]
    else:
        last_signal = 0
    return last_signal
def initTickerStatus(ticker):
#    if ticker not in tickerStatus.keys():
    tickerStatus[ticker] = {'position':float("nan"), 
                            'trendSignal':float("nan"),
                            'entry_price':float("nan"),
                            'limit1':float("nan"),
                            'limit2':float("nan"),
                            'sl1':float("nan"),
                            'sl2':float("nan")
                            }     
def getTickerPosition(ticker):
    if ticker not in tickerStatus:
        return float ('nan')
    elif 'position' not in tickerStatus[ticker]:
        return float ('nan')
    else:
        return tickerStatus[ticker]['position']
def tickerHasPosition(ticker):
    return (not np.isnan(getTickerPosition(ticker))
                and getTickerPosition(ticker) != 0)
def tickerHasLongPosition(ticker):
    return getTickerPosition(ticker) == 1
def tickerHasShortPosition(ticker):
    return getTickerPosition(ticker) == -1
def setTickerMaxPriceForTrade(ticker,entry_price,signal):
    default_min = 1000000
    default_max = 0
    if 'max_price' not in tickerStatus[ticker] or \
        signal == 0:
        tickerStatus[ticker]['max_price'] = default_max
        tickerStatus[ticker]['min_price'] = default_min
    if np.isnan(signal):
        if tickerStatus[ticker]['max_price'] != default_max:
            tickerStatus[ticker]['max_price'] = max(entry_price,tickerStatus[ticker]['max_price'])
        if tickerStatus[ticker]['min_price'] != default_min:
            tickerStatus[ticker]['min_price'] = min(entry_price,tickerStatus[ticker]['min_price'])
    elif signal == 1:
        tickerStatus[ticker]['max_price'] = max(entry_price,tickerStatus[ticker]['max_price'])
    elif signal == -1:
        tickerStatus[ticker]['min_price'] = min(entry_price,tickerStatus[ticker]['min_price'])

def setTickerPosition(ticker,signal, entry_price,limit1, limit2, sl1, sl2):
    # logging.info(f"setting ticker position for {ticker} to {signal} entry{entry_price} pos:{tickerStatus[ticker]['position']}")
    setTickerMaxPriceForTrade(ticker,entry_price,signal)
    (tickerStatus[ticker]['limit1'], tickerStatus[ticker]['limit2'], tickerStatus[ticker]['sl1'], tickerStatus[ticker]['sl2']) = (limit1, limit2, sl1, sl2)
    #logging.info(f"signal: {signal} entry: {tickerStatus[ticker]['entry_price']} max: {tickerStatus[ticker]['max_price']} min: {tickerStatus[ticker]['min_price']}")
    if np.isnan(signal):
        return # nan signal does not change position
    if ticker not in tickerStatus:
        tickerStatus[ticker] = {}
    if tickerStatus[ticker]['position'] != signal:
        tickerStatus[ticker]['position'] = signal
        if signal == 0:
            tickerStatus[ticker]['entry_price'] = float('nan')
        else:
            tickerStatus[ticker]['entry_price'] = entry_price
            # logging.info(f"setting entry price for {ticker} to {entry_price}")
def getTickerEntryPrice(ticker):
    return tickerStatus[ticker]['entry_price']
def getTickerLimitSLOderPrices(ticker):
    s = tickerStatus[ticker]
    return (abs(s['limit1']),abs(s['limit2']),abs(s['sl1']),abs(s['sl2']))
def getTickerLimit1(ticker):
    return abs(tickerStatus[ticker]['limit1'])
def getTickerLimit2(ticker):
    return abs(tickerStatus[ticker]['limit2'])
def getTickerSL1(ticker):
    return abs(tickerStatus[ticker]['sl1'])
def getTickerSL2(ticker):
    return abs(tickerStatus[ticker]['sl2'])
def isLongLimit1Order(ticker):
    return tickerStatus[ticker]['limit1'] > 0
def isShortLimit1Order(ticker):
    return tickerStatus[ticker]['limit1'] < 0
def isShortSL1Order(ticker):
    return tickerStatus[ticker]['sl1'] < 0
def isLongSL1Order(ticker):
    return tickerStatus[ticker]['sl1'] > 0
def enoughReturnForDay(ret):
    return ret >= cfgEnoughReturnForTheDay or \
                ret <= -cfgEnoughLossForTheDay
def getTickerRenkoTargetPrices(row):
    renkoBrickSize = row['renko_brick_high'] - row['renko_brick_low']
    longTarget = row['renko_brick_high']+(renkoBrickSize*cfgRenkoBrickMultiplierLongTarget)
    longSL = row['renko_brick_low']-(renkoBrickSize*cfgRenkoBrickMultiplierLongSL)
    shortTarget = row['renko_brick_low']-(renkoBrickSize*cfgRenkoBrickMultiplierShortTarget)
    shortSL = row['renko_brick_high']+(renkoBrickSize*cfgRenkoBrickMultiplierShortSL)
    return(longTarget,longSL,shortTarget,shortSL)

def getTickerTargetPrices(ticker):
    entryPrice = getTickerEntryPrice(ticker)
    longTarget = ((1+cfgTarget)*entryPrice)
    longSL = ((1-cfgStopLoss)*entryPrice)
    longSLFromPeak = ((1-cfgStopLossFromPeak)*tickerStatus[ticker]['max_price'])
    shortTarget = ((1-cfgTarget)*entryPrice)
    shortSL = ((1+cfgStopLoss)*entryPrice)
    shortSLFromPeak = ((1+cfgStopLossFromPeak)*tickerStatus[ticker]['min_price'])
    return(longTarget,max(longSL,longSLFromPeak),shortTarget,min(shortSL,shortSLFromPeak))
def logTickerStatus(ticker):
    if tickerStatus[ticker]['trendSignal'] == 1:
        logging.info(f"TickerStatus: {ticker} is trending up")
    elif tickerStatus[ticker]['trendSignal'] == -1:
        logging.info(f"TickerStatus: {ticker} is trending down")
    else:
        logging.info(f"TickerStatus: {ticker} is not trending")
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
    #print("setTickerTrend: ",ticker,trend)
    
def tickerIsTrending(ticker):
    t = getTickerTrend(ticker)
    return True if (t == 1 or t == -1) else False

def tickerIsTrendingUp(ticker):
    t = getTickerTrend(ticker)
    return True if (t == 1) else False

def tickerIsTrendingDn(ticker):
    t = getTickerTrend(ticker)
    return True if (t == -1) else False
def tickerPositionIsLong(ticker):
    t = getTickerPosition(ticker)
    return True if (t == 1) else False
def tickerPositionIsShort(ticker):
    t = getTickerPosition(ticker)
    return True if (t == -1) else False

def isRenkoUpTrend(row):
    return row['renko_uptrend']
    brickNum =row['renko_brick_num']
    if abs(brickNum) >= cfgRenkoNumBricksForTrend:
        return True if brickNum > 0 else False
    close = row['Adj Close']
    brickLow = row['renko_brick_low']
    brickHigh = row['renko_brick_high']
    uptrend = row['renko_uptrend']
    
    if uptrend and close > brickLow:
        return True
    elif uptrend == False and close < brickHigh:
        return False
    else:
        return False if uptrend else True 

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
    # return
    rowInfo = f'{row.symbol}:{row.i} '
    rowPrice = f'p:{round(row["Adj Close"],1)} '
    sigInfo = f'sig:{"-" if np.isnan(signal) else signal} s:{"-" if np.isnan(s) else s} {"E" if window == 1 else "X"} '
    dataStrings = {
        "ohlv": f"OHLV O:{round(row['Open'],1)} H:{round(row['High'],1)} L:{round(row['Low'],1)} | ",
        "adxData" : f"ADX:{round(row['ADX'],1)} > {adxThresh}(*{round(adxThreshYellowMultiplier,1)}) adxSLP:{round(row['ADX-PCT-CHNG'],2)}*{numCandlesForSlopeProjection} | " if "ADX" in row else 'No ADX Data',
        "maSlpData" : f"maSlp:{round(row['SLOPE-OSC'],2)} >= {maSlopeThresh}(*{maSlopeThreshYellowMultiplier}) maSlpChng:{round(row['SLOPE-OSC-SLOPE'],2)}>*{numCandlesForSlopeProjection} | " if "SLOPE-OSC" in row else 'No Slope Data',
        "obvData" : f"OBV:{round(row['OBV-OSC'],2)} > {obvOscThresh}(*{obvOscThreshYellowMultiplier}) obvSLP:{round(row['OBV-OSC-PCT-CHNG'],2)}>*{numCandlesForSlopeProjection} | " if "OBV-OSC" in row else 'No Volume Data',
        "RenkoData": f"Renko Trend:{'↑' if row['renko_uptrend'] else '↓'}:{round(row['renko_brick_num'])}({round(row['renko_brick_diff'])}) V:{round(row['renko_brick_volume_osc'],1)} StC:{round(row['renko_static_candles']) if not np.isnan(row['renko_static_candles']) else 'nan'} H:{round(row['renko_brick_high'],1)} L:{round(row['renko_brick_low'],1)} | " if "renko_uptrend" in row and not np.isnan(row.renko_brick_num) else 'No Renko Data',
        "svp": f"SVP {row['dayHigh']}-{row['vah']}({round(row['slpVah'],1)})-{row['poc']}({round(row['slpPoc'],1)})-{row['val']}({round(row['slpVal'],1)})-{row['dayLow']} VWAP:{row['VWAP']} SVP-15 {round(row['15mHigh'],1)}-{round(row['vah15m'],1)}-{round(row['poc15m'],1)}-{round(row['val15m'],1)}-{round(row['15mLow'],1)} | " if "poc" in row else 'No SVP Data'
    }
    dataString = ''
    for key in reqData:
        dataString = dataString+dataStrings[key]
    if isLastRow and (signalChanged(s,signal) or logWithNoSignalChange):
        logging.info(rowInfo+' => '+rowPrice+' => '+msg+' '+extra+' '+sigInfo+ ' => '+dataString)
    elif signalChanged(s,signal) or logWithNoSignalChange:
        rowTime = row.name.strftime("%d/%m %I:%M")
        rowInfo = rowInfo+f':{rowTime} ' #backtest needs date
        logging.debug(rowInfo+' => '+rowPrice+' => '+msg+' '+extra+' '+sigInfo+ ' => '+dataString)

def skipFilter (signal,type):
    # Since this is a FILTER, we only negate long and short signals
    # on extreme ADX, with MA SLOPE pointing the opposite direction of signal
    # for nan or 0, we just return the signal
    #
    # Only Exception is if it is exit time frame and signal is nan, in that
    # scenario, BB CX will not return a signal, so we need to protect against
    # extreme conditions using filters
    if isLongOrShortSignal(signal):
        return False
    if type == 0 and np.isnan(signal):
        return False
    return True

#####END OF UTILTIY FUNCTIONS#########

####### POPULATE FUNCTIONS #######
# Functions that populate the dataframe with the indicators
def populateBB (df):
    addBBStats(df)

def populateATR(df):
    df['ATR'] = ATR(df,atrLen)
    
def populateADX (df):
    (df['ADX'],df['ADX-PCT-CHNG']) = ADX(df,adxLen)

def populateOBV (df):
    # if (df['Volume'].max() == 0):
    #     return False # Index has no volume data so skip it
    (df['OBV-OSC'],df['OBV-OSC-PCT-CHNG'], df['OBV'], df['OBV-PCT-CHNG']) = OBV(df)
def populateSuperTrend (df):
    (df['SuperTrend'],df['SuperTrendDirection'],df['SuperTrendUpper'],df['SuperTrendLower']) = supertrend(df)

def populateCandleStickPatterns(df):
    #(df['HANGINGMAN']) = 
    candleStickPatterns(df)
def populateRenko(df):
    renkoDF = renko(df)
    renkoDF.columns = ["Date","open","renko_brick_high","renko_brick_low","close","uptrend","bar_num","brick_size"]
    df["Date"] = df.index
    # print(renkoDF)
    # exit(0)
    # print(f"Max: {renkoDF['bar_num'].max()} Min: {renkoDF['bar_num'].min()}")
    df_renko_ohlc = df.merge(renkoDF.loc[:,["Date","uptrend","renko_brick_high","renko_brick_low","bar_num","brick_size"]],how="outer",on="Date")
    df_renko_ohlc["uptrend"].fillna(method='ffill',inplace=True)
    df_renko_ohlc["bar_num"].fillna(method='ffill',inplace=True)
    df_renko_ohlc["renko_brick_high"].fillna(method='ffill',inplace=True)
    df_renko_ohlc["renko_brick_low"].fillna(method='ffill',inplace=True)
    df_renko_ohlc.set_index('Date', drop=True, inplace=True)
    df['renko_uptrend'] = df_renko_ohlc['uptrend']
    df['renko_brick_num'] = df_renko_ohlc['bar_num']
    
    diff = df['renko_brick_num'].diff()
    df['renko_brick_diff'] = diff

    df['unique_brick_id'] = np.where(df['renko_brick_diff'] != 0, 
                                    df.index.astype(str) + df['renko_brick_num'].astype(str), 
                                    np.nan)
    df['unique_brick_id'].fillna(method='ffill', inplace=True)  # Forward fill the NaN values

    df['renko_brick_volume'] = df.groupby('unique_brick_id')['Volume'].transform('sum')
    df['renko_brick_volume_osc'] = df['renko_brick_volume']/np.mean(df['renko_brick_volume'].unique())
    # df = df.drop(columns='unique_brick_id')
    # df['renko_brick_vol'] = df_renko_ohlc['brick_volume']

    # Calculate renko_static_candles
    df['renko_static_candles'] = (
        df.loc[df['renko_brick_diff'] == 0, 'renko_brick_diff']
        .groupby((df['renko_brick_diff'] != 0).cumsum())
        .cumcount()
        .where(df['renko_brick_diff'] == 0)
        .fillna(0)
    )
    df['renko_static_candles'].fillna(0)
    df['renko_static_candles'] = df['renko_static_candles'].where(df['renko_static_candles'] >= 1, 0)

    same_sign = np.sign(df['renko_brick_num']) == np.sign(df['renko_brick_num'].shift(1))
    renko_brick_diff = diff.where(same_sign, 0)
    df['renko_brick_diff'] = renko_brick_diff

    df['renko_brick_high'] = df_renko_ohlc['renko_brick_high']
    df['renko_brick_low'] = df_renko_ohlc['renko_brick_low']
    
    df.drop(["Date"],inplace=True,axis=1)
    df.to_csv('renko.csv')
def populateSVP(df):
    window_size = 15  # 15-minute window size in terms of number of rows
    df['poc15m'] = df['vah15m'] = df['val15m'] = df['poc'] = df['vah'] = df['val']  = df['slpPoc'] = df['slpVah'] = df['slpVal'] = df['slp15Poc'] = df['slp15Vah'] = df['slp15Val'] = df['dayHigh']  = df['dayLow'] = df['15mHigh']  = df['15mLow'] = np.nan

    for end in range(window_size, len(df)):
        start = end-window_size
        window_df = df.iloc[start:end]
        (poc, vah, val,dayHigh,dayLow) = session_volume_profile(window_df,type='normal')
        df.iloc[end, df.columns.get_loc('poc15m')] = poc
        df.iloc[end, df.columns.get_loc('vah15m')] = vah
        df.iloc[end, df.columns.get_loc('val15m')] = val
        df.iloc[end, df.columns.get_loc('15mHigh')] = dayHigh
        df.iloc[end, df.columns.get_loc('15mLow')] = dayLow
        df.iloc[end, df.columns.get_loc('slp15Poc')] = (poc - df.iloc[end-3, df.columns.get_loc('poc15m')])
        df.iloc[end, df.columns.get_loc('slp15Vah')] = (vah - df.iloc[end-3, df.columns.get_loc('vah15m')])
        df.iloc[end, df.columns.get_loc('slp15Val')] = (val - df.iloc[end-3, df.columns.get_loc('val15m')])

    for end in range(5, len(df)):
        start = 0
        window_df = df.iloc[start:end]
        (poc, vah, val,dayHigh,dayLow) = session_volume_profile(window_df)
        df.iloc[end, df.columns.get_loc('poc')] = poc
        df.iloc[end, df.columns.get_loc('vah')] = vah
        df.iloc[end, df.columns.get_loc('val')] = val        
        df.iloc[end, df.columns.get_loc('slpPoc')] = (poc - df.iloc[end-(2*cfgSVPSlopeCandles), df.columns.get_loc('poc')])/cfgSVPSlopeCandles
        df.iloc[end, df.columns.get_loc('slpVah')] = (vah - df.iloc[end-cfgSVPSlopeCandles, df.columns.get_loc('vah')])/cfgSVPSlopeCandles
        df.iloc[end, df.columns.get_loc('slpVal')] = (val - df.iloc[end-cfgSVPSlopeCandles, df.columns.get_loc('val')])/cfgSVPSlopeCandles
        df.iloc[end, df.columns.get_loc('dayHigh')] = dayHigh
        df.iloc[end, df.columns.get_loc('dayLow')] = dayLow
        # print(f"start={start}, end={end}, poc={poc}, vah={vah}, val={val}")
    df['slpPoc'] = df['slpPoc'].ewm(span=cfgSVPSlopeCandles,min_periods=cfgSVPSlopeCandles).mean()
    df['slpVah'] = df['slpVah'].ewm(span=cfgSVPSlopeCandles,min_periods=cfgSVPSlopeCandles).mean()
    df['slpVal'] = df['slpVal'].ewm(span=cfgSVPSlopeCandles,min_periods=cfgSVPSlopeCandles).mean()
    df['slpPoc'].clip(lower=1, upper=-1, inplace=True)
    df['slpVah'].clip(lower=0, upper=2, inplace=True)
    df['slpVal'].clip(lower=-2, upper=0, inplace=True)

def genAnalyticsForDay(df_daily,analyticsGenerators): 
    if df_daily.empty or len(df_daily) < 2:
        return df_daily
    for analGen in analyticsGenerators:
        analGen(df_daily)
    
    return df_daily

def generateAnalyticsForFreq(analyticsGenerators,df,freq):
    df.index = pd.to_datetime(df.index, utc=True)
    # Convert timezone-aware datetime index from UTC to IST
    df.index = df.index.tz_convert(ist)
    
    # Assuming the input dataframe is 'df' with a datetime index
    # 1. Split the dataframe into separate dataframes for each day
    daily_dataframes = [group for _, group in df.groupby(pd.Grouper(freq=freq))]

    # 2. Run the 'genAnalyticsForDay' function on each day's dataframe
    daily_analytics = [genAnalyticsForDay(day_df, analyticsGenerators) for day_df in daily_dataframes]

    # 3. Combine the resulting pandas series from the analytics function
    combined_analytics = pd.concat(daily_analytics)

    # 4. Merge the combined series with the original dataframe 'df'
    df_with_analytics = df.merge(combined_analytics, left_index=True, right_index=True)
    return combined_analytics

def generateAnalytics(analyticsGenerators,df):
    dfWithAnalytics = generateAnalyticsForFreq(analyticsGenerators['hourly'],df,'H')
    dfWithAnalytics = generateAnalyticsForFreq(analyticsGenerators['daily'],dfWithAnalytics,'D')
    return dfWithAnalytics

########### END OF POPULATE FUNCTIONS ###########
### NOT IN USE CODE ###
## SIGNAL GENERATION functions
# Functions that generate the signals based on the indicators
# type is 1 for entry or exit, 0 for exit only time frame close
# to end of trading day
# def getSig_BB_CX(type,signal, isLastRow, row, df):
#     # First get the original signal
#     s = signal
    
#     superUpperBandBreached = row['Adj Close'] >= row['super_upper_band']
#     superLowerBandBreached = row['Adj Close'] <= row['super_lower_band']
#     superBandsbreached = superUpperBandBreached or superLowerBandBreached
    
#     lowerBandBreached = row['Adj Close'] <= row['lower_band']
#     upperBandBreached = row['Adj Close'] >= row['upper_band']
    
#     lowerMiniBandBreached =  row['Adj Close'] <= row['mini_lower_band']
#     upperMiniBandBreached = row['Adj Close'] >= row['mini_upper_band']
    
#     if superBandsbreached and (not tickerIsTrending(row['symbol'])):
#         s = 0 # Exit if super bands are breached; we are no longer mean reverting within the BB range
#     else:
#         if lowerMiniBandBreached and \
#             (not tickerIsTrendingDn(row['symbol'])): #Dn trending tickers dance around the lower band, dont use that as an exit signal:
#             if type == 0: # EXIT timeframe
#                 if tickerPositionIsShort(row['symbol']):
#                     s = 0 # Only Exit short positions on lower band breach; long positions will wait for better exit opportunities - or Filters 
#             elif type == 1:
#                 s = 1 if lowerBandBreached else 0 # Only type=1; only enter positions on lower bandbreach, lowermini is for exits
#             else:
#                 raise Exception(f'Invalid type {type}')
            
#         if upperMiniBandBreached and \
#             (not tickerIsTrendingUp(row['symbol'])): #Up trending tickers dance around the upper band, dont use that as an exit signal
#             if type == 0: # EXIT timeframe
#                 if tickerPositionIsLong(row['symbol']):
#                     s = 0 # Only Exit long positions on upper band breach; short positions will wait for better exit opportunities - or Filters
#             elif type == 1:
#                 s = -1 if upperBandBreached else 0 # Only type=1; only enter positions on upper bandbreach, uppermini is for exits
#             else:
#                 raise Exception(f'Invalid type {type}')
    
#     if tickerIsTrending(row['symbol']) and signalChanged(s,signal):
#         if isLastRow:
#             logging.warning(f"{row.symbol}  signal:{signal} s:{s} trend:{getTickerTrend(row['symbol'])} BB CX reset trend before exit trend exiter")
#             #Ideally this should never happen, as trend exiter should exit before this, but if it does happen then 
#         setTickerTrend(row.symbol, 0) #reset trend if we changed a trending ticker
    
#     logSignal('BB-X-CX',["adxData"],signal,s,row,type,isLastRow)

#     return s

# # START OF BB FILTERS
# # Filters only negate BB signals to 0 under certain trending conditions
# # Never give buy or sell signals if BB is not already signalling it 

# # ADX is best for strength of trend, catches even
# # gentle sloping ma, with low OBV, as long as it is long lived
# def getSig_ADX_FILTER (type,signal, isLastRow,row,df):
#     if skipFilter(signal,type):
#         return signal
    
#     s = signal
    
#     if adxIsHigh(row):
        
#         i = df.index.get_loc(row.name) # get index of current row
#         rollbackCandles = round(adxLen*.6) # how many candles to look back
#         # Since ADX is based on last 14 candles, we should look
#         # at slope in the median of that period to understand 
#         # the direction of slope that created this ADX trend
#         # if the trend reverses, ADX may stay high, but slope may
#         # reverse.  Therefore we need to rollback and look at 
#         # old slops to relate it to ADX value
#         oldSlope = df.iloc[i - rollbackCandles,
#                           df.columns.get_loc('SLOPE-OSC')]  
#         if s == 1 and oldSlope < 0 and adxIsBearish(row):
#             s = 0
#         elif s == -1 and oldSlope > 0 and adxIsBullish(row):
#             s = 0
    
#     logSignal('ADX-FLTR',["adxData","maSlpData"],signal,s,row,type,isLastRow)
        
#     return s

# # MA SLOPE FILTER is to catch spikes, get out dont get caught in them
# def getSig_MASLOPE_FILTER (type,signal, isLastRow,row,df):
#     if skipFilter(signal,type):
#         return signal
    
#     s=signal
#     breached = valueOrProjectedValueBreachedThreshold(row['SLOPE-OSC'],
#                                       maSlopeThresh,row['SLOPE-OSC-SLOPE'],
#                                       maSlopeThreshYellowMultiplier, "H_OR_L")

#     # Since this is a FILTER, we only negate long and short signals
#     # on extreme MSSLOPE.
#     # for nan or 0, we just return the signal
    
#     if breached == False:
#         return s
#     else:
#         if s == 1 and breached == 'L':
#             # We want to go long, but the ticker is diving down, ma pct change is too low
#             # so we filter this signal out
#             s = 0
#         elif s == -1 and breached == 'H':
#             s = 0
    
#     logSignal('SLP-FLTR',["maSlpData"],signal,s,row,type,isLastRow)
        
#     return s

# #OBV is best as a leading indicator, flashes and spikes before 
# #the price moves dramatically and it beomes and trend that shows up
# #in MA or ADX etc 
# def getSig_OBV_FILTER (type,signal, isLastRow,row, df):
#     if skipFilter(signal,type)  or (not 'OBV-OSC' in row):
#         return signal
    
#     s = signal
    
#     # Since this is a FILTER, we only negate long and short signals
#     # on extreme OBV.
#     # for nan or 0, we just return the signal
    
#     breached = valueOrProjectedValueBreachedThreshold(row['OBV-OSC'],obvOscThresh,
#                     row['OBV-OSC-PCT-CHNG'], 
#                     obvOscThreshYellowMultiplier, "H_OR_L")

#     if breached == False:
#         return s
#     else:
#         if s == 1 and breached == 'L':
#             s = 0
#         elif s == -1 and breached == 'H':
#             s = 0
#     logSignal('OBV-FLTR',["obvData"],signal,s,row,type,isLastRow)

#     return s

# ## END OF FILTERS

# ### OVERRIDE SIGNAL GENERATORS
# # These are the signal generators that override the other signals
# # They are caleld with other signal generators have already come up
# # with a signal.  These can override in extreme cases such as 
# # sharpe declines or rises in prices or extreme ADX, and can 
# # provide buy/sell signals that override BB for breakout trends
# def getSig_exitAnyExtremeADX_OBV_MA20_OVERRIDE (type, signal, isLastRow, row, df, 
#                                                 last_signal=float('nan')):    
#     if (not np.isnan(signal)) or (tickerIsTrending(row.symbol)):
#         # if singal =0, we are exiting already, nothing for this filter to do
#         # if it is 1 or -1, its already been through relavent filters
#         # Only if it is nan, then we have to ensure conditions have not gotten too bad
#         # and potentially exit before it hits a BB band. 
#         #
#         # This function mostly comes into use during the last exit hour
#         # where we ignore lower BB cross under for long positions
#         # and ignore upper BB cross over for short positions
#         #
#         # In that exit hour, we are waiting for a good exit opportunity
#         # either upper BB crossover for longs, or lower for shorts
#         # but if we dont get that then we will want to exit if the
#         # conditions get extreme; 
#         #
#         # NOTE: WE DO NOT WANT TO EXIT ON GENTLE BB CROSSOVERS IN THE EXIT HOUR
#         # ONLY WE GET A GOOD PRICE OR IF CONDITIONS GET EXTREME IN THE WRONG DIRECTION
#         #
#         # Also ignore trending tickers; let signal exit handle them
#         return signal 

#     if not tickerHasPosition(row['symbol']):
#         return signal # nothing to override, we have no position

#     positionToAnalyse =  getTickerPosition(row['symbol'])    
    
#     if ('OBV-OSC' in row) and (not np.isnan(row['OBV-OSC'])):
#         obvBreached = valueBreachedThreshold(row['OBV-OSC'],obvOscThresh, "H_OR_L")
#     else:
#         obvBreached = False
    
#     if not np.isnan(row['ADX']):
#         adxBreached = valueBreachedThreshold(row['ADX'],adxThresh, "H")
#     else:
#         adxBreached = False
    
#     if not np.isnan(row['SLOPE-OSC']):
#         slopebreached = valueBreachedThreshold(row['SLOPE-OSC'],
#                                         maSlopeThresh, "H_OR_L")
#     else:
#         slopebreached = False
        
#     if obvBreached == False and adxBreached == False and slopebreached == False:
#         return signal    

#     s = signal 
#     #logging.info(f"obvBreached:{obvBreached} adxBreached:{adxBreached} slopebreached:{slopebreached}")
#     if obvBreached:
#         breach = obvBreached
#     elif adxBreached:
#         i = df.index.get_loc(row.name) # get index of current row
#         rollbackCandles = round(adxLen*.6) # how many candles to look back
#         # Since ADX is based on last 14 candles, we should look
#         # at slope in the median of that period to understand 
#         # the direction of slope that created this ADX trend
#         # if the trend reverses, ADX may stay high, but slope may
#         # reverse.  Therefore we need to rollback and look at 
#         # old slops to relate it to ADX value
#         oldSlope = df.iloc[i - rollbackCandles,
#                           df.columns.get_loc('SLOPE-OSC')]  
#         if oldSlope < 0:
#             breach = 'L'
#         else:
#             breach = 'H'
#     elif slopebreached:
#         breach = slopebreached

#     if positionToAnalyse == 1 and breach == 'L':
#         s = 0
#     elif positionToAnalyse == -1 and breach == 'H':
#         s = 0
        
#     logSignal(f'EXIT-EXTRME-COND pToAnal({positionToAnalyse}) obv:{obvBreached} adx{adxBreached} sl{slopebreached}',["obvData","adxData","maSlpData"],signal,s,row,type,isLastRow)
                
#     return s

# def getSig_followAllExtremeADX_OBV_MA20_OVERRIDE (type, signal, isLastRow, row, df, 
#                                                   last_signal=float('nan')):
    
#     if type == 0:# Dont take new positions when its time to exit only
#         return signal
#     s = signal 
#     adxIsHigh = 1 if row['ADX'] >= adxThresh else 0
#     if 'OBV-OSC' in row:
#         obvIsHigh = 1 if (abs(row['OBV-OSC']) >= obvOscThresh) else 0
#     else:
#         obvIsHigh = 1 # if we dont have obv, then we assume its high
        
#     # if adx and obv are high, then maSlope needs to be just somewhat high (yello multiplier)
#     # obv and adx are more telling of trends than ma which could be delayed or less extreme
#     slopeIsHigh = 1 if abs(row['SLOPE-OSC']) >= maSlopeThresh*maSlopeThreshYellowMultiplier else 0
    
#     obvOsc = None if (not 'OBV-OSC' in row) else row['OBV-OSC']
#     obvIsPositive = True if ((obvOsc is None) or (obvOsc > 0)) else False
#     obvIsNegative = True if ((obvOsc is None) or (obvOsc < 0)) else False
        
#     maSlopesUp = row['SLOPE-OSC'] > 0
#     maSlopesDn = row['SLOPE-OSC'] < 0
#     if (adxIsHigh + obvIsHigh + slopeIsHigh) >= cfgNumConditionsForTrendFollow:
#         #We are breaking out Ride the trend
#         #print(f"Extreme ADX/OBV/MA20 OVERRIDE FOLLOW TREND: {row.symbol}@{row.name}")
#         if obvIsPositive and maSlopesUp and signal != 1:
#             if (last_signal != 1 and signal != 1):
#                 s = 1
#         elif obvIsNegative and maSlopesDn and signal != -1:
#             if (last_signal != -1 and signal != -1):
#                 s = -1
        
#         if signalChanged(s,signal):
#             #print("entering trend following", row.i)
#             setTickerTrend(row.symbol, s)
#             # if isLastRow:
#             #     logging.info(f"{row.symbol}:{row.i} => FOLLOW TREND => Extreme ADX/OBV/MA20 OVERRIDE signal ({signal})  s={s} / last_signal ({last_signal}) TO FOLLOW TREND.ADX:{row['ADX']} > {adxThresh} AND OBV:{obvOsc} > {obvOscThresh} AND MA20:{row['SLOPE-OSC']} > {maSlopeThresh}")
#             # else:
#             #     logging.debug(f"{row.symbol}:{row.i}:{row.name}  => FOLLOW TREND => Extreme ADX/OBV/MA20 OVERRIDE signal ({signal})  s={s} / last_signal ({last_signal}) TO FOLLOW TREND.ADX:{row['ADX']} > {adxThresh} AND OBV:{obvOsc} > {obvOscThresh} AND MA20:{row['SLOPE-OSC']} > {maSlopeThresh}")
#     logSignal('FLW-TRND',["obvData","adxData","maSlpData"],signal,s,row,type,isLastRow)

#     return s

# def exitTrendFollowing(type, signal, isLastRow, row, df, 
#                         last_signal=float('nan')):
#     if (not tickerIsTrending(row.symbol)) or \
#         ((not np.isnan(signal)) and (getTickerTrend(row.symbol) == signal)):
#         return signal
#     #If We get here then ticker is trending, and trend signal no longer matches the trend
#     #Trend may not continue, need to finda  good spot to exit 
#     currTrend = getTickerTrend(row.symbol)
#     s = signal
#     i = df.index.get_loc(row.name) # get index of current row
#     # oldSlowMA = df.iloc[i - 1,df.columns.get_loc('ma20')]  
#     # oldFastMA = df.iloc[i - 1,df.columns.get_loc('MA-FAST')]
    
#     adxIsGettingLower = projectedValue(row['ADX'],row['ADX-PCT-CHNG']) <= adxThresh
#     maIsGettingLower = projectedValue(row['SLOPE-OSC'], row['SLOPE-OSC-SLOPE']) <= maSlopeThresh
#     maIsGettingHigher = projectedValue(row['SLOPE-OSC'], row['SLOPE-OSC-SLOPE']) >= -maSlopeThresh
    
#     fastMACrossedOverSlow = row['MA-FAST'] >= row['ma20']
#     fastMACrossedUnderSlow = row['MA-FAST'] <= row['ma20']
    
#     if 'OBV-OSC-PCT-CHNG' in row:
#         obvIsGettingLower = projectedValue(row['OBV-OSC'], row['OBV-OSC-PCT-CHNG']) <= obvOscThresh
#         obvIsGettingHigher = projectedValue(row['OBV-OSC'], row['OBV-OSC-PCT-CHNG']) >= -obvOscThresh
#     else:
#         obvIsGettingLower = True
#         obvIsGettingHigher = True
            
#     #This ticker is trending, lets see if its time to exit
#     trend = getTickerTrend(row.symbol)
#     if trend == 1:
#         if fastMACrossedUnderSlow:
#             # and \
#             # (adxIsGettingLower) and \
#             # (maIsGettingLower) and \
#             # (obvIsGettingLower):
#             s = 0
#     elif trend == -1:   
#         if fastMACrossedOverSlow:
#             # and \
#             # (adxIsGettingLower) and \
#             # (maIsGettingHigher) and \
#             # (obvIsGettingHigher):
#             s = 0
#     else:
#         logging.error("Wierd ! trend should always be 1 or -1")
#         return signal
    
#     if signalChanged(s,signal):
#         setTickerTrend(row.symbol, 0)
#         logSignal('EXT-TRND',["obvData","adxData","maSlpData"],signal,s,row,type,isLastRow,f"({currTrend})")
#         #logString = f"{row.symbol}:{row.i}:{{row.name if isLastRow else ''}}  => EXIT TREND({trend}) on fastMA crossover ADX:{row['ADX']} > {adxThresh} AND OBV:{row['OBV-OSC-PCT-CHNG']} > {obvOscThresh} AND MA20:{row['SLOPE-OSC']} > {maSlopeThresh}"
#     else:
#         adxString = ""
#         maString = ""
#         obvString = ""
#         if adxIsGettingLower:
#             adxString = "adx L"
#         if maIsGettingHigher:
#             maString = "ma H"
#         elif maIsGettingLower:
#             maString = "ma L"
        
#         if 'OBV-OSC-PCT-CHNG' in row:
#             if obvIsGettingHigher:
#                 obvString = "obv H"
#             elif obvIsGettingLower:
#                 obvString = "obv L"
#         else:
#             obvString = "obv N/A"
            
#         logSignal('CNT-TRND',["obvData","adxData","maSlpData"],signal,s,row,type,isLastRow,
#                   f'({currTrend}) cx:{"Ov" if fastMACrossedOverSlow else "Un"} {adxString} {obvString} {maString} ',
#                   logWithNoSignalChange=True)
#         #logString = f"{row.symbol}:{row.i}:{{row.name if isLastRow else ''}}  => DONT EXIT TREND YET ({trend}) cxOver:{fastMACrossedOverSlow} cxUndr:{fastMACrossedUnderSlow} ADX:{row['ADX']} > {adxThresh} AND OBV:{row['OBV-OSC-PCT-CHNG']} > {obvOscThresh} AND MA20:{row['SLOPE-OSC']} > {maSlopeThresh} "
#     # if isLastRow:
#     #     logging.info(logString)
#     # else:
#     #     logging.debug(logString)
    
#     return s

# def followTrendReversal (type, signal, isLastRow, row, df, 
#                         last_signal=float('nan')):
#     # SLOPE-OSC below yellow threshold, but coming up fast (REVERSALy)
#     # and obv above yellow threshold, adx above yellow
    
#     s = signal
    
#     maSlopeIsLow = isAlmostLow(row['SLOPE-OSC'], maSlopeThresh, maSlopeThreshYellowMultiplier)
#     maSlopeIsGettingHigher = isGettingHigher(row['SLOPE-OSC'], maSlopeThresh)
#     slopeOscIsWillCrossOverLowThreshold = projectedValueBreachedThreshold \
#         (row['SLOPE-OSC'], -maSlopeThresh, maSlopeThreshYellowMultiplier,
#          row['SLOPE-OSC-SLOPE'], 'H')
#     slopeHasReversedUp = maSlopeIsLow and maSlopeIsGettingHigher and slopeOscIsWillCrossOverLowThreshold
        
#     maSlopeIsHigh = isAlmostHigh(row['SLOPE-OSC'], maSlopeThresh, maSlopeThreshYellowMultiplier)
#     maSlopeIsGettingLower = isGettingLower(row['SLOPE-OSC'], maSlopeThresh)
#     slopeOscIsWillCrossUnderHighThreshold = projectedValueBreachedThreshold \
#         (row['SLOPE-OSC'], -maSlopeThresh, maSlopeThreshYellowMultiplier,
#          row['SLOPE-OSC-SLOPE'], 'L')
#     slopeHasReversedDn = maSlopeIsHigh and maSlopeIsGettingLower and slopeOscIsWillCrossUnderHighThreshold
    
#     adxAboveYellow = row['ADX'] > (adxThresh*adxThreshYellowMultiplier)
    
#     obvBreached = valueOrProjectedValueBreachedThreshold(row['OBV-OSC'],obvOscThresh,
#                     row['OBV-OSC-PCT-CHNG'], 
#                     obvOscThreshYellowMultiplier, "H_OR_L")

#     if slopeHasReversedUp and adxAboveYellow and (obvBreached == 'H'):
#         s = 1
#     elif slopeHasReversedDn and adxAboveYellow and (obvBreached == 'L'):
#         s = -1
#     if signalChanged(s,signal):
#         #print("entering trend following", row.i)
#         setTickerTrend(row.symbol, s)
#         if isLastRow:
#             logging.info(f"{row.symbol}:{row.i} => FOLLOW TREND-REVERSAL => Extreme ADX/OBV/MA20 OVERRIDE signal ({signal})  s={s} / last_signal ({last_signal}) TO FOLLOW TREND.ADX:{row['ADX']} > {adxThresh} AND OBV:{obvOsc} > {obvOscThresh} AND MA20:{row['SLOPE-OSC']} > {maSlopeThresh}")
#         else:
#             logging.debug(f"{row.symbol}:{row.i}:{row.name}  => FOLLOW TREND-REVERSAL => Extreme ADX/OBV/MA20 OVERRIDE signal ({signal})  s={s} / last_signal ({last_signal}) TO FOLLOW TREND.ADX:{row['ADX']} > {adxThresh} AND OBV:{obvOsc} > {obvOscThresh} AND MA20:{row['SLOPE-OSC']} > {maSlopeThresh}")
#     return s 
# def justFollowADX(type, signal, isLastRow, row, df, 
#                         last_signal=float('nan')):
#     s = signal
#     if adxIsHigh(row):
#         if adxIsBullish(row):
#             s = 1
#         elif adxIsBearish(row):
#             s = -1
#     elif tickerIsTrending(row.symbol) and adxIsLow(row):
#         s = 0
#     setTickerTrend(row.symbol, s) if signalChanged(s,signal) else None

#     print(f"{row.symbol}:{row.i}:{row.name}  => FOLLOW ADX => adxi is low:{adxIsLow(row)} Trend?: {tickerIsTrending(row.symbol)} ADX:{row['ADX']} > {adxThresh} yellowMult:{adxThreshYellowMultiplier} exitMult:{cfgAdxThreshExitMultiplier} s={s} / last_signal ({last_signal})")
#     return s
# def justFollowMA(type, signal, isLastRow, row, df, 
#                         last_signal=float('nan')):
#     s = signal
#     if maSteepSlopeUp(row):
#         s = 1
#     elif maSteepSlopeDn(row):
#         s = -1
#     #    else: No need to exit if ADX is High, and trend has not yet fully reversed; no new entry, but no exit either in hte mid zone
#     setTickerTrend(row.symbol, s) if signalChanged(s,signal) else None
#     return s
# def followMAandADX(type, signal, isLastRow, row, df,
#                         last_signal=float('nan')):
#     s = signal
#     if adxIsHigh(row):
#         adxStatus = 'ADX-HIGH'
#         if maSteepSlopeUp(row):
#             s = 1
#         elif maSteepSlopeDn(row):
#             s = -1
#     elif tickerIsTrending(row.symbol) and maDivergesFromTrend(row):
#         s = 0
#     setTickerTrend(row.symbol, s) if signalChanged(s,signal) else None
#     logSignal('TRND-SLP-ADX',["adxData","maSlpData"],signal,s,row,type,isLastRow)

#     return s
# def followSuperTrend(type, signal, isLastRow, row, df, 
#                         last_signal=float('nan')):
#     #return row['SuperTrendDirection']
#     s = signal
#     if row['SuperTrend'] > 0:
#         s = 1 if row['ma_superTrend_pct_change'] > 50 else 0
#     elif row['SuperTrend'] < 0:
#         s = -1 if row['ma_superTrend_pct_change'] < -50 else 0
#     setTickerTrend(row.symbol, s) if signalChanged(s,signal) else None
#     if isLastRow:
#         logging.info(f"{row.symbol}:{row.i} => FOLLOW SUPERTREND => {s} ")

#     return s
# def followObvAdxMA(type, signal, isLastRow, row, df, 
#                         last_signal=float('nan')):
#     s = signal
#     obvBreach = projectedValueBreachedThreshold(row['OBV-OSC'],obvOscThresh,
#                 row['OBV-OSC-PCT-CHNG'], 
#                 obvOscThreshYellowMultiplier, "H_OR_L")

#     if obvBreach == 'L' and adxIsHigh(row) and maSteepSlopeDn(row):
#         s = -1
#     elif obvBreach == 'H' and adxIsHigh(row) and maSteepSlopeUp(row):
#         s = 1
        
#     setTickerTrend(row.symbol, s) if signalChanged(s,signal) else None
#     logSignal('TRND-OBV-ADX-SLP',["obvData","adxData","maSlpData"],signal,s,row,type,isLastRow,logWithNoSignalChange=True)
#     return s
# def followObvMA(type, signal, isLastRow, row, df, 
#                         last_signal=float('nan')):
#     s = signal
#     if obvIsBearish(row) and maSteepSlopeDn(row):
#         s = -1
#     elif obvIsBullish(row) and maSteepSlopeUp(row):
#         s = 1
        
#     setTickerTrend(row.symbol, s) if signalChanged(s,signal) else None
#     logSignal('TRND-OBV-SLP',["obvData","maSlpData"],signal,s,row,type,isLastRow,logWithNoSignalChange=True)
#     return s

# def exitOBV(type, signal, isLastRow, row, df, 
#                         last_signal=float('nan')):
#     if (not tickerHasPosition(row.symbol)):
#         return signal
#     s = signal
#     pos = getTickerPosition(row.symbol)
    
#     if tickerHasLongPosition(row.symbol) and obvNoLongerHigh(row):
#         log = "EXIT-OBV-NO-LONGER-HIGH"
#         s = 0  
#     elif tickerHasShortPosition(row.symbol) and obvNoLongerLow(row):
#         log = "EXIT-OBV-NO-LONGER-LOW"
#         s = 0  
#     if signalChanged(s,signal):
#         setTickerTrend(row.symbol, s) if tickerIsTrending(row.symbol) else None
#         logSignal(log,["obvData","adxData","maSlpData"],signal,s,row,type,isLastRow)
#     return s 


# def followOBVSlope(type, signal, isLastRow, row, df, 
#                         last_signal=float('nan')):
#     s = signal
#     breached = valueOrProjectedValueBreachedThreshold(row['OBV-OSC'],obvOscThresh,
#                 row['OBV-OSC-PCT-CHNG'], 
#                 obvOscThreshYellowMultiplier, "H_OR_L")

#     if row['OBV-OSC-PCT-CHNG'] > -0.02 and row['OBV-OSC'] >= obvOscThresh:
#         s = 1
#     elif row['OBV-OSC-PCT-CHNG'] < 0.02 and row['OBV-OSC'] <= -obvOscThresh:
#         s = -1
#     else:
#         s = 0

#     setTickerTrend(row.symbol, s) if signalChanged(s,signal) else None
#     if isLastRow:
#         logging.info(f"{row.symbol}:{row.i} => FOLLOW SUPERTREND => {s} ")

#     return s

# def fastSlowMACX(type, signal, isLastRow, row, df, 
#                         last_signal=float('nan')):
#     s = signal
#     if adxIsHigh(row):
#         if crossOver(row['MA-FAST'], row['ma20']):
#             s = 1
#         elif crossUnder(row['MA-FAST'], row['ma20']):
#             s = -1
#         else:
#             s = 0
#     else:
#         s = 0
#     setTickerTrend(row.symbol, s)
#     return s



# def filterRenko(type, signal, isLastRow, row, df, 
#                         last_signal=float('nan')):
#     if np.isnan(signal) or signal == 0:
#         return signal

# def followRenkoWithTargetedEntry(type, signal, isLastRow, row, df, 
#                         last_signal=float('nan')):
#     s = signal
#     trade_price = float('nan')
#     if row['renko_uptrend'] == True:
#         # we just entered a trend may bounce around entry brick lines; 
#         if not (row['renko_brick_num'] == 1 and getTickerTrend(row.symbol) == 1): 
#             if (type == 1 and row['renko_brick_num'] >= 2):
#                 if row.Low <= row.lower_band:
#                     s = 1
#                     trade_price = row.lower_band
#             else:
#                 s = 0
#     else:
#         if not (row['renko_brick_num'] == -1 and getTickerTrend(row.symbol) == -1):
#             if (type == 1 and row['renko_brick_num'] <= -2):
#                 if row.High >= row.upper_band:
#                     s = -1
#                     trade_price = row.upper_band
#             else:
#                 s = 0
#     if signalChanged(s,signal):
#         setTickerTrend(row.symbol, s)
#     logSignal('FOLLW-RENKO',['RenkoData'],signal,s,row,type,isLastRow,'',logWithNoSignalChange=True)
#     return (s,trade_price)
# def randomSignalGenerator(type, signal, isLastRow, row, df, 
#                         last_signal=float('nan')):
#     if random.randint(0,100) > 90:
#         s = random.randint(-1,1)
#         setTickerTrend(row.symbol, s)
#     else:
#         s = signal
#     return s
# def exitTarget(type, signal, isLastRow, row, df):
#     if not (tickerHasPosition(row.symbol) and \
#             cfgTarget):
#         # return if ticker has no position
#         # or if stop loss not configured
#         # or if current signal is 1 or -1 or 0
#         return signal
    
#     s = signal
#     entryPrice = getTickerEntryPrice(row.symbol)
#     # We proceed only if tickerHasPosition, and stop loss is configured
#     # and current signal is nan 
    
#     if tickerHasLongPosition(row.symbol):
#         if row.High >= ((1+cfgTarget)*entryPrice):
#             s = 0
#     else:
#         if row.Low <= ((1-cfgTarget)*entryPrice):
#             s = 0
#     if signalChanged(s,signal):
#         #print("entering trend following", row.i)
#         setTickerTrend(row.symbol, s) if tickerIsTrending(row.symbol) else None
#         # logSignal('TARGET-HIT',["obvData","adxData","maSlpData"],signal,s,row,type,isLastRow,
#         #           f'(E:{entryPrice}, L:{row.Low}, H:{row.High} sl:{cfgStopLoss}) ',
#         #           logWithNoSignalChange=True)
#     return s 
# def exitStopLoss(type, signal, isLastRow, row, df):
#     if not (tickerHasPosition(row.symbol) and \
#             cfgStopLoss):
#         # return if ticker has no position
#         # or if stop loss not configured
#         # or if current signal is 1 or -1 or 0
#         return signal
    
#     s = signal
#     entryPrice = getTickerEntryPrice(row.symbol)
#     # We proceed only if tickerHasPosition, and stop loss is configured
#     # and current signal is nan 
#     renkoBrickSize = row['renko_brick_high'] - row['renko_brick_low']
#     if tickerHasLongPosition(row.symbol):
#         sl = ((1-cfgStopLoss)*entryPrice)
#         sl = row['renko_brick_low']-renkoBrickSize
#         if row.Low <= sl:
#             s = 0
#     else:
#         sl = ((1+cfgStopLoss)*entryPrice)
#         sl = row['renko_brick_high']+(renkoBrickSize*3)
#         if row.High >= sl:
#             s = 0
#     if signalChanged(s,signal):
#         #print("entering trend following", row.i)
#         setTickerTrend(row.symbol, s) if tickerIsTrending(row.symbol) else None
#         # logSignal('STOP-LOSS',["obvData","adxData","maSlpData"],signal,s,row,type,isLastRow,
#         #           f'(E:{entryPrice}, L:{row.Low}, H:{row.High} sl:{cfgStopLoss}) ',
#         #           logWithNoSignalChange=True)
#     return s 

#     return s
# def exitCandleStickReversal(type, signal, isLastRow, row, df):
#     return row['candlestick_signal']
#     print(f"{row.name} hanngingMan:{row['HANGINGMAN']}")
#     s=signal
#     if row['HANGINGMAN'] == 1:
#         s = 0
#         exitType='EXIT-HANGING-MAN'
#     if signalChanged(s,signal):
#         setTickerTrend(row.symbol, s) if tickerIsTrending(row.symbol) else None
#         logSignal(exitType,["obvData","adxData","maSlpData"],signal,s,row,type,isLastRow)
#     return s
    
# enoughForDay = []
# def exitEnoughForTheDay(type, signal, isLastRow, row, df, last_signal=float('nan')):
#     global enoughForDay
#     s = signal = row.signal
#     if signal == 0 :
#         return row.signal
#     date = row.name.date()
#     if date in enoughForDay:
#         return 0
    
#     todayDF = df.loc[df.index.date == date]
#     todayDF = todayDF.loc[todayDF.index <= row.name]
#     perf.prep_dataframe(todayDF, close_at_end=False)
#     trades = perf.get_trades(todayDF)
#     if (len(trades) > 0):
#         for index,trade in trades.iterrows():
#             ret = trade["sum_return"]
#             if enoughReturnForDay(ret):
#                 break
        
#         if enoughReturnForDay(ret):
#             s = 0
#         else:
#             #check if we are in a trade
#             pos = trades.iloc[-1].loc['position']
#             if trades['position'].iloc[-1] != 0:
#                 trade_entry = trades.iloc[-1].loc['Open']
#                 curr_price = todayDF.iloc[-1].loc['Adj Close']
#                 trade_ret = pos*(curr_price - trade_entry)/trade_entry
#                 ret = ret + trade_ret
#                 if enoughReturnForDay(ret):
#                         s = 0
#     else:
#         ret = 0
    
#     if s == 0:
#         enoughForDay.append(date)
#         print(f"enough for {date}")
          
#     if signalChanged(s,signal):
#         setTickerTrend(row.symbol, s) if tickerIsTrending(row.symbol) else None
#     logSignal('EXIT-ENOUGH-FOR-TODAY',["obvData","adxData","maSlpData"],signal,s,row,type,isLastRow,
#                 f'(ret:{ret}) > {cfgEnoughReturnForTheDay}')
#     trades.to_csv('trades.csv')
#     return s
### END OF NOT IN USE FUNCTIONS
def getNumBricksForLongTrend(row):
    return cfgRenkoNumBricksForTrend if getSVPquadrant(row) != 'Low' or row['slpPoc'] <= -cfgSVPSlopeThreshold else cfgRenkoNumBricksForTrend-1
def getNumBricksForShortTrend(row):
    return cfgRenkoNumBricksForTrend if getSVPquadrant(row) != 'High' or row['slpPoc'] >= cfgSVPSlopeThreshold else cfgRenkoNumBricksForTrend-1

def checkRenkoLongEntry(s,row,df,isLastRow, entry_price,limit1,limit2,sl1,sl2,logString):
    if isRenkoUpTrend(row) != True or tickerHasPosition(row.symbol):
        print(f"ERROR: checkRenkoLongEntry called with position or without uptrend.  Trend:{row['renko_uptrend']} Position:{tickerHasPosition(row.symbol)}")
        exit(1)
    (brickNum,brickSize,brickHigh,brickLow,close) = (row['renko_brick_num'],row['renko_brick_high'] - row['renko_brick_low'],row['renko_brick_high'],row['renko_brick_low'],row['Adj Close'])
    prevLimit1 = getTickerLimit1(row.symbol) if isLongLimit1Order(row.symbol) else float('nan')
    logString = "NO-RENKO-TREND"
    resistance = longResistance(row)
    
    if svpTrendsDown(row):
        logging.info(f"Skipping Long Entry because SVP is trending down") if isLastRow or cfgIsBackTest else None
        return (s,entry_price,limit1,limit2,sl1,sl2,logString)

    if brickNum >= 1:
        if (resistance > (max(close,brickHigh)+brickSize)): # at least a brick away from resistance
            if brickNum < getNumBricksForLongTrend(row):
                if row.High >= prevLimit1:
                    s = 1
                    entry_price = prevLimit1 if prevLimit1 >= row.Low else row.Open 
                    logString = "RENKO-LONG-ENTRY"
                    logging.info(f"Limit order hit at {prevLimit1}. Next row will form next brick.")  if isLastRow else None
                else:
                    potentialLongEntry = brickHigh + ((getNumBricksForLongTrend(row)-brickNum)*brickSize)
                    if resistance > potentialLongEntry+brickSize:
                        logString = "RENKO-WAITING-FOR-LONG-TREND"
                        limit1 = potentialLongEntry
                    else:
                        logging.info(f"Skipping Long Entry limit order because resistance {resistance} is too close to LongEntryTarget:{potentialLongEntry}") if isLastRow else None
            else:
                s = 1
                entry_price = prevLimit1 if row.High >= prevLimit1 else row.Open
                logString = "RENKO-LONG-ENTRY"
                logging.info(f"Enter LONG @ {prevLimit1} Resistance {resistance} far from close:{close} or brickHigh {brickHigh} + BrickSize {brickSize}")  if isLastRow else None
        else:
            logging.info(f"Long Resistance {resistance} is too close to brickHigh:{brickHigh} or close:{close} to enter long")  if isLastRow else None
                
    return (s,entry_price,limit1,limit2,sl1,sl2,logString)

def checkRenkoShortEntry(s,row,df,isLastRow, entry_price,limit1,limit2,sl1,sl2,logString):
    if isRenkoUpTrend(row) != False or tickerHasPosition(row.symbol):
        print(f"ERROR: checkRenkoShortEntry called with position or without downtrend.  Trend:{row['renko_uptrend']} Position:{tickerHasPosition(row.symbol)}")
        exit(1)
        
    (brickNum,brickSize,brickHigh,brickLow, close) = (row['renko_brick_num'],row['renko_brick_high'] - row['renko_brick_low'],row['renko_brick_high'],row['renko_brick_low'], row['Adj Close'])
    logString = "NO-RENKO-TREND"
    prevLimit1 = getTickerLimit1(row.symbol) if isShortLimit1Order(row.symbol) else float('nan')
    absBrickNum = abs(brickNum)
    resistance = shortResistance(row)

    if svpTrendsUp(row):
        logging.info(f"Skipping Short Entry because SVP is trending up") if isLastRow or cfgIsBackTest else None
        return (s,entry_price,limit1,limit2,sl1,sl2,logString)

    if absBrickNum >= 1:
        if (resistance < (min(close,brickLow)-brickSize)): # at least a brick away from resistance

            if absBrickNum < getNumBricksForShortTrend(row):
                if row.Low <= prevLimit1:
                    s = -1
                    entry_price = prevLimit1 if row.High >= prevLimit1 else row.Open
                    logString = "RENKO-SHORT-ENTRY"
                    logging.info(f"Limit order hit at {prevLimit1}. Next row will form next brick.")  if isLastRow else None
                else:
                    potentialShortEntry = (brickLow - ((getNumBricksForShortTrend(row)-absBrickNum) * brickSize))
                    if resistance < potentialShortEntry-brickSize:
                        logString = "RENKO-WAITING-FOR-SHORT-TREND"
                        limit1 = -potentialShortEntry
                    else:
                        logging.info(f"Skipping Short Entry limit order because resistance {resistance} is too close to ShortEntryTarget: {potentialShortEntry}")  if isLastRow else None
            else:
                s = -1
                entry_price = prevLimit1 if row.High >= prevLimit1 else row.Open
                logString = "RENKO-SHORT-ENTRY"
                logging.info(f"Enter Short @ {prevLimit1} Resistance {resistance} far from close{close} or BrickLow {brickLow} - BrickSize {brickSize}") if isLastRow else None
        else:
            logging.info(f"Short Resistance {resistance} is too close to brickLow:{brickLow} or close:{close} to enter Short") if isLastRow else None

    return (s,entry_price,limit1,limit2,sl1,sl2,logString)

def checkRenkoLongExit(s,row,df,isLastRow, exit_price,limit1,limit2,sl1,sl2,logString):
    if not tickerHasLongPosition(row.symbol):
        print(f"ERROR: checkRenkoLongExit called without long position")
        exit(1)
    (upTrend,brickNum,brickSize,brickHigh,brickLow,close) = (isRenkoUpTrend(row),row['renko_brick_num'],row['renko_brick_high'] - row['renko_brick_low'],row['renko_brick_high'],row['renko_brick_low'],row['Adj Close'])
    logString = "RENKO-LONG-CONTINUE"
    prevSL1 = getTickerSL1(row.symbol) if isShortSL1Order(row.symbol) else float('nan')
    prevTarget = getTickerLimit1(row.symbol) if isShortLimit1Order(row.symbol) else float('nan')
    resistance = longResistance(row)
    support = longSupport(row)
    
    if (not upTrend):
        s = 0
        exit_price = prevSL1 if prevSL1<=row.High and prevSL1>=row.Low else row.Open
        logString = "RENKO-LONG-EXIT"
        logging.info(f"EXIT Long @ SL {prevSL1}") if isLastRow else None
    else:
        if row.Low <= prevSL1:
            s = 0
            exit_price = prevSL1 if prevSL1<=row.High else row.Open
            logString = "RENKO-LONG-EXIT"
            logging.info(f"Stop Loss order hit at {prevSL1}. Next row should reflect exit.")  if isLastRow else None
        elif row.High >= prevTarget:
            s = 0
            exit_price = prevTarget if prevTarget >= row.Low else row.Open
            logString = "RENKO-LONG-EXIT"
            logging.info(f"Target hit at {prevTarget}") if isLastRow else None
        else:
            sl1 = -max(support*(1-cfgSLPercentageFromSupport), (brickLow - (brickSize*cfgRenkoBrickMultiplierLongSL)))
            sl1 = sl1 - 10 if row.slpVal >= cfgSVPSlopeThreshold else sl1
            limit1 = -min(resistance*(1-cfgTargetPercentageFromResistance),(close + (brickSize * cfgRenkoBrickMultiplierLongTarget))) if row.renko_brick_diff == 0 else -(brickHigh + (brickSize * 10))
    return (s,exit_price,limit1,limit2,sl1,sl2,logString)

def checkRenkoShortExit(s,row,df,isLastRow, exit_price,limit1,limit2,sl1,sl2,logString):
    if not tickerHasShortPosition(row.symbol):
        print(f"ERROR: checkRenkoShortExit called without Short position")
        exit(1)
    (upTrend,brickNum,brickSize,brickHigh,brickLow,close) = (isRenkoUpTrend(row),row['renko_brick_num'],row['renko_brick_high'] - row['renko_brick_low'],row['renko_brick_high'],row['renko_brick_low'],row['Adj Close'])
    logString = "RENKO-SHORT-CONTINUE"
    prevSL1 = getTickerSL1(row.symbol) if isLongSL1Order(row.symbol) else float('nan')
    prevTarget = getTickerLimit1(row.symbol) if isLongLimit1Order(row.symbol) else float('nan')
    
    resistance = shortResistance(row)
    support = shortSupport(row)

    if upTrend:
        s = 0
        exit_price = prevSL1 if prevSL1<=row.High and prevSL1>=row.Low else row.Open
        logString = "RENKO-SHORT-EXIT"
        logging.info(f"EXIT Short @ SL: {prevSL1}") if isLastRow else None
    else:
        if row.High >= prevSL1:
            s = 0
            exit_price = prevSL1 if prevSL1>=row.Low else row.Open
            logString = "RENKO-SHORT-EXIT"
            logging.info(f"Stop Loss order hit at {sl1}. Next row should reflect exit.") if isLastRow else None
        elif row.Low <= prevTarget:
            s = 0
            exit_price = prevTarget if prevTarget<=row.High else row.Open
            logString = "RENKO-SHORT-EXIT"
            logging.info(f"Short Target hit at {prevTarget}")     if isLastRow else None
        else:
            sl1 = min(support*(1+cfgSLPercentageFromSupport), brickHigh + (brickSize*cfgRenkoBrickMultiplierShortSL))
            sl1 = sl1 + 10 if row.slpVal <= -cfgSVPSlopeThreshold else sl1
            limit1 = max(resistance*(1-cfgTargetPercentageFromResistance), close - (brickSize * cfgRenkoBrickMultiplierShortTarget)) if row.renko_brick_diff == 0 else brickLow - (brickSize * 10)
    return (s,exit_price,limit1,limit2,sl1,sl2,logString)

def checkRenkoExit(s,row,df,isLastRow, entry_price,limit1,limit2,sl1,sl2,logString=''):
    # Potential Strategies
    # 1.  Avg bricks max is around x continuous bricks, avg is y etc .. exit after y 
    # 2.  Exit based on slope of vah/val/poc
    # 3.  Exit when approach support / resistance levels in SVP - vah/val/poc depending on where we are, potentially also depending on slope of vah/l/poc, for instance, we wont exit on hitting vah if vah is sloping up
    # 4.  Exit when progressive bricks are of declining volume ? Could just take total brick volume, or could calculate volume profile of brick, and check if poc is towards bottom or top of brick
    # 5.  Exit when trend reverses based on chart patterns (like head and shoulders, double/triple top/bottom,Rising and Falling Three Methods)

    
    if tickerHasLongPosition(row.symbol) :
        (s,entry_price,limit1,limit2,sl1,sl2,logString) = checkRenkoLongExit(s,row,df,isLastRow, entry_price,limit1,limit2,sl1,sl2,logString)
    elif tickerHasShortPosition(row.symbol):
        (s,entry_price,limit1,limit2,sl1,sl2,logString) = checkRenkoShortExit(s,row,df,isLastRow, entry_price,limit1,limit2,sl1,sl2,logString)
    else:
        print(f"ERROR: checkRenkoExit called for ticker without position. Position:{getTickerPosition(row.symbol)}")
        exit(1)
    return (s,entry_price,limit1,limit2,sl1,sl2,logString)
def checkRenkoEntry(s,row,df,isLastRow, entry_price,limit1,limit2,sl1,sl2,logString=''):
    if isRenkoUpTrend(row) == True:
        (s,entry_price,limit1,limit2,sl1,sl2,logString) = checkRenkoLongEntry(s,row,df,isLastRow, entry_price,limit1,limit2,sl1,sl2,logString)
    elif isRenkoUpTrend(row) == False:
        (s,entry_price,limit1,limit2,sl1,sl2,logString) = checkRenkoShortEntry(s,row,df,isLastRow, entry_price,limit1,limit2,sl1,sl2,logString)
    else: # nan renko_uptrend
        (s,entry_price,logString) = (float('nan'),float('nan'),"No-RENKO-TREND")
    return (s,entry_price,limit1,limit2,sl1,sl2,logString)

def followRenkoWithOBV(type, signal, isLastRow, row, df, 
                        last_signal=float('nan'), prev_trade_price=float('nan'),
                        prev_limit1=float('nan'), prev_limit2=float('nan'),
                        prev_sl1=float('nan'), prev_sl2=float('nan')):
    s = signal
    (entry_price,entry_price,limit1,limit2,sl1,sl2) = (prev_trade_price,prev_trade_price,prev_limit1,prev_limit2,prev_sl1,prev_sl2)
    if np.isnan(row['renko_uptrend']):
        return (s,entry_price,limit1,limit2,sl1,sl2)
    brick_size = row.renko_brick_high - row.renko_brick_low
    
    if tickerHasPosition(row.symbol):
        (s,entry_price,limit1,limit2,sl1,sl2,logString) = checkRenkoExit(s,row,df, isLastRow,entry_price,limit1,limit2,sl1,sl2)
    elif type == 1:
        (s,entry_price,limit1,limit2,sl1,sl2,logString) = checkRenkoEntry(s,row,df, isLastRow, entry_price,limit1,limit2,sl1,sl2)
    else:
         (s,entry_price,logString) = (float('nan'),float('nan'),"RENKO-NO-NEW-ENTRY")
    if signalChanged(s,signal):
        setTickerTrend(row.symbol, s)
    logSignal(logString,['RenkoData','ohlv','svp'],signal,s,row,type,isLastRow,f'TrdPrice:{entry_price} LIM:{limit1} SL:{sl1}',logWithNoSignalChange=True)
    return (s,entry_price, limit1, limit2, sl1, sl2)

def checkSVPShortEntry(s,row,df,isLastRow, entry_price,limit1,limit2,sl1,sl2,logString):
    prevSL1 = getTickerSL1(row.symbol) if isShortSL1Order(row.symbol) else float('nan')
    prevTarget = getTickerLimit1(row.symbol) if isShortLimit1Order(row.symbol) else float('nan')

    logString = 'WAITING-FOR-SHORT-ENTRY'

    if row.Low <= prevSL1:
        s = -1
        entry_price = min(prevSL1,row.Open)
        if prevSL1 - row.Open > 2:
            entry_price = float('nan')
            print(f"ERROR: checkSVPShortEntry: Entry price {row.Open} is more than 2 away from SL {prevSL1}")
            return (s,entry_price, limit1, limit2, sl1, sl2,logString)
            exit(0)

        logString = "SHORT-ENTRY"
        logging.info(f"SL order hit at {entry_price}.") if isLastRow else None
    elif row.High >= prevTarget:
        s = -1
        entry_price = max(prevTarget,row.Open)
        if row.Low > entry_price:
            print(f"ERROR: checkSVPShortEntry: How can row.Low {row.Low} be greater than limit entry {entry_price}")
            #possible I guess because our limit order was not actually there, this is just a simulation backtest, if it was there then row high would have taken it out
            #exit(0)

        logString = "SHORT-ENTRY"
        logging.info(f"Target Limit order hit at {entry_price}.") if isLastRow else None

    else:
        (s, limit1, limit2, sl1, sl2,logString) = \
            signalGenerator.checkShortEntry(s,row,df,isLastRow,limit1,limit2,sl1,sl2,logString)

    return (s,entry_price, limit1, limit2, sl1, sl2,logString)

def checkSVPLongEntry(s,row,df,isLastRow, entry_price,limit1,limit2,sl1,sl2,logString):
    prevSL1 = getTickerSL1(row.symbol) if isLongSL1Order(row.symbol) else float('nan')
    prevTarget = getTickerLimit1(row.symbol) if isLongLimit1Order(row.symbol) else float('nan')
    
    logString = 'WAITING-FOR-LONG-ENTRY'
    if row.High >= prevSL1:
        s = 1
        entry_price = max(prevSL1,row.Open)
        if row.Open - prevSL1 > 2:
            print(f"ERROR: checkSVPLongEntry: Entry price {row.Open} is more than 2 away from SL {prevSL1}")
            # exit(0)
        logString = "LONG-ENTRY"
        logging.info(f"SL order hit at {entry_price}.")  if isLastRow else None
    elif row.Low <= prevTarget:
        s = 1
        entry_price = min(prevTarget,row.Open)
        if row.High < prevTarget:
            print(f"ERROR: checkSVPLongEntry: How can row high {row.High} be less than limit entry {prevTarget}")
            #possible I guess because our limit order was not actually there, this is just a simulation backtest, if it was there then row high would have taken it out
            #exit(0)
        logString = "LONG-ENTRY"
        logging.info(f"Target Limit order hit at {entry_price}.")  if isLastRow else None

    else:
        (s, limit1, limit2, sl1, sl2,logString) = \
            signalGenerator.checkLongEntry(s,row,df,isLastRow,limit1,limit2,sl1,sl2,logString)
    return (s,entry_price, limit1, limit2, sl1, sl2,logString)

def checkSVPLongExit(s,row,df,isLastRow, exit_price,limit1,limit2,sl1,sl2,logString):
    if not tickerHasLongPosition(row.symbol):
        print(f"ERROR: checkRenkoLongExit called without long position")
        exit(1)
    logString = "LONG-CONTINUE"
    prevSL1 = getTickerSL1(row.symbol) if isShortSL1Order(row.symbol) else float('nan')
    prevTarget = getTickerLimit1(row.symbol) if isShortLimit1Order(row.symbol) else float('nan')
    entryPrice = getTickerEntryPrice(row.symbol)    
    
    if row.Low <= prevSL1: #SL
        s = 0
        exit_price = min(prevSL1,row.Open)
        logString = "LONG-EXIT"
        logging.info(f"Stop Loss order hit at {prevSL1}. Next row should reflect exit.")  if isLastRow else None
    elif row.High >= prevTarget:#Target
        s = 0
        exit_price = max(prevTarget,row.Open)
        logString = "LONG-EXIT"
        logging.info(f"Target hit at {exit_price}") if isLastRow else None
    else:
        (s, limit1, limit2, sl1, sl2,logString) = \
            signalGenerator.checkLongExit(s,row,df,isLastRow, entryPrice,limit1,limit2,sl1,sl2,logString)

    return (s,exit_price,limit1,limit2,sl1,sl2,logString)

def checkSVPShortExit(s,row,df,isLastRow, exit_price,limit1,limit2,sl1,sl2,logString):
    if not tickerHasShortPosition(row.symbol):
        print(f"ERROR: checkRenkoShortExit called without Short position")
        exit(1)
    logString = "SHORT-CONTINUE"
    prevSL1 = getTickerSL1(row.symbol) if isLongSL1Order(row.symbol) else float('nan')
    prevTarget = getTickerLimit1(row.symbol) if isLongLimit1Order(row.symbol) else float('nan')
    entryPrice = getTickerEntryPrice(row.symbol)
        
    if row.High >= prevSL1:
        s = 0
        exit_price = max(prevSL1,row.Open) if prevSL1>=row.Low else row.Open
        logString = "SHORT-EXIT"
        logging.info(f"Stop Loss order hit at {exit_price}. Next row should reflect exit.") if isLastRow else None
    elif row.Low <= prevTarget:
        s = 0
        exit_price = min(prevTarget,row.Open)
        logString = "SHORT-EXIT"
        logging.info(f"Short Target hit at {exit_price}")     if isLastRow else None
    else:
        (s, limit1, limit2, sl1, sl2,logString) = \
            signalGenerator.checkShortExit(s,row,df,isLastRow, entryPrice,limit1,limit2,sl1,sl2,logString)

    return (s,exit_price,limit1,limit2,sl1,sl2,logString)



def checkSVPExit(s,row,df,isLastRow, entry_price,limit1,limit2,sl1,sl2,logString=''):
    
    if tickerHasLongPosition(row.symbol) :
        (s,entry_price,limit1,limit2,sl1,sl2,logString) = checkSVPLongExit(s,row,df,isLastRow, entry_price,limit1,limit2,sl1,sl2,logString)
    elif tickerHasShortPosition(row.symbol):
        (s,entry_price,limit1,limit2,sl1,sl2,logString) = checkSVPShortExit(s,row,df,isLastRow, entry_price,limit1,limit2,sl1,sl2,logString)
    else:
        print(f"ERROR: checkSVPExit called for ticker without position. Position:{getTickerPosition(row.symbol)}")
        exit(1)
    return (s,entry_price,limit1,limit2,sl1,sl2,logString)

def checkSVPEntry(s,row,df,isLastRow, entry_price,limit1,limit2,sl1,sl2,logString=''):
    if signalGenerator.OkToEnterLong(row):
        (s,entry_price,limit1,limit2,sl1,sl2,logString) = checkSVPLongEntry(s,row,df,isLastRow, entry_price,limit1,limit2,sl1,sl2,logString)
    elif signalGenerator.OkToEnterShort(row):
        (s,entry_price,limit1,limit2,sl1,sl2,logString) = checkSVPShortEntry(s,row,df,isLastRow, entry_price,limit1,limit2,sl1,sl2,logString)
    else: # nan renko_uptrend
        (s,entry_price,logString) = (float('nan'),float('nan'),"No-SVP-TREND")
    return (s,entry_price,limit1,limit2,sl1,sl2,logString)

def limitOrderSanityCheck(s,row,df, isLastRow,entry_price,limit1,limit2,sl1,sl2,logString):
#    if not np.isnan(s):
    return (s,entry_price,limit1,limit2,sl1,sl2,logString)
    #(o,h,l,c,l1,sl) = (row.Open,row.High,row.Low,row['Adj Close'],limit1,sl1)

def followSVP(type, signal, isLastRow, row, df, 
                        last_signal=float('nan'), prev_trade_price=float('nan'),
                        prev_limit1=float('nan'), prev_limit2=float('nan'),
                        prev_sl1=float('nan'), prev_sl2=float('nan')):
    (s,entry_price,entry_price,limit1,limit2,sl1,sl2) = (signal,prev_trade_price,prev_trade_price,prev_limit1,prev_limit2,prev_sl1,prev_sl2)
    logArray = []
    if tickerHasPosition(row.symbol):
        (s,entry_price,limit1,limit2,sl1,sl2,logString) = checkSVPExit(s,row,df, isLastRow,entry_price,limit1,limit2,sl1,sl2)
        # (s,entry_price,limit1,limit2,sl1,sl2,logString) = limitOrderSanityCheck(s,row,df, isLastRow,entry_price,limit1,limit2,sl1,sl2,logString)
    elif type == 1:
        (s,entry_price,limit1,limit2,sl1,sl2,logString) = checkSVPEntry(s,row,df, isLastRow, entry_price,limit1,limit2,sl1,sl2)
    else:
         (s,entry_price,logString) = (float('nan'),float('nan'),"NO-NEW-TRADES")
         
    if signalChanged(s,signal):
        setTickerTrend(row.symbol, s)
    logSignal(logString,signalGenerator.getLogArray(),signal,s,row,type,isLastRow,f"TrdPrice:{entry_price} LIM:{limit1} SL:{sl1} "+signalGenerator.getExtraLogString(), logWithNoSignalChange=True)
    return (s,entry_price, limit1, limit2, sl1, sl2)

def exitTargetOrSL(type, signal, isLastRow, row, df, 
                        last_signal=float('nan'), prev_trade_price=float('nan'),
                        prev_limit1=float('nan'), prev_limit2=float('nan'),
                        prev_sl1=float('nan'), prev_sl2=float('nan')):
    (s, entry_price,entry_price,limit1,limit2,sl1,sl2) = (signal, prev_trade_price,prev_trade_price,prev_limit1,prev_limit2,prev_sl1,prev_sl2)

    if (not tickerHasPosition(row.symbol) and np.isnan(signal)) :
        # return if ticker has no position
        # or if stop loss not configured
        # or if current signal is 1 or -1 or 0
        return (s,entry_price, limit1, limit2, sl1, sl2)
    
    entryTradePrice = getTickerEntryPrice(row.symbol)

    if np.isnan(entryTradePrice):
        return signal,entry_price # likely got position just now, so entry price not yet set
    # We proceed only if tickerHasPosition, and stop loss is configured
    # and current signal is nan 
    
    if row.renko_brick_diff!=0:
        return signal,entry_price # We do not exit if renko brick is not 0
    
    close = row['Adj Close']
    high = row['High']
    low = row['Low']
    opn = row['Open']
    (longTarget,longSL,shortTarget,shortSL) = getTickerRenkoTargetPrices(row)
    #getTickerTargetPrices(row.symbol)
    trade_price = prev_trade_price
    if tickerHasLongPosition(row.symbol):
        target = longTarget
        sl = longSL
        if high >= longTarget: 
            trade_price = longTarget
            s=0 if (cfgIsBackTest or cfgPartialExitPercent==1) else s#Targets are only partial exits in live
            logStr = "LONG-TARGET-HIT"
        elif low <= longSL:
            trade_price = longSL if opn >= longSL else trade_price
            s=0  
            logStr = "LONG-STOP-LOSS-HIT"
    else:
        target = shortTarget
        sl = shortSL
        if low <= shortTarget: 
            trade_price = shortTarget
            s = 0 if (cfgIsBackTest or cfgPartialExitPercent==1) else s #Targets are only partial exits in live
            logStr = "SHORT-TARGET-HIT"
        elif high >= shortSL:
            trade_price = shortSL if opn <= shortSL else trade_price
            s = 0            
            logStr = "SHORT-STOP-LOSS-HIT"

    if signalChanged(s,signal):
        setTickerTrend(row.symbol, s) if tickerIsTrending(row.symbol) else None
        logSignal(logStr,["obvData","adxData","maSlpData"],signal,s,row,type,isLastRow,
                  f'(Entry:{entryTradePrice} Exit:{trade_price} H:{high} L:{low} C:{close} sl:{sl} target:{target} hasLongPos:{tickerHasLongPosition(row.symbol)} hasShortPos:{tickerHasShortPosition(row.symbol)}) ')
    return (s,entry_price, limit1, limit2, sl1, sl2)

######### END OF SIGNAL GENERATION FUNCTIONS #########
def getOverrideSignal(row,ovSignalGenerators, df):
    
    s = row['signal'] #the signal that non-override sig generators generated
    #Could be nan, diff rom getLastSignal, which returns last valid, non-nan,
    #signal in dfs

    #Return nan if its not within trading hours
    if(row.name >= tradingStartTime) & \
        (row.name.time() >= cfgStartTimeOfDay):
            if row.name.time() < cfgEndNewTradesTimeOfDay:
                type = 1 # Entry or Exit
            elif row.name.time() < cfgEndExitTradesOnlyTimeOfDay:
                # Last time period before intraday exit; only exit positions
                # No new psitions will be entered
                type = 0 
            else:
                return 0 # Outside of trading hours EXIT ALL POSITIONS

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

def weShouldTrade(row):
    return (row.name >= tradingStartTime) & \
            (row.name.time() >= cfgStartTimeOfDay) & \
                (row["Adj Close"] > cfgMinPriceForOptions)

def weCanEnterNewTrades(row):
    return row.name.time() < cfgEndNewTradesTimeOfDay and \
        (row["Adj Close"] > cfgMinPriceForOptions+6)

def getSignal(row,signalGenerators, df):
    s = trade_price = limit1 = limit2 = sl1 = sl2 = float("nan")
    isLastRow = (row.name == df.index[-1]) or cfgIsBackTest
    row_time = row.name.time()

    #Return nan if its not within trading hours
    if weShouldTrade(row):
            if weCanEnterNewTrades(row):
                type = 1 # Entry or Exit
            elif row.name.time() < cfgEndExitTradesOnlyTimeOfDay:
                # Last time period before intraday exit; only exit positions
                # No new psitions will be entered
                type = 0 
            else:
                return (0,trade_price, limit1, limit2, sl1, sl2) # Outside of trading hours EXIT ALL POSITIONS

            for sigGen in signalGenerators:
                # these functions can get the signal for *THIS* row, based on the
                # what signal Generators previous to this have done
                # they cannot get or act on signals generated in previous rows
                # signal s from previous signal generations is passed in as an 
                # argument

                result = sigGen(type,s, isLastRow, row, df, prev_trade_price=trade_price,
                                prev_limit1=limit1, prev_limit2=limit2, prev_sl1=sl1, prev_sl2=sl2)
                (s,trade_price, limit1, limit2, sl1, sl2) = result if isinstance(result, tuple) else (result,trade_price, limit1, limit2, sl1, sl2)
                if (trade_price < 0):
                    print(f"trade price({trade_price}) less than 0. Signal:{s}")
                    print(row)
                    exit(0)
                if not np.isnan(s) and not np.isnan(trade_price) and (trade_price*0.95 > row.High or trade_price*1.05 < row.Low):
                    logging.error(f"!!! ERRORR !!!! trade price {trade_price} is outside of high/low range {row.High}/{row.Low}")
                    print(f"!!! ERRORR !!!! trade price {trade_price} is outside of high/low range {row.High}/{row.Low}")
                    print(f"{row.name}:{row.i} => {s}")
                    exit(0)
                # else:
                #     logging.info(f"s:{'-' if np.isnan(s) else s} @ {trade_price}=> {row.High}/{row.Low} - brick:{row.renko_brick_num} : {row.renko_brick_high}/{row.renko_brick_low}; l/sl:{limit1}/{sl1}")

            # logging.info(f"trade price is {trade_price}")
            setTickerPosition(row.symbol, s, row['Adj Close'] if np.isnan(trade_price) else trade_price,\
                limit1, limit2, sl1, sl2)
    else:
        #reset at start of day
        initTickerStatus(row.symbol)
        return (0,trade_price, limit1, limit2, sl1, sl2) # Exit all positions outside of trading hours
    # if isLastRow:
    #     logTickerStatus(row.symbol)
    return (s,trade_price, limit1, limit2, sl1, sl2)

def cacheAnalytics(df):
    fname = f"Data/analyticsCache/{df['symbol'][0]}-{df.index[0]}-{df.index[-1]}.pickle"
    with open(fname,"wb") as f:
        pickle.dump(df,f)

def hasCachedAnalytics(df):
    # return False
    fname = f"Data/analyticsCache/{df['symbol'][0]}-{df.index[0]}-{df.index[-1]}.pickle"
    if os.path.exists(fname):
        return True
    else:
        return False
def getCachedAnalytics(df):
    fname = f"Data/analyticsCache/{df['symbol'][0]}-{df.index[0]}-{df.index[-1]}.pickle"
    with open(fname, "rb") as f:
        df = pickle.load(f)
    return df
## MAIN APPLY STRATEGY FUNCTION
def applyIntraDayStrategy(df,analyticsGenerators=[populateBB], signalGenerators=[followRenkoWithOBV],
                        overrideSignalGenerators=[],tradeStartTime=None, \
                        applyTickerSpecificConfig=True):
    global tradingStartTime
    
    if applyTickerSpecificConfig:
        applyTickerSpecificCfg(df['symbol'][0]) 
        #printCFG()
    
    tradingStartTime = tradeStartTime
    if tradingStartTime is None:
        tradingStartTime = datetime.datetime(2000,1,1,10,0,0) #Long ago :-)
        tradingStartTime = ist.localize(tradingStartTime)

    if hasCachedAnalytics(df):
        df = getCachedAnalytics(df)
    else: 
        df['signal'] = float("nan")
      
        dfWithAnalytics = generateAnalytics(analyticsGenerators,df)
                
        # Select the columns that are present in dfWithAnalytics but not in df
        new_cols = [col for col in dfWithAnalytics.columns if col not in df.columns]
        # Copy the new columns from dfWithAnalytics to df   
        for col in new_cols:
            df[col] = dfWithAnalytics[col]

        #cache df with analytics for future
        cacheAnalytics(df)
    initTickerStatus(df['symbol'][0])
    # df['SuperTrendDirection'] = dfWithAnalytics['SuperTrendDirection']
    # df['ma_superTrend_pct_change'] = dfWithAnalytics['ma_superTrend_pct_change']
    # apply the condition function to each row of the DataFrame
    # these functions can get the signal for *THIS* row, based on the
    # what signal Generators previous to this have done
    # they cannot get or act on signals generated in previous rows
    #(df['signal'],df['trade_price']) 
    x = df.apply(getSignal, 
        args=(signalGenerators, df), axis=1)
    #print(x)
    
    (df['signal'],df['trade_price'], df['limit1'], df['limit2'], df['sl1'], df['sl2']) = zip(*x)
    df['Open'] = np.where(df['trade_price'].shift(1).notnull(), df['trade_price'].shift(1), df['Open'])
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

    return df