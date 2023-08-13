#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  5 14:26:36 2023

@author: nikhilsama
"""
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
import tickerCfg
import utils
import random
import talib as ta
import analytics
from itertools import compress
from candlerankings import candle_rankings
from signal_generators.SignalGenerator import SignalGenerator
from signal_generators.svb import svb
from signal_generators.renko import Renko
from signal_generators.renkoburst import RenkoBurst
from signal_generators.meanrev import MeanRev
from signal_generators.supertrend import SuperTrend
from signal_generators.voldelta import VolDelta
from signal_generators.burstfinder import BurstFinder
from signal_generators.bolingerband import BollingerBand

# signalGenerator = svb()
#cfg has all the config parameters make them all globals here
import cfg
globals().update(vars(cfg))

# set timezone to IST
ist = pytz.timezone('Asia/Kolkata')

tickerStatus = {}
def getSignalGenerator(row):
    # return VolDelta()
    # return SuperTrend()
    # return MeanRev()
    # return BurstFinder()
    # return svb()
    # return BollingerBand() ##-- WORKS, good sharp ratio
    # return RenkoBurst()

    sigGen = SignalGenerator()
    
    obImb = sigGen.getOrderBookImbalance(row)
    if obImb != 0 or True:        
        return Renko(limitExitOrders=True,limitEntryOrders=True,
                    slEntryOrders=True,slExitOrders=True,
                    exitStaticBricks=False,useSVPForEntryExitPrices=False,useVolDelta=True)
    else:
        return svb()
    
def updateCFG(ma_slope_thresh, ma_slope_thresh_yellow_multiplier, \
            obv_osc_thresh, \
            obv_osc_thresh_yellow_multiplier, \
            obv_ma_len, ma_slope_period, 
            RenkoBrickMultiplier, atrlen, \
            RenkoBrickMultiplierLongTarget, RenkoBrickMultiplierLongSL, \
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
                            'prevPosition': float("nan"),
                            'trendSignal':float("nan"),
                            'entry_price':float("nan"),
                            'max_price':getDefaultTickerMaxPrice(),
                            'min_price':getDefaultTickerMinPrice(),
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
def getTickerPrevPosition(ticker):
    if ticker not in tickerStatus:
        return float ('nan')
    elif 'prevPosition' not in tickerStatus[ticker]:
        return float ('nan')
    else:
        return tickerStatus[ticker]['prevPosition']
def tickerHasPosition(ticker):
    return (not np.isnan(getTickerPosition(ticker))
                and getTickerPosition(ticker) != 0)
def tickerHasLongPosition(ticker):
    return getTickerPosition(ticker) == 1
def tickerHasShortPosition(ticker):
    return getTickerPosition(ticker) == -1
def getDefaultTickerMaxPrice():
    return int(0)
def getDefaultTickerMinPrice():
    return int(10000000)
def setTickerMaxPriceForTrade(ticker,entry_price,high,low,signal):
    default_min = getDefaultTickerMinPrice()
    default_max = getDefaultTickerMaxPrice()
    if 'max_price' not in tickerStatus[ticker]:
        tickerStatus[ticker]['max_price'] = default_max
        tickerStatus[ticker]['min_price'] = default_min
    if np.isnan(signal):
        tickerStatus[ticker]['max_price'] = round(max(abs(high),tickerStatus[ticker]['max_price']))
        tickerStatus[ticker]['min_price'] = round(min(abs(low),tickerStatus[ticker]['min_price']))
    else: # signal is 1/-1/0, start or end of a new trade
        tickerStatus[ticker]['max_price'] = tickerStatus[ticker]['min_price'] = round(abs(entry_price),1)
    
def setTickerPosition(ticker,signal, entry_price,high, low, limit1, limit2, sl1, sl2):
    # logging.info(f"setting ticker position for {ticker} to {signal} entry{entry_price} pos:{tickerStatus[ticker]['position']}")
    setTickerMaxPriceForTrade(ticker,entry_price,high,low,signal)
    (tickerStatus[ticker]['limit1'], tickerStatus[ticker]['limit2'], tickerStatus[ticker]['sl1'], tickerStatus[ticker]['sl2']) = (limit1, limit2, sl1, sl2)
    #logging.info(f"signal: {signal} entry: {tickerStatus[ticker]['entry_price']} max: {tickerStatus[ticker]['max_price']} min: {tickerStatus[ticker]['min_price']}")
    if np.isnan(signal):
        return # nan signal does not change position
    if ticker not in tickerStatus:
        tickerStatus[ticker] = {}
    if tickerStatus[ticker]['position'] != signal:
        if tickerStatus[ticker]['position'] != 0:
            #Store last long or short position
            tickerStatus[ticker]['prevPosition'] = tickerStatus[ticker]['position']
        tickerStatus[ticker]['position'] = signal
        if signal == 0:
            tickerStatus[ticker]['entry_price'] = float('nan')
        else:
            tickerStatus[ticker]['entry_price'] = round(entry_price,1)
            # logging.info(f"setting entry price for {ticker} to {entry_price}")
def getTickerEntryPrice(ticker):
    return round(abs(tickerStatus[ticker]['entry_price'])) if not np.isnan(tickerStatus[ticker]['entry_price']) else 0
def getTickerMaxPrice(ticker):
    return abs(tickerStatus[ticker]['max_price'])
def getTickerMinPrice(ticker):
    return abs(tickerStatus[ticker]['min_price'])
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

def logOHLV(row):
    return f"OHLV O:{round(row['Open'],1)} H:{round(row['High'],1)} L:{round(row['Low'],1)} | "
def logADX(row):
    return f"ADX:{round(row['ADX'],1)} > {adxThresh}(*{round(adxThreshYellowMultiplier,1)}) adxSLP:{round(row['ADX-PCT-CHNG'],2)}*{numCandlesForSlopeProjection} | " \
        if "ADX" in row else 'No ADX Data'
def logMASLP(row):
    return f"maSlp:{round(row['SLOPE-OSC'],2)} >= {maSlopeThresh}(*{maSlopeThreshYellowMultiplier}) maSlpChng:{round(row['SLOPE-OSC-SLOPE'],2)}>*{numCandlesForSlopeProjection} | " \
        if "SLOPE-OSC" in row else 'No Slope Data'
def logOBV(row):
    return f"OBV:{round(row['OBV-OSC'],2)} > {obvOscThresh}(*{obvOscThreshYellowMultiplier}) obvSLP:{round(row['OBV-OSC-PCT-CHNG'],2)}>*{numCandlesForSlopeProjection} | " \
        if "OBV-OSC" in row else 'No Volume Data',
def logRenko(row):
    return f"Renko Trend:{'↑' if row['renko_uptrend'] else '↓'}:{round(row['renko_brick_num'])}({round(row['renko_brick_diff'])}) V:{round(row['renko_brick_volume_osc'],1)} StC:{round(row['renko_static_candles']) if not np.isnan(row['renko_static_candles']) else 'nan'} L:{round(row['renko_brick_low'],1)} H:{round(row['renko_brick_high'],1)} | " \
        if "renko_uptrend" in row and not np.isnan(row.renko_brick_num) else 'No Renko Data'
def logSVP(row):
    return f"SVP {round(row['dayHigh'])}-{round(row['vah'])}({round(row['slpVah'],1)})-{round(row['poc'])}({round(row['slpPoc'],1)})-{round(row['val'])}({round(row['slpVal'],1)})-{round(row['dayLow'])}  | " \
        if "poc" in row and not np.isnan(row['poc']) and not np.isnan(row['slpVah']) and not np.isnan(row['slpPoc']) and not np.isnan(row['slpVal'] \
            and not np.isnan(row.dayHigh) and not np.isnan(row.dayLow) and not np.isnan(row.vah) and not np.isnan(row.poc) and not np.isnan(row.val)) else ''
def logSVPST(row):
    return f"SVP-ST {round(row['ShrtTrmHigh'],1)}-{round(row['vahShrtTrm'],1)}({round(row.slpSTVah,1)})-{round(row['pocShrtTrm'],1)}-{round(row['valShrtTrm'],1)}({round(row.slpSTVal,1)})-{round(row['ShrtTrmLow'],1)} | " \
        if "poc" in row else 'No SVP Data'
def logCandlestick(row):
    return f"Candlestick {row['candlestick_pattern']} | CndlCnt {row.candlestick_match_count} | CndlSignal {row.candlestick_signal}" \
        if "candlestick_pattern" in row else 'No Candlestick Data'
def logSuperTrend(row):
    if 'SuperTrend' in row:
        if np.isnan(row.SuperTrendLower):
            st = f"↓{round(row.SuperTrendUpper)}" if not np.isnan(row.SuperTrendUpper) else 'nan'
        else:
            st = f"↑{round(row.SuperTrendLower)}"
        return f"SuperTrend {st}" 
    else:
        return 'No SuperTrend Data'
def logRSI(row):
    if 'RSI' in row and not np.isnan(row.RSI):
        return f"RSI {round(row.RSI)} | "
    else:
        return ''
def rndLog(n):
    n = float(n)
    if np.isnan(n):
        return n
    elif abs(n) > 1000000:
        return f"{round(n/1000000)}M"
    elif abs(n) > 1000:
        return f"{round(n/1000)}K"
    else:
        return f"{round(n)}"
def logVolDelta(row):
    if 'volDelta' in row:
        niftyVolDelta = rndLog(row.niftyUpVol-row.niftyDnVol)
        futVolDelta = rndLog(row.niftyFutureUpVol-row.niftyFutureDnVol)
        
        str = ''
        str += f"niftyVolDelta {niftyVolDelta} " if niftyVolDelta != '0' else ''
        str += f"futVolDelta: {futVolDelta} | " if futVolDelta != '0' else ''
        
        fullOderImbalance = f"{round(row.futOrderBookBuyQt-row.futOrderBookSellQt)}({round(row.futOrderBookBuyQt/row.futOrderBookSellQt, 1) if row.futOrderBookSellQt!=0 else 'nan'})" if (not np.isnan(row.futOrderBookBuyQt)) and (not np.isnan(row.futOrderBookSellQt)) else 'nan'
        stOderImbalance = f"{round(row.futOrderBookBuyQtLevel1-row.futOrderBookSellQtLevel1)}({round(row.futOrderBookBuyQtLevel1/row.futOrderBookSellQtLevel1,1) if row.futOrderBookSellQtLevel1 != 0 else 'nan'})" if (not np.isnan(row.futOrderBookBuyQtLevel1)) and (not np.isnan(row.futOrderBookSellQtLevel1)) else 'nan'
        # str += f"fullOrderImbalance: {round(row.futOrderBookBuyQt/row.futOrderBookSellQt, 1) if (not np.isnan(row.futOrderBookSellQt)) and row.futOrderBookSellQt!=0 else 'nan'} |" 
        # str += f"stOrderImbalance: {round(row.futOrderBookBuyQtLevel1/row.futOrderBookSellQtLevel1,1) if row.futOrderBookSellQtLevel1 != 0 else 'nan'} | "
        str += f"OderImbalance F:{fullOderImbalance} | S:{stOderImbalance} | "
        # str= f"VolDelta {rndLog(row.volDelta)} VolThresh: {rndLog(row.volDeltaThreshold)} Cum: {rndLog(row.cumVolDelta)} Max: {rndLog(row.maxVolDelta)} Min: {rndLog(row.minVolDelta)} | "
        # str+=f"stOrderImbalance: {round(row.obSTImabalance) if not np.isnan(row.obSTImabalance) else 'nan'}({round(row.volDeltaRatio2,1)  if not np.isnan(row.volDeltaRatio2) else 'nan'}) | fulldOrderImbalance: {rndLog(row.obImabalance)}({round(row.volDeltaRatio1,1)})) | " if 'obSTImabalance' in row else ''
    else:
        return ''
    return str
    
def logBB(row):
    str = ''
    if 'upper_band' in row and not np.isnan(row.upper_band):
        str = f"BB {round(row.upper_band)}/{round(row.ma20)}/{round(row.lower_band)} | "
    return str
def fastMA(row):
    str = '' 
    if 'MA-FAST' in row and not np.isnan(row['MA-FAST']) and not np.isnan(row['MA-FAST-SLP']):
        str = f"FastMA {round(row['MA-FAST'])} ({row['MA-FAST-SLP']})| "
    return str
def logSignal(msg,reqData,signal,s,row,window,isLastRow,extra='',logWithNoSignalChange=False):
    # return
    rowInfo = f'{row.symbol}:{row.i} '
    rowPrice = f'p:{round(row["Adj Close"],1)} '
    sigInfo = f'sig:{"-" if np.isnan(signal) else signal} s:{"-" if np.isnan(s) else s} {"E" if window == 1 else "X"} '
    tradeInfo = f"Pos/Prev:{getTickerPosition(row.symbol)}{getTickerPrevPosition(row['symbol'])} Entry:{getTickerEntryPrice(row['symbol'])} Max:{getTickerMaxPrice(row['symbol'])} Min:{getTickerMinPrice(row['symbol'])}" 
    # \
    #     if int(getTickerMaxPrice(row['symbol'])) != getDefaultTickerMaxPrice() and \
    #         int(getTickerMinPrice(row['symbol'])) != getDefaultTickerMinPrice() else ''
    dataStrings = {
        "ohlv": logOHLV(row),
        "adxData" : logADX(row),
        "maSlpData" : logMASLP(row),
        "obvData" : logOBV(row),
        "RenkoData": logRenko(row),
        "svp": logSVP(row),
        "svpST": logSVPST(row),
        "candlestick": logCandlestick(row),
        "supertrend": logSuperTrend(row),
        "voldelta": logVolDelta(row),
        "rsi": logRSI(row),
        'bb': logBB(row),
        'fastma': fastMA(row)
    }
    dataString = ''
    for key in reqData:
        dataString = dataString+dataStrings[key]
    if isLastRow and (signalChanged(s,signal) or logWithNoSignalChange):
        logging.info(rowInfo+' => '+rowPrice+' => '+msg+' '+tradeInfo+' '+' '+extra+' '+sigInfo+ ' => '+dataString)
    elif signalChanged(s,signal) or logWithNoSignalChange:
        rowTime = row.name.strftime("%d/%m %I:%M")
        rowInfo = rowInfo+f':{rowTime} ' #backtest needs date
        logging.debug(rowInfo+' => '+rowPrice+' => '+msg+' '+tradeInfo+' '+' '+extra+' '+sigInfo+ ' => '+dataString)

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
        logging.error(f"ERROR: checkRenkoLongEntry called with position or without uptrend.  Trend:{row['renko_uptrend']} Position:{tickerHasPosition(row.symbol)}")
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
        errString = f"ERROR: checkRenkoShortEntry called with position or without downtrend.  Trend:{row['renko_uptrend']} Position:{tickerHasPosition(row.symbol)}"
        print(errString)
        logging.erorr(errString)
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
        logging.error(f"ERROR: checkRenkoLongExit called without long position")
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
        logging.error(f"ERROR: checkRenkoShortExit called without Short position")
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
        logging.error(f"ERROR: checkRenkoExit called for ticker without position. Position:{getTickerPosition(row.symbol)}")
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
        (s,entry_price,limit1,limit2,sl1,sl2,logString) = checkRenkoExit(s,row,df, isLastRow, entry_price,limit1,limit2,sl1,sl2)
    elif type == 1:
        (s,entry_price,limit1,limit2,sl1,sl2,logString) = checkRenkoEntry(s,row,df, isLastRow, entry_price,limit1,limit2,sl1,sl2)
    else:
         (s,entry_price,logString) = (float('nan'),float('nan'),"RENKO-NO-NEW-ENTRY")
    if signalChanged(s,signal):
        setTickerTrend(row.symbol, s)
    logSignal(logString,['RenkoData','ohlv','svp'],signal,s,row,type,isLastRow,f'TrdPrice:{entry_price} LIM:{limit1} SL:{sl1}',logWithNoSignalChange=True)
    return (s,entry_price, limit1, limit2, sl1, sl2)

longSLExits = shortSLExits= longLimitExits = shortLimitExits = longSLEntrys = shortSLEntrys= longLimitEntrys= shortLimitEntrys = 0
def printOrderCountStats():
    global longSLExits,longLimitExits,shortSLExits,shortLimitExits,shortSLEntrys,shortLimitEntrys,longSLEntrys,longLimitEntrys
    print(f"LongSLExits:{longSLExits} LongLimitExits:{longLimitExits} ShortSLExits:{shortSLExits} ShortLimitExits:{shortLimitExits}")
    print(f"shortSLEntrys{shortSLEntrys} shortLimitEntrys{shortLimitEntrys} longSLEntrys{longSLEntrys} longimitEntrys{longLimitEntrys}")
def checkLongExits(s,row,df,isLastRow,limit1,limit2,sl1,sl2,logString=''):
    global longSLExits,longLimitExits
    prevSL1 = getTickerSL1(row.symbol) if isShortSL1Order(row.symbol) else float('nan')
    prevTarget = getTickerLimit1(row.symbol) if isShortLimit1Order(row.symbol) else float('nan')
    exit_price = float('nan')
    
    if row.Low <= prevSL1: #SL
        longSLExits+=1
        s = 0
        exit_price = min(utils.priceWithSlippage(prevSL1,'longExit'),row.Open)
        logString = "LONG-EXIT-SL"
        logging.info(f"Stop Loss order hit at {prevSL1}. Next row should reflect exit.")  if isLastRow else None
        
    elif row.High >= prevTarget:#Target
        longLimitExits+=1
        s = 0
        exit_price = max(prevTarget,row.Open)
        logString = "LONG-EXIT-LMT"
        logging.info(f"Target hit at {exit_price}") if isLastRow else None
    return (s,exit_price,logString)

def checkShortExits(s,row,df,isLastRow,limit1,limit2,sl1,sl2,logString=''):
    global shortSLExits,shortLimitExits

    prevSL1 = getTickerSL1(row.symbol) if isLongSL1Order(row.symbol) else float('nan')
    prevTarget = getTickerLimit1(row.symbol) if isLongLimit1Order(row.symbol) else float('nan')
    exit_price = float('nan')
    
    if row.High >= prevSL1:
        shortSLExits+=1
        s = 0
        exit_price = max(utils.priceWithSlippage(prevSL1,'shortExit'),row.Open) if prevSL1>=row.Low else row.Open
        logString = "SHORT-EXIT-SL"
        logging.info(f"Stop Loss order hit at {exit_price}. Next row should reflect exit.") if isLastRow else None
    elif row.Low <= prevTarget:
        shortLimitExits+=1
        s = 0
        exit_price = min(prevTarget,row.Open)
        logString = "SHORT-EXIT-LMT"
        logging.info(f"Short Target hit at {exit_price}")     if isLastRow else None
    return (s,exit_price,logString)

def checkPrevOrderEntry(s,row,df,isLastRow,limit1,limit2,sl1,sl2,logString=''):
    global shortSLEntrys,shortLimitEntrys,longSLEntrys,longLimitEntrys

    prevSL1 = getTickerSL1(row.symbol)  
    prevTarget = getTickerLimit1(row.symbol) 
    entry_price = float('nan')
    
    if isLongLimit1Order(row.symbol) and row.Low <= prevTarget:
        longLimitEntrys+=1
        s = 1
        entry_price = min(row.Open,prevTarget)
        # if row.High < prevTarget:
        #     print(f"ERROR: {row.name} checkSVPLongEntry: How can row high {row.High} be less than limit entry {prevTarget}")
        #     logging.error(f"ERROR: {row.name} checkSVPLongEntry: How can row high {row.High} be less than limit entry {prevTarget}")
            #possible I guess because our limit order was not actually there, this is just a simulation backtest, if it was there then row high would have taken it out
            #exit(0)
        logString = "LONG-ENTRY-LMT"
        logging.info(f"Target Limit order hit at {entry_price}.")  if isLastRow else None
    elif isShortLimit1Order(row.symbol) and row.High >= prevTarget:
        shortLimitEntrys+=1
        s = -1
        entry_price = -max(row.Open,prevTarget)
        logString = "SHORT-ENTRY-LMT"
        logging.info(f"Short Target Limit order hit at {entry_price}.") if isLastRow else None
    elif isShortSL1Order(row.symbol) and row.Low <= prevSL1:
        shortSLEntrys+=1
        s = -1
        entry_price = -min(utils.priceWithSlippage(prevSL1,'shortEntry'),row.Open)
        entry_price = -utils.priceWithSlippage(prevSL1,'shortEntry')
        logString = "SHORT-ENTRY-SL"
        logging.info(f"SL order hit at {entry_price}.") if isLastRow else None

    elif isLongSL1Order(row.symbol) and row.High >= prevSL1:
        longSLEntrys+=1
        s = 1
        entry_price = utils.priceWithSlippage(prevSL1,'longEntry')
        # entry_price = max(utils.priceWithSlippage(prevSL1,'longEntry'),row.Open)
        # if row.Open - prevSL1 > 2:
        #     print(f"ERROR: {row.name} checkSVPLongEntry: Entry price {row.Open} is more than 2 away from SL {prevSL1}")
        #     logging.error(f"ERROR: {row.name} checkSVPLongEntry: Entry price {row.Open} is more than 2 away from SL {prevSL1}")
            # exit(0)
        logString = "LONG-ENTRY-SL"
        logging.info(f"SL order hit at {entry_price}.")  if isLastRow else None
            
    return (s,entry_price,logString)

def checkOrderStatus(s,row,df,isLastRow,limit1,limit2,sl1,sl2):
    signal = s
    (entry_price,exit_price) = (float('nan'),float('nan'))
    
    if tickerHasPosition(row.symbol):
        if tickerHasLongPosition(row.symbol):
            (s,exit_price,logString) =\
                checkLongExits(s,row,df,isLastRow,limit1,limit2,sl1,sl2)
        elif tickerHasShortPosition(row.symbol):
            (s,exit_price,logString) =\
                checkShortExits(s,row,df,isLastRow,limit1,limit2,sl1,sl2)
        else:
            logging.error(f"ERROR: checkPrevOrderEntry called for ticker with wierd positions {getTickerPosition(row.symbol)}.")
    else: # Entry0    
        (s,entry_price,logString) =\
            checkPrevOrderEntry(s,row,df,isLastRow,limit1,limit2,sl1,sl2)
    
    if signalChanged(s,signal):
        logging.debug(f'Signal changed from {signal} to {s} at {row.name}. Position: {getTickerPosition(row.symbol)}')
        setTickerTrend(row.symbol, s)
        setTickerPosition(row.symbol, s, row['Adj Close'] if np.isnan(entry_price) else entry_price,\
            row.High, row.Low, limit1, limit2, sl1, sl2)

    logSignal(logString,getSignalGenerator(row).getLogArray(),signal,s,row,type,isLastRow,f"TrdPrice:{entry_price} LIM:{round(limit1,1)} SL:{round(sl1,1)} "+getSignalGenerator(row).getExtraLogString(), logWithNoSignalChange=False)
    return (s,entry_price,exit_price)


def checkSVPShortEntry(s,row,df,isLastRow, entry_price,limit1,limit2,sl1,sl2,logString):
    logString = 'WAITING-FOR-SHORT-ENTRY'

    (s, limit1, limit2, sl1, sl2,logString) = \
        getSignalGenerator(row).checkShortEntry(s,row,df,getTickerPrevPosition(row.symbol),getTickerMaxPrice(row.symbol),
                                            getTickerMinPrice(row.symbol),isLastRow,limit1,limit2,sl1,sl2,logString)

    return (s, limit1, limit2, sl1, sl2,logString)

def checkSVPLongEntry(s,row,df,isLastRow, entry_price,limit1,limit2,sl1,sl2,logString):
    logString = 'WAITING-FOR-LONG-ENTRY'
    (s, limit1, limit2, sl1, sl2,logString) = \
        getSignalGenerator(row).checkLongEntry(s,row,df,getTickerPrevPosition(row.symbol),getTickerMaxPrice(row.symbol),
                                            getTickerMinPrice(row.symbol),isLastRow,limit1,limit2,sl1,sl2,logString)
    return (s, limit1, limit2, sl1, sl2,logString)

def checkSVPLongExit(s,row,df,isLastRow,limit1,limit2,sl1,sl2,logString):
    if not tickerHasLongPosition(row.symbol):
        print(f"ERROR: checkRenkoLongExit called without long position")
        logging.error(f"ERROR: checkRenkoLongExit called without long position")
        exit(1)
    logString = "LONG-CONTINUE"
    entryPrice = getTickerEntryPrice(row.symbol)    
    
    (s, limit1, limit2, sl1, sl2,logString) = \
        getSignalGenerator(row).checkLongExit(s,row,df,isLastRow, entryPrice,limit1,limit2,sl1,sl2,logString,
                                            getTickerEntryPrice(row.symbol),getTickerMaxPrice(row.symbol),
                                            getTickerMinPrice(row.symbol))

    return (s,limit1,limit2,sl1,sl2,logString)

def checkSVPShortExit(s,row,df,isLastRow,limit1,limit2,sl1,sl2,logString):
    if not tickerHasShortPosition(row.symbol):
        print(f"ERROR: checkRenkoShortExit called without Short position")
        logging.error(f"ERROR: checkRenkoShortExit called without Short position")
        exit(1)
    logString = "SHORT-CONTINUE"
    entryPrice = getTickerEntryPrice(row.symbol)
    
    (s, limit1, limit2, sl1, sl2,logString) = \
        getSignalGenerator(row).checkShortExit(s,row,df,isLastRow, entryPrice,limit1,limit2,sl1,sl2,logString,
                                            getTickerEntryPrice(row.symbol),getTickerMaxPrice(row.symbol),
                                            getTickerMinPrice(row.symbol))

    return (s,limit1,limit2,sl1,sl2,logString)

def checkSVPExit(s,row,df,isLastRow,limit1,limit2,sl1,sl2,logString=''):
    
    if tickerHasLongPosition(row.symbol) :
        (s,limit1,limit2,sl1,sl2,logString) = checkSVPLongExit(s,row,df,isLastRow,limit1,limit2,sl1,sl2,logString)

    elif tickerHasShortPosition(row.symbol):
        (s,limit1,limit2,sl1,sl2,logString) = checkSVPShortExit(s,row,df,isLastRow,limit1,limit2,sl1,sl2,logString)
    else:
        print(f"ERROR: checkSVPExit called for ticker without position. Position:{getTickerPosition(row.symbol)}")
        logging.error(f"ERROR: checkSVPExit called for ticker without position. Position:{getTickerPosition(row.symbol)}")
        exit(1)
    return (s,limit1,limit2,sl1,sl2,logString)

def checkSVPEntry(s,row,df,isLastRow, entry_price,limit1,limit2,sl1,sl2,logString=''):
    if getSignalGenerator(row).OkToEnterLong(row):
        (s,limit1,limit2,sl1,sl2,logString) = checkSVPLongEntry(s,row,df,isLastRow, entry_price,limit1,limit2,sl1,sl2,logString)
    elif getSignalGenerator(row).OkToEnterShort(row):
        (s,limit1,limit2,sl1,sl2,logString) = checkSVPShortEntry(s,row,df,isLastRow, entry_price,limit1,limit2,sl1,sl2,logString)
    else: # nan renko_uptrend
        (logString) = ("No-SVP-TREND")
        # (s,entry_price,limit1,limit2,sl1,sl2,logString) = checkPrevOrderEntry(s,row,df,isLastRow, entry_price,limit1,limit2,sl1,sl2,logString)

    return (s,limit1,limit2,sl1,sl2,logString)

def limitOrderSanityCheck(s,row,df, isLastRow,entry_price,limit1,limit2,sl1,sl2,logString):
#    if not np.isnan(s):
    return (s,entry_price,limit1,limit2,sl1,sl2,logString)
    #(o,h,l,c,l1,sl) = (row.Open,row.High,row.Low,row['Adj Close'],limit1,sl1)

def followSVP(type, signal, isLastRow, row, df, 
                        last_signal=float('nan'), prev_trade_price=float('nan'),
                        prev_limit1=float('nan'), prev_limit2=float('nan'),
                        prev_sl1=float('nan'), prev_sl2=float('nan')):
    (s,entry_price,limit1,limit2,sl1,sl2) = (signal,prev_trade_price,prev_limit1,prev_limit2,prev_sl1,prev_sl2)
    logArray = []
    
    # Check if any of our old limit/sl orders have executed, if so, update position accordingly
    (s,entry_price, exit_price) = checkOrderStatus(s,row,df,isLastRow,limit1,limit2,sl1,sl2)

    if tickerHasPosition(row.symbol):
        (s,limit1,limit2,sl1,sl2,logString) = checkSVPExit(s,row,df, isLastRow,limit1,limit2,sl1,sl2)
        # (s,entry_price,limit1,limit2,sl1,sl2,logString) = limitOrderSanityCheck(s,row,df, isLastRow,entry_price,limit1,limit2,sl1,sl2,logString)
    elif type == 1:
        (s,limit1,limit2,sl1,sl2,logString) = checkSVPEntry(s,row,df, isLastRow, entry_price,limit1,limit2,sl1,sl2)
    else:
         (logString) = ("NO-NEW-TRADES")
    
    if signalChanged(s,signal):
        setTickerTrend(row.symbol, s)
        
    logSignal(logString,getSignalGenerator(row).getLogArray(),signal,s,row,type,isLastRow,f"TrdPrice:{entry_price} LIM:{round(limit1,1)} SL:{round(sl1,1)} "+getSignalGenerator(row).getExtraLogString(), logWithNoSignalChange=True)
    return (s,entry_price, exit_price, limit1, limit2, sl1, sl2)

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
    s = entry_price = exit_price = limit1 = limit2 = sl1 = sl2 = float("nan")
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
                return (0,entry_price, exit_price, limit1, limit2, sl1, sl2) # Outside of trading hours EXIT ALL POSITIONS

            for sigGen in signalGenerators:
                # these functions can get the signal for *THIS* row, based on the
                # what signal Generators previous to this have done
                # they cannot get or act on signals generated in previous rows
                # signal s from previous signal generations is passed in as an 
                # argument

                result = sigGen(type,s, isLastRow, row, df, prev_trade_price=entry_price,
                                prev_limit1=limit1, prev_limit2=limit2, prev_sl1=sl1, prev_sl2=sl2)
                (s,entry_price, exit_price, limit1, limit2, sl1, sl2) = result if isinstance(result, tuple) else (result,entry_price, limit1, limit2, sl1, sl2)
                # if (entry_price < 0 or exit_price < 0 ):
                #     print(f"trade price({entry_price} or ({exit_price})) less than 0. Signal:{s}")
                #     print(row)
                #     logging.error(f"trade price({entry_price} or ({exit_price}) less than 0. Signal:{s}")
                #     logging.error(row)
                #     exit(0)
                if not np.isnan(s) and not np.isnan(entry_price) and (abs(entry_price)*0.95 > row.High or abs(entry_price)*1.05 < row.Low):
                    logging.error(f"!!! ERRORR !!!! entry price {entry_price} is outside of high/low range {row.High}/{row.Low}")
                    print(f"!!! ERRORR !!!! entry price {entry_price} is outside of high/low range {row.High}/{row.Low}")
                    print(f"{row.name}:{row.i} => {s}")
                    exit(0)
                if not np.isnan(s) and not np.isnan(exit_price) and (exit_price*0.95 > row.High or exit_price*1.05 < row.Low):
                    logging.error(f"!!! ERRORR !!!! exit price {exit_price} is outside of high/low range {row.High}/{row.Low}")
                    print(f"!!! ERRORR !!!! exit price {exit_price} is outside of high/low range {row.High}/{row.Low}")
                    print(f"{row.name}:{row.i} => {s}")
                    exit(0)

                # else:
                #     logging.info(f"s:{'-' if np.isnan(s) else s} @ {trade_price}=> {row.High}/{row.Low} - brick:{row.renko_brick_num} : {row.renko_brick_high}/{row.renko_brick_low}; l/sl:{limit1}/{sl1}")

            # logging.info(f"trade price is {trade_price}")
            setTickerPosition(row.symbol, s, row['Adj Close'] if np.isnan(entry_price) else entry_price,\
                row.High, row.Low, limit1, limit2, sl1, sl2)
    else:
        #reset at start of day
        initTickerStatus(row.symbol)
        return (0,entry_price, exit_price,limit1, limit2, sl1, sl2) # Exit all positions outside of trading hours
    # if isLastRow:
    #     logTickerStatus(row.symbol)
    return (s,entry_price, exit_price, limit1, limit2, sl1, sl2)



## MAIN APPLY STRATEGY FUNCTION
def applyIntraDayStrategy(df,analyticsGenerators, signalGenerators,
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
      
    if analytics.hasCachedAnalytics(df):
        df = analytics.getCachedAnalytics(df)
    else:
        df['signal'] = float("nan")
        dfWithAnalytics = analytics.generateAnalytics(analyticsGenerators,df)
                        
        # Select the columns that are present in dfWithAnalytics but not in df
        new_cols = [col for col in dfWithAnalytics.columns if col not in df.columns]
        # Copy the new columns from dfWithAnalytics to df   
        for col in new_cols:
            df[col] = dfWithAnalytics[col]

        #cache df with analytics for future
        analytics.cacheAnalytics(df)


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
    
    (df['signal'],df['entry_price'], df['exit_price'],df['limit1'], df['limit2'], df['sl1'], df['sl2']) = zip(*x)
    
    # df['Open'] = np.where(df['entry_price'].shift(1).notnull(), df['entry_price'].shift(1), df['Open'])
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
        
    printOrderCountStats()

    return df

#### LFT Strategy
def applyLFTStrategy(df,analyticsGenerators, signalGenerators):
    df['signal'] = float("nan")
    
    dfWithAnalytics = generateAnalytics(analyticsGenerators,df)
            
    # Select the columns that are present in dfWithAnalytics but not in df
    new_cols = [col for col in dfWithAnalytics.columns if col not in df.columns]
    # Copy the new columns from dfWithAnalytics to df   
    for col in new_cols:
        df[col] = dfWithAnalytics[col]
    
    initTickerStatus(df['symbol'][0])

    (df['signal'],df['entry_price'], df['exit_price'],df['limit1'], df['limit2'], df['sl1'], df['sl2']) = (float('nan'),float('nan'),float('nan'),float('nan'),float('nan'),float('nan'),float('nan'))
    return df
