#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 12:06:54 2023

@author: nikhilsama
"""
#cfg has all the config parameters make them all globals here
import numpy as np
import pandas as pd
import logging
import cfg
from datetime import datetime, time, timedelta

globals().update(vars(cfg))

class SignalGenerator:
    logArray = []
    extraLogString = ''
    def __init__(self, ):
        pass
    def timeWindow(self,row):
        return True
    def OkToEnterLong(self,row):
        return True
    def OkToEnterShort(self,row):
        return True
    def getLogArray(self):
        return self.logArray
    def getExtraLogString(self):
        return self.extraLogString
    #Helpers
    def crossOver(self,fast,slow,lastFast=None,lastSlow=None):
        return fast > slow and \
            (lastFast is None or (lastFast <= lastSlow))
    def crossUnder(self,fast,slow,lastFast=None,lastSlow=None):
        return fast < slow and \
            (lastFast is None or (lastFast >= lastSlow))
            
    def getPCR(self,row):
        return 1
    def getOrderBookImbalance(self,row):
        if ('futOrderBookBuyQt' in row):
            obImbalanceRatio = row.futOrderBookBuyQt/row.futOrderBookSellQt if row.futOrderBookSellQt > 0 else 10
            obImbalance = row.futOrderBookBuyQt - row.futOrderBookSellQt
        else:
            obImbalanceRatio = 1
            obImbalance = 0
        if obImbalance > 10000 and obImbalanceRatio > 1.9:
            return 1
        elif obImbalance < -10000 and obImbalanceRatio < 0.9: # excess sellers are ususually more aggressive than aggressive buyers, so 20% excess sellers is the same as 60% excess buyers
            return -1
        else:
            return 0
    def getSTOrderBookImbalance(self,row):
        if 'futOrderBookBuyQtLevel1' in row:
            obImbalanceRatio = row.futOrderBookBuyQtLevel1/row.futOrderBookSellQtLevel1 if row.futOrderBookSellQtLevel1 > 0 else 10
            obImbalance = row.futOrderBookBuyQtLevel1 - row.futOrderBookSellQtLevel1
        else:
            obImbalance = 0
            obImbalanceRatio = 1
        if obImbalance > 100 and obImbalanceRatio > 1.15:
            return 1
        elif obImbalance < -100 and obImbalanceRatio < 0.85:
            return -1
        else:
            return 0

    def marketIsBullish(self,row):
        return self.getOrderBookImbalance(row) > 0
    def marketIsBearish(self,row):
        return self.getOrderBookImbalance(row) < 0
    def marketIsNeutral(self,row):
        return self.getOrderBookImbalance(row) == 0
    def timeToLookForAGoodExit(self,row):
        ten_minutes = timedelta(minutes=9)

        # combine today's date with the time
        exitTime = datetime.datetime.combine(datetime.datetime.today(), cfgEndExitTradesOnlyTimeOfDay) 

        # subtract 10 minutes
        exitTime -= ten_minutes

        if row.name.time() > exitTime.time():
            return True
        else:
            return False
        if row.name.time().hour == 15 and row.name.time().minute > 10:
            return True
        else:
            return False

    def getVolDeltaSignal(self,row, sigType):
        (niftyUp,niftyDn,futUp,futDn) = row[['niftyUpVol','niftyDnVol','niftyFutureUpVol','niftyFutureDnVol']] if 'niftyUpVol' in row else (0,0,0,0)
        niftyVolDelta = niftyUp - niftyDn
        futVolDelta = futUp - futDn
        
        if row.name.time().hour > 14:
            return 0 # Vol Delta has no impact after 2 PM
        
        if sigType == 'longExit' or sigType == 'shortEntry':
            if niftyVolDelta < -cfgNiftyVolDeltaThreshold or futVolDelta < -cfgFutVolDeltaThreshold:
                return 1
        if sigType == 'shortExit' or sigType == 'longEntry':
            if niftyVolDelta > cfgNiftyVolDeltaThreshold or futVolDelta > cfgFutVolDeltaThreshold:
                return 1
        return 0

# class svb(SignalGenerator):
#     #Helpers
#     def svpTrendsDown(self,row):
#         return row['slpPoc'] <= -cfgSVPSlopeThreshold
#     def svpTrendsUp(self,row):
#         return row['slpPoc'] >= cfgSVPSlopeThreshold
#     def getSVP(self,row,project=False):
#         (vah,poc,val,slpVah,slpVal,slpPoc) = row[['vah','poc','val','slpVah','slpVal','slpPoc']]

#         if project:
#             (vah,poc,val) = getProjectedSVP(row)
#         else:
#             vah = vah + 10 if slpVah > 0.1 else vah
#             val = val - 10 if slpVal < -0.1 else val
#         return (vah,poc,val)
#     def getProjectedSVP(self,row):
#         (vah,poc,val,slpVah,slpVal,slpPoc) = row[['vah','poc','val','slpVah','slpVal','slpPoc']]
#         vah = vah + (slpVah*cfgSVPSlopeProjectionCandles)
#         val = val + (slpVal*cfgSVPSlopeProjectionCandles)
#         poc = poc + (slpPoc*cfgSVPSlopeProjectionCandles)
#         return(vah,poc,val)
#     def getSVPquadrant(self,row):
#         close = row['Adj Close']
#         (vah,poc,val) = self.getSVP(row)
#         if close > vah:
#             q = 'High'
#         elif close > poc:
#             q = 'Upper'
#         elif close > val:
#             q = 'Lower'
#         else:
#             q = 'Low'
#         return q
            
#     def longResistance(self,row):
#         close = row['Adj Close']
#         if row.i < 100: 
#             return close+100 # suppor/resistance bands not formed before 100 candles
#         elif row.slpPoc <= -cfgSVPSlopeThreshold:
#             return row.poc
#         else:
#             return max(close,row.vah) if row.slpVah <= cfgSVPSlopeThreshold else close+8
        
#         (vah,poc,val) = getSVP(row)
#         status = getSVPquadrant(row)
        
#         if status == 'High':
#             r = close+1
#         elif status == 'Upper':
#             r = vah
#         elif status == 'Lower':
#             r = vah
#         else:
#             r = val
#         return r
#     def longSupport(self,row):
#         close = row['Adj Close']
#         if row.i < 100: 
#             return close-100 # suppor/resistance bands not formed before 100 candles
#         elif row.slpPoc >= cfgSVPSlopeThreshold:
#             return row.poc
#         else:
#             return min(close,row.val) if row.slpVal >= -cfgSVPSlopeThreshold else row.val-8
#         (vah,poc,val) = getProjectedSVP(row)
#         status = getSVPquadrant(row)
        
#         if status == 'High':
#             r = vah
#         elif status == 'Upper':
#             r = poc
#         elif status == 'Lower':
#             r = val
#         else:
#             r = close-1
#         return r
#     def shortSupport(self,row):
#         return self.longResistance(row)
            
#     def shortResistance(self,row):
#         return self.longSupport(row)

#     #MAIN
#     def OkToEnterLong(self,row):
#         return self.svpTrendsUp(row)
#     def OkToEnterShort(self,row):
#         return self.svpTrendsDown(row)
#     def checkLongEntry(self,s,row,df,isLastRow,limit1,limit2,sl1,sl2,logString):
#         (close,val15,slpVal15) = (row['Adj Close'],row.val15m,row.slp15Val)
#         i = df.index.get_loc(row.name)    
#         prevClose = df.iloc[i - 1,df.columns.get_loc('Adj Close')]
#         prevVal15 = df.iloc[i - 1,df.columns.get_loc('val15m')]

#         if self.crossOver(close,row.val15m,prevClose,prevVal15):
#             s = 1
#             logString = "SVP-LONG-ENTRY"
#             logging.info(f"TARGET Hit: We Crossed Over Val15") if isLastRow else None
#         elif slpVal15 >= -cfgSVPSlopeThreshold:
#             logString = "SVP-WAITING-FOR-LONG-TREND"
#             limit1 = val15
#             sl1 = -(val15*0.95)
#         else:
#             logString = "SVP-SKIP-LONG-VAL-SLOPING-DOWN"    
#         return (s, limit1, limit2, sl1, sl2,logString)

#     def checkShortEntry(self,s,row,df,isLastRow,limit1,limit2,sl1,sl2,logString):
#         (close,vah15,slpVah15) = (row['Adj Close'],row.vah15m,row.slp15Vah)
#         i = df.index.get_loc(row.name)    
#         prevClose = df.iloc[i - 1,df.columns.get_loc('Adj Close')]
#         prevVah15 = df.iloc[i - 1,df.columns.get_loc('vah15m')]

#         if self.crossUnder(close,row.vah15m,prevClose,prevVah15) and self.getSVPquadrant(row) != 'Low':
#             s = -1
#             logString = "SVP-SHORT-ENTRY"
#             logging.info(f"TARGET Hit: We CrossUnder Vah15") if isLastRow else None
#         elif slpVah15 <= cfgSVPSlopeThreshold and self.getSVPquadrant(row) != 'Low':
#             logString = "SVP-WAITING-FOR-SHORT-TREND"
#             limit1 = -vah15
#             sl1 = -(vah15*1.05)
#         else:
#             logString = "SVP-SKIP-SHORT-VAH-SLOPING-UP" if self.getSVPquadrant(row) != 'Low' else "SVP-SKIP-SHORT-IN-LOW-QUADRANT"
    
#         return (s, limit1, limit2, sl1, sl2,logString)

#     def checkLongExit(self,s,row,df,isLastRow, entryPrice,limit1,limit2,sl1,sl2,logString):
#         (close,vah15,slpVah15) = (row['Adj Close'],row.vah15m,row.slp15Vah)

#         i = df.index.get_loc(row.name)    
#         prevClose = df.iloc[i - 1,df.columns.get_loc('Adj Close')]
#         prevVah15 = df.iloc[i - 1,df.columns.get_loc('vah15m')]
        
#         if self.crossUnder(close,vah15,prevClose,prevVah15):
#             s = 0
#             logString = "SVP-LONG-EXIT"
#             logging.info(f"TARGET Hit: We Crossed Under vah15") if isLastRow else None
#         elif slpVah15 <= cfgSVPSlopeThreshold:
#             sl1 = -entryPrice*.95
#             limit1 = -vah15
#         else:
#             logString = "SVP-SKIP-LONG-VAH-SLOPING-UP"
        
#         return (s, limit1, limit2, sl1, sl2,logString)

#     def checkShortExit(self,s,row,df,isLastRow, entryPrice,limit1,limit2,sl1,sl2,logString):
#         (close,val15,slpVal15) = (row['Adj Close'],row.val15m,row.slp15Val)
#         i = df.index.get_loc(row.name)    
#         prevClose = df.iloc[i - 1,df.columns.get_loc('Adj Close')]
#         prevVal15 = df.iloc[i - 1,df.columns.get_loc('val15m')]
        
#         if self.crossOver(close,val15,prevClose,prevVal15):
#             s = 0 
#             logString = "SVP-SHORT-EXIT"
#             logging.info(f"TARGET HIT: We crosssed over Val 15.")     if isLastRow else None
#         elif slpVal15 >= -cfgSVPSlopeThreshold:
#             sl1 = entryPrice*1.05
#             limit1 = val15 
#         else:
#             logString = "SVP-SKIP-SHORT-EXIT-VAL-SLOPING-DOWN"
            
#         return (s, limit1, limit2, sl1, sl2,logString)

