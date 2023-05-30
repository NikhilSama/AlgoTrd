from pygments import highlight
from .SignalGenerator import SignalGenerator

import numpy as np
import pandas as pd
import logging
import cfg
globals().update(vars(cfg))

class Renko(SignalGenerator):
    logArray = ['RenkoData','ohlv','svp']
    limitExitOrders = False
    limitEntryOrders = False
    slEntryOrders = False
    slExitOrders = False
    exitStaticBricks = False
    useSVPForEntryExitPrices = False
    def __init__(self, limitExitOrders=False, limitEntryOrders=False, slEntryOrders=False, slExitOrders=False,exitStaticBricks=False,useSVPForEntryExitPrices=False,**kwargs):
        self.limitExitOrders = limitExitOrders
        self.limitEntryOrders = limitEntryOrders
        self.slEntryOrders = slEntryOrders
        self.slExitOrders = slExitOrders
        self.exitStaticBricks = exitStaticBricks
        self.useSVPForEntryExitPrices = useSVPForEntryExitPrices
        super().__init__(**kwargs)
        
    #Helpers
    def svpTrendsDown(self,row):
        return row['slpPoc'] <= -cfgSVPSlopeThreshold
    def svpTrendsUp(self,row):
        return row['slpPoc'] >= cfgSVPSlopeThreshold
    def getSVP(self,row,project=False):
        (vah,poc,val,slpVah,slpVal,slpPoc) = row[['vah','poc','val','slpVah','slpVal','slpPoc']]

        if project:
            (vah,poc,val) = getProjectedSVP(row)
        else:
            vah = vah + 10 if slpVah > 0.1 else vah
            val = val - 10 if slpVal < -0.1 else val
        return (vah,poc,val)
    def getProjectedSVP(self,row):
        (vah,poc,val,slpVah,slpVal,slpPoc) = row[['vah','poc','val','slpVah','slpVal','slpPoc']]
        vah = vah + (slpVah*cfgSVPSlopeProjectionCandles)
        val = val + (slpVal*cfgSVPSlopeProjectionCandles)
        poc = poc + (slpPoc*cfgSVPSlopeProjectionCandles)
        return(vah,poc,val)
    def getSVPquadrant(self,row):
        close = row['Adj Close']
        (vah,poc,val) = self.getSVP(row)
        if close > vah:
            q = 'High'
        elif close > poc:
            q = 'Upper'
        elif close > val:
            q = 'Lower'
        else:
            q = 'Low'
        return q
            
    def longResistance(self,row):
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
    def longSupport(self,row):
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
    def shortSupport(self,row):
        return self.longResistance(row)
            
    def shortResistance(self,row):
        return self.longSupport(row)
    def getNumBricksForLongTrend(self,row, type='longEntrySL'):
        if type == 'shortExitSL':
            return abs(row['renko_brick_num']) + 1 
        else:
            return cfgRenkoNumBricksForTrend #if getSVPquadrant(row) != 'Low' or row['slpPoc'] <= -cfgSVPSlopeThreshold else cfgRenkoNumBricksForTrend-1
    def getNumBricksForShortTrend(self,row, type='shortEntrySL'):
        if type == 'longExitSL':
            return abs(row['renko_brick_num']) + 1 
        else:
            return cfgRenkoNumBricksForTrend #if getSVPquadrant(row) != 'High' or row['slpPoc'] >= cfgSVPSlopeThreshold else cfgRenkoNumBricksForTrend-1
    def getImmidiateResistance(self,row):
        #Use SVP
        return row.vah15m if row.slp15Vah < cfgSVPSlopeThreshold else row['Adj Close'] + 10
    def getImmidiateSupport(self,row):
        #Use SVP
        return row.val15m if row.slp15Val > -cfgSVPSlopeThreshold else row['Adj Close'] - 10
    def getSLPrice (self,row,type):
        (brickNum,brickSize,brickHigh,brickLow,close) = \
            (row['renko_brick_num'],row['renko_brick_high'] - row['renko_brick_low'],row['renko_brick_high'],row['renko_brick_low'],row['Adj Close'])
        lowSL = brickLow - ((self.getNumBricksForShortTrend(row,type)-abs(brickNum)) * brickSize)
        highSL = brickHigh + ((self.getNumBricksForLongTrend(row, type)-abs(brickNum)) * brickSize)
        if type == 'longEntrySL':
            return highSL
        elif type == 'longExitSL':
            return -lowSL
        elif type == 'shortEntrySL':
            return -lowSL
        elif type == 'shortExitSL':
            return highSL
        else:
            print(f"Unkonwn SL type {type} in getHighSL")
            exit(-1)
    def getLimit1Price(self,row,type):
        (brickNum,brickSize,brickHigh,brickLow,staticCandles,close) = \
            (row['renko_brick_num'],row['renko_brick_high'] - row['renko_brick_low'],row['renko_brick_high'],row['renko_brick_low'],row['renko_static_candles'],row['Adj Close'])
        
        if self.exitStaticBricks and staticCandles >= cfgMinStaticCandlesForMeanRev :
            
            if self.useSVPForEntryExitPrices:
                h = self.getImmidiateResistance(row)
                l = self.getImmidiateSupport(row)
            else:
                h = (max(close,brickHigh))
                l = min(close,brickLow)
        elif 'Entry' in type:
            if abs(brickNum) < (cfgRenkoNumBricksForTrend + 2):
                h = l = float('nan') # immidiate entry if candles are not static
            else: # We limit exited, and now need to re-enter at a good price
                h = max(brickHigh,close) #if not self.useSVPForEntryExitPrices else self.getImmidiateResistance(row)
                l = min(brickLow,close) #if not self.useSVPForEntryExitPrices else self.getImmidiateSupport(row)
        else: # Exit limit when candles are non-static or we are not using exiStaticBricks
            h = (max(brickHigh,close) + 10*brickSize)
            l = min(brickLow,close) - 10*brickSize

            if self.useSVPForEntryExitPrices:
                h = close - 2 if row.slpVah <= 0 and staticCandles > 2 else h
                l = close + 2 if row.slpVal >= 0 and staticCandles > 2 else l
            
        h = round(h,1)
        l = round(l,1)
                
        if type == 'longExitLimit1':
            return -h
        elif type == 'shortExitLimit1':
            return l
        elif type == 'longEntryLimit1':
            return l
        elif type == 'shortEntryLimit1':
            return -h
        else:
            print(f"Unkonwn limit type {type} in getLimit1Price")
            exit(-1)
    printedDates = set()
    def wontTrendToday(self,row,df):
        (dayHigh,dayLow,voh,vol) = row[['dayHigh','dayLow','vah','val']]
        
        if row.name.time() < cfgTimeToCheckDayTrendInfo:
            return False
        df = df[df.index.date == row.name.date()]
        uniqueRenkoBrickNum = df['renko_brick_num'].unique()
        maxRenko = np.nanmax(np.abs(uniqueRenkoBrickNum))

        if maxRenko < 2:
            print(f"{row.name.date()} is not trending") if row.name.date() not in self.printedDates else None
            self.printedDates.add(row.name.date())
            return True
        else:
            return False
    #MAIN
    def OkToEnterLong(self,row):
        return row['renko_uptrend']
    def OkToEnterShort(self,row):
        return not row['renko_uptrend']
    def checkLongEntry(self,s,row,df,isLastRow,limit1,limit2,sl1,sl2,logString):
        (brickNum,brickSize,brickHigh,brickLow,close) = (row['renko_brick_num'],row['renko_brick_high'] - row['renko_brick_low'],row['renko_brick_high'],row['renko_brick_low'],row['Adj Close'])
        
        self.wontTrendToday(row,df)
        if brickNum >= self.getNumBricksForLongTrend(row):
            # if self.meanRevAtStaticCandles and staticCandles > cfgMinStaticCandlesForMeanRev:
            limit1 = self.getLimit1Price(row,'longEntryLimit1') if self.limitEntryOrders else float('nan')
            if np.isnan(limit1):
                s = 1
                logString = "RENKO-LONG-ENTRY"
            # sl1 = (brickHigh + brickSize) if not np.isnan(limit1) else sl1
        elif self.slEntryOrders and brickNum >= 1:
            sl1 = self.getSLPrice(row,'longEntrySL') 
        return (s, limit1, limit2, sl1, sl2,logString)

    def checkShortEntry(self,s,row,df,isLastRow,limit1,limit2,sl1,sl2,logString):
        (brickNum,brickSize,brickHigh,brickLow,close) = (row['renko_brick_num'],row['renko_brick_high'] - row['renko_brick_low'],row['renko_brick_high'],row['renko_brick_low'],row['Adj Close'])
        self.wontTrendToday(row,df)

        if brickNum <= -self.getNumBricksForShortTrend(row):
            limit1 = self.getLimit1Price(row,'shortEntryLimit1') if self.limitEntryOrders else float('nan')
            # sl1 = -(brickLow - brickSize) if not np.isnan(limit1) else sl1
            if np.isnan(limit1):
                s = -1 
                logString = "RENKO-SHORT-ENTRY"

        elif self.slEntryOrders and brickNum <= -1:
            sl1 = self.getSLPrice(row,'shortEntrySL') 
        return (s, limit1, limit2, sl1, sl2,logString)

    def checkLongExit(self,s,row,df,isLastRow, entryPrice,limit1,limit2,sl1,sl2,logString):
        (brickNum,uptrend,brickSize,brickHigh,brickLow,close) = (row['renko_brick_num'],row['renko_uptrend'],row['renko_brick_high'] - row['renko_brick_low'],row['renko_brick_high'],row['renko_brick_low'],row['Adj Close'])

        if not uptrend:
            if brickNum <= -cfgRenkoNumBricksForTrend:
                s = -1
                logString = "RENKO-LONG-EXIT-AND-SHORT-ENTRY"
            else:
                s = 0
                logString = "RENKO-LONG-EXIT"
        else:
            if self.slExitOrders:
                sl1 = self.getSLPrice(row,'longExitSL') 
            if self.limitExitOrders:
                limit1 = self.getLimit1Price(row,'longExitLimit1')
        return (s, limit1, limit2, sl1, sl2,logString)

    def checkShortExit(self,s,row,df,isLastRow, entryPrice,limit1,limit2,sl1,sl2,logString):
        (brickNum,uptrend,brickSize,brickHigh,brickLow,close) = (row['renko_brick_num'],row['renko_uptrend'],row['renko_brick_high'] - row['renko_brick_low'],row['renko_brick_high'],row['renko_brick_low'],row['Adj Close'])

        if uptrend:
            if brickNum >= cfgRenkoNumBricksForTrend:
                s = 1
                logString = "RENKO-SHORT-EXIT-AND-LONG-ENTRY"
            else:
                s = 0
                logString = "RENKO-SHORT-EXIT"
        else:
            if self.slExitOrders:
                sl1 = self.getSLPrice(row,'shortExitSL') 
            if self.limitExitOrders:
                limit1 = self.getLimit1Price(row,'shortExitLimit1')
        return (s, limit1, limit2, sl1, sl2,logString)
