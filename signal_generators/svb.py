from .SignalGenerator import SignalGenerator

import numpy as np
import pandas as pd
import logging
import cfg
globals().update(vars(cfg))

class svb(SignalGenerator):
    logArray = ['voldelta','svp','ohlv']
    extraLogString = ''
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
    def timeWindow(self,row):
        return True if row.name.hour >= 11 else False

    #MAIN
    def OkToEnterLong(self,row):
        close = row['Adj Close']
        if self.marketIsNeutral(row) and close < row.poc and self.timeWindow(row):
            return True
        else:
            return False

    def OkToEnterShort(self,row):
        close = row['Adj Close']
        if self.marketIsNeutral(row) and close > row.poc and self.timeWindow(row):
            return True
        else:
            return False
    def checkLongEntry(self,s,row,df,isLastRow,limit1,limit2,sl1,sl2,logString):
        (close,poc,vah,val) = (row['Adj Close'],row.poc,row.vah,row.val)

        if self.getVolDeltaSignal(row,'longEntry') or \
            (close <= val and self.getSTOrderBookImbalance(row) == 1):
            s = 1
            logString = "SVP-LONG-ENTRY"
        else:
            logString = "SVP-WAITING-FOR-LONG-ENTRY"
            limit1 = val-5
        return (s, limit1, limit2, sl1, sl2,logString)

    def checkShortEntry(self,s,row,df,isLastRow,limit1,limit2,sl1,sl2,logString):
        (close,poc,vah,val) = (row['Adj Close'],row.poc,row.vah,row.val)

        if self.getVolDeltaSignal(row,'shortEntry') or \
            (close >= vah and self.getSTOrderBookImbalance(row) == -1):
            s = -1
            logString = "SVP-SHORT-ENTRY"
        else:
            logString = "SVP-WAITING-FOR-SHORT-ENTRY"
            limit1 = -(vah+5)
            
        return (s, limit1, limit2, sl1, sl2,logString)

    def checkLongExit(self,s,row,df,isLastRow, entryPrice,limit1,limit2,sl1,sl2,logString,
                      tradeEntry,tradeMax):
        (close,poc,vah,val) = (row['Adj Close'],row.poc,row.vah,row.val)

        if self.getVolDeltaSignal(row,'longExit') or \
            (close >= poc and self.getSTOrderBookImbalance(row) == -1):
            s = 0
            logString = "SVP-LONG-EXIT"
        else:
            limit1 = -vah
            sl1 = -(max(tradeEntry,tradeMax) - 15)
        return (s, limit1, limit2, sl1, sl2,logString)

    def checkShortExit(self,s,row,df,isLastRow, entryPrice,limit1,limit2,sl1,sl2,logString,
                       tradeEntry,tradeMin):
        (close,poc,vah,val) = (row['Adj Close'],row.poc,row.vah,row.val)
        
        if self.getVolDeltaSignal(row,'shortExit') or \
            (close <= poc and self.getSTOrderBookImbalance(row) == 1):
            s = 0 
            logString = "SVP-SHORT-EXIT"
        else:
            limit1 = val
            sl1 = min(tradeEntry,tradeMin) + 15
        return (s, limit1, limit2, sl1, sl2,logString)
