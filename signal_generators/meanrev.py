from pygments import highlight
from .SignalGenerator import SignalGenerator

import numpy as np
import pandas as pd
import logging
import cfg
globals().update(vars(cfg))

class MeanRev(SignalGenerator):
    logArray = ['ohlv','fastma','svp','svpST']
    
    def __init__(self, limitExitOrders=False, limitEntryOrders=False, slEntryOrders=False, slExitOrders=False,exitStaticBricks=False,useSVPForEntryExitPrices=False,**kwargs):
        super().__init__(**kwargs)

    def meanRevTimeWindow(self,row):
        return cfgMeanRevStartTime <= row.name.time() <= cfgMeanRevEndTime
            
    #MAIN
    def pocIsFlat(self,row):
        return row.slpSTPoc < .3 and row.slpSTPoc > -.3 and row.slpSDSTPoc <= 0.3
    def vahNotFlying(self,row):
        return row.slpSTVah < 1 
    def valNotDiving(self,row):
        return row.slpSTVal > -1 

    def OkToEnterLong(self,row):
        return row.High <= row.val  and row.slpPoc > -2 and row.slpVal > -5 and self.meanRevTimeWindow(row)
        return abs(row.slpPoc) < 5 and row['Adj Close'] < row.pocShrtTrm and self.meanRevTimeWindow(row) \
            and abs(row.renko_brick_num) < 2 and row.slpSTPoc <= 1
        close = row['Adj Close']
        poc = row.pocShrtTrm
        return close < poc and self.meanRevTimeWindow(row) and self.pocIsFlat(row) and self.valNotDiving(row)
        # return row.slpPoc >= 0 and row.slpSTPoc >= 0 and self.meanRevTimeWindow(row) and row['Adj Close'] < row.vah
    
    def OkToEnterShort(self,row):
        return row.Low >= row.vah  and row.slpPoc < 2 and row.slpVah < 5 and self.meanRevTimeWindow(row)

        return abs(row.slpPoc) < 5 and row['Adj Close'] > row.pocShrtTrm and self.meanRevTimeWindow(row) \
            and abs(row.renko_brick_num) < 2 and abs(row.slpSTPoc) <= 1

        close = row['Adj Close']
        poc = row.pocShrtTrm
        return close > poc and self.meanRevTimeWindow(row) and self.pocIsFlat(row) and self.vahNotFlying(row)
        #return row.slpPoc <= 0 and row.slpSTPoc <= 0 and self.meanRevTimeWindow(row) and row['Adj Close'] > row.val
    
    def checkLongEntry(self,s,row,df,prevPosition,tradeHigh,tradeLow,isLastRow,limit1,limit2,sl1,sl2,logString):
        (stSVPLow,stSlpSVPLow,close, ShrtTrmLow) = (row['valShrtTrm'],row['slpSTVal'],row['Adj Close'],row['ShrtTrmLow'])
        (brickNum,brickSize,brickHigh,brickLow,close,high) = (row['renko_brick_num'],row['renko_brick_high'] - row['renko_brick_low'],row['renko_brick_high'],row['renko_brick_low'],row['Adj Close'],row.High)

        if close > row.ShrtTrmHigh:
            s = 1
            logString = "LONG-BURST-ENTRY"
        else:
            # limit1 = row.pocShrtTrm
            sl1 = row.val 
        return (s, limit1, limit2, sl1, sl2,logString)

    def checkShortEntry(self,s,row,df,prevPosition,tradeHigh,tradeLow,isLastRow,limit1,limit2,sl1,sl2,logString):
        (stSVPHigh,stSlpSVPHigh,close, ShrtTrmHigh) = (row['vahShrtTrm'],row['slpSTVah'],row['Adj Close'],row['ShrtTrmHigh'])
        (brickNum,brickSize,brickHigh,brickLow,close,high) = (row['renko_brick_num'],row['renko_brick_high'] - row['renko_brick_low'],row['renko_brick_high'],row['renko_brick_low'],row['Adj Close'],row.High)
        
        if close < row.ShrtTrmLow:
            s = -1
            logString = "SHORT-BURST-ENTRY"
        else:
            # limit1 = -(row.pocShrtTrm)
            sl1 = -(row.vah)
        
        return (s, limit1, limit2, sl1, sl2,logString)

    def checkLongExit(self,s,row,df,isLastRow, entryPrice,limit1,limit2,sl1,sl2,logString,
                      tradeEntry,tradeHigh,tradeLow):
        (stSVPHigh,close, ShrtTrmHigh, ShrtTrmLow,poc) = (row['vahShrtTrm'],row['Adj Close'],row['ShrtTrmHigh'], row['ShrtTrmLow'],row['pocShrtTrm'])
        (brickNum,brickSize,brickHigh,brickLow,close,high) = (row['renko_brick_num'],row['renko_brick_high'] - row['renko_brick_low'],row['renko_brick_high'],row['renko_brick_low'],row['Adj Close'],row.High)

        limit1 = -(row.poc)
        sl1 = -(ShrtTrmLow-5)

        return (s, limit1, limit2, sl1, sl2,logString)

    def checkShortExit(self,s,row,df,isLastRow, entryPrice,limit1,limit2,sl1,sl2,logString,
                       tradeEntry,tradeHigh,tradeLow):
        (stSVPLow,close, ShrtTrmHigh, ShrtTrmLow,poc) = (row['valShrtTrm'],row['Adj Close'],row['ShrtTrmHigh'], row['ShrtTrmLow'],row['pocShrtTrm'])
        (brickNum,brickSize,brickHigh,brickLow,close,high) = (row['renko_brick_num'],row['renko_brick_high'] - row['renko_brick_low'],row['renko_brick_high'],row['renko_brick_low'],row['Adj Close'],row.High)

        limit1 = (row.poc)
        sl1 = (ShrtTrmHigh+5)

        return (s, limit1, limit2, sl1, sl2,logString)


