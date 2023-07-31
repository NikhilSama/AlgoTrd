from pygments import highlight
from .SignalGenerator import SignalGenerator

import numpy as np
import pandas as pd
import logging
import cfg
globals().update(vars(cfg))

class BurstFinder (SignalGenerator):
    logArray = ['ohlv','svp','svpST']
    
    def __init__(self, limitExitOrders=False, limitEntryOrders=False, slEntryOrders=False, slExitOrders=False,exitStaticBricks=False,useSVPForEntryExitPrices=False,useVolDelta=False,**kwargs):
        super().__init__(**kwargs)

    def timeWindow(self,row):
        return True
        
    #MAIN
    def OkToEnterLong(self,row):
        return row['Adj Close'] < row.pocShrtTrm
        return row.slpSTPoc >=0 
        close = row['Adj Close']
        return close <= row.pocShrtTrm and row.slpSTLow >= 0
        return True

    def OkToEnterShort(self,row):
        return row['Adj Close'] > row.pocShrtTrm
        return row.slpSTPoc <=0 

        close = row['Adj Close']
        return close > row.pocShrtTrm and row.slpSTHigh <= 0
        return True
    
    def checkLongEntry(self,s,row,df,prevPosition,tradeHigh,tradeLow,isLastRow,limit1,limit2,sl1,sl2,logString):
        (poc,vah,val,h,l) = (row['pocShrtTrm'],row['vahShrtTrm'],row['valShrtTrm'],row['ShrtTrmHigh'],row['ShrtTrmLow'])
        # sl1 = h+8
        limit1 = l-4
        return (s, limit1, limit2, sl1, sl2,logString)

    def checkShortEntry(self,s,row,df,prevPosition,tradeHigh,tradeLow,isLastRow,limit1,limit2,sl1,sl2,logString):
        (poc,vah,val,h,l) = (row['pocShrtTrm'],row['vahShrtTrm'],row['valShrtTrm'],row['ShrtTrmHigh'],row['ShrtTrmLow'])
        # sl1 = -(l-8)
        limit1 = -(h+4)
        return (s, limit1, limit2, sl1, sl2,logString)

    def checkLongExit(self,s,row,df,isLastRow, entryPrice,limit1,limit2,sl1,sl2,logString,
                      tradeEntry,tradeHigh,tradeLow):
        (poc,vah,val,h,l,close) = (row['pocShrtTrm'],row['vahShrtTrm'],row['valShrtTrm'],row['ShrtTrmHigh'],\
            row['ShrtTrmLow'],row['Adj Close'])      
        (brickNum,brickSize,brickHigh,brickLow,close,high) = (row['renko_brick_num'],row['renko_brick_high'] - row['renko_brick_low'],row['renko_brick_high'],row['renko_brick_low'],row['Adj Close'],row.High)
 
        sl1 = -(l-5) # (max(tradeEntry-7,tradeHigh-20))
        if close <= abs(sl1):
            s = 0
            sl1 = float('nan')
            logString = "EXIT-SL"

        limit1 = -(tradeEntry+5)
        limit1 = -(poc)
        limit1 = -(h)
        
        sl1 = -(tradeEntry-10)
        limit1 = -(tradeEntry+10)
        return (s, limit1, limit2, sl1, sl2,logString)

    def checkShortExit(self,s,row,df,isLastRow, entryPrice,limit1,limit2,sl1,sl2,logString,
                       tradeEntry,tradeHigh,tradeLow):
        (poc,vah,val,h,l,close) = (row['pocShrtTrm'],row['vahShrtTrm'],row['valShrtTrm'],row['ShrtTrmHigh'],\
            row['ShrtTrmLow'],row['Adj Close'])       
        (brickNum,brickSize,brickHigh,brickLow,close,high) = (row['renko_brick_num'],row['renko_brick_high'] - row['renko_brick_low'],row['renko_brick_high'],row['renko_brick_low'],row['Adj Close'],row.High)

        sl1 = h+5 # min(tradeEntry+7,tradeLow+20)
        if close >= sl1:
            s = 0
            sl1 = float('nan')
            logString = "EXIT-SL"

        limit1 = tradeEntry-5
        limit1 = (poc)
        limit1 = (l)
        sl1 = tradeEntry+10
        limit1 = tradeEntry-10

        return (s, limit1, limit2, sl1, sl2,logString)
