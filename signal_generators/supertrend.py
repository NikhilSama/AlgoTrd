from pygments import highlight
from .SignalGenerator import SignalGenerator

import numpy as np
import pandas as pd
import logging
import cfg
globals().update(vars(cfg))

class SuperTrend(SignalGenerator):
    logArray = ['ohlv','supertrend']
    
    def __init__(self, limitExitOrders=False, limitEntryOrders=False, slEntryOrders=False, slExitOrders=False,exitStaticBricks=False,useSVPForEntryExitPrices=False,**kwargs):
        super().__init__(**kwargs)
    
    def timeWindow(self,row):
        return  row.name.time().hour > 9
    
    def priceRange(self,row):
        return row['Adj Close'] > 120 
    
    def isBigBurst(self,row,df):
        return True
        (o,h,l,c) = (row['Open'],row['High'],row['Low'],row['Adj Close'])
        
    #MAIN

    def OkToEnterLong(self,row):
        return row.SuperTrend == 1 and self.timeWindow(row) 
        return row.SuperTrendDirection == True and self.timeWindow(row) and self.priceRange(row)
    
    def OkToEnterShort(self,row):
        return row.SuperTrend == -1 and self.timeWindow(row) 
        return  row.SuperTrendDirection == False and self.timeWindow(row)
    
    def checkLongEntry(self,s,row,df,tradeHigh,tradeLow,isLastRow,limit1,limit2,sl1,sl2,logString):
        (close, upper, lower) = (row['Adj Close'],row['SuperTrendUpper'], row['SuperTrendLower'])
        s = 1 if self.isBigBurst(row,df) else s
        return (s, limit1, limit2, sl1, sl2,logString)

    def checkShortEntry(self,s,row,df,tradeHigh,tradeLow,isLastRow,limit1,limit2,sl1,sl2,logString):
        (close, upper, lower) = (row['Adj Close'],row['SuperTrendUpper'], row['SuperTrendLower'])
        s = -1 if self.isBigBurst(row,df) else s
        return (s, limit1, limit2, sl1, sl2,logString)

    def checkLongExit(self,s,row,df,isLastRow, entryPrice,limit1,limit2,sl1,sl2,logString,
                      tradeEntry,tradeHigh,tradeLow):
        (close, upper, lower) = (row['Adj Close'],row['SuperTrendUpper'], row['SuperTrendLower'])

        if not row.SuperTrendDirection:
            s = 0
        else:
            limit1 = -(tradeEntry*1.05)
            sl1 = -(tradeEntry*0.98)

        return (s, limit1, limit2, sl1, sl2,logString)

    def checkShortExit(self,s,row,df,isLastRow, entryPrice,limit1,limit2,sl1,sl2,logString,
                       tradeEntry,tradeHigh,tradeLow):
        (close, upper, lower) = (row['Adj Close'],row['SuperTrendUpper'], row['SuperTrendLower'])

        if row.SuperTrendDirection:
            s = 0
        else:
            limit1 = tradeEntry*.95
            sl1 = tradeEntry*1.02

        return (s, limit1, limit2, sl1, sl2,logString)


