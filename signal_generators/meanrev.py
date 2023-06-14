from pygments import highlight
from .SignalGenerator import SignalGenerator

import numpy as np
import pandas as pd
import logging
import cfg
globals().update(vars(cfg))

class MeanRev(SignalGenerator):
    logArray = ['RenkoData','ohlv','svp', 'candlestick']
    def __init__(self, limitExitOrders=False, limitEntryOrders=False, slEntryOrders=False, slExitOrders=False,exitStaticBricks=False,useSVPForEntryExitPrices=False,**kwargs):
        super().__init__(**kwargs)

    def meanRevTimeWindow(self,row):
        return cfgMeanRevStartTime <= row.name.time() <= cfgMeanRevEndTime
            
    #MAIN
    def OkToEnterLong(self,row):
        close = row['Adj Close']
        poc = row.pocShrtTrm
        return close < poc and self.meanRevTimeWindow(row)
        # return row.slpPoc >= 0 and row.slpSTPoc >= 0 and self.meanRevTimeWindow(row) and row['Adj Close'] < row.vah
    
    def OkToEnterShort(self,row):
        close = row['Adj Close']
        poc = row.pocShrtTrm
        return close > poc and self.meanRevTimeWindow(row)
        #return row.slpPoc <= 0 and row.slpSTPoc <= 0 and self.meanRevTimeWindow(row) and row['Adj Close'] > row.val
    
    def checkLongEntry(self,s,row,df,isLastRow,limit1,limit2,sl1,sl2,logString):
        (stSVPLow,stSlpSVPLow,close, ShrtTrmLow) = (row['valShrtTrm'],row['slpSTVal'],row['Adj Close'],row['ShrtTrmLow'])
        
        limit1 = min(close,stSVPLow+stSlpSVPLow)
        sl1 = -(limit1 - 4)
            
        return (s, limit1, limit2, sl1, sl2,logString)

    def checkShortEntry(self,s,row,df,isLastRow,limit1,limit2,sl1,sl2,logString):
        (stSVPHigh,stSlpSVPHigh,close, ShrtTrmHigh) = (row['vahShrtTrm'],row['slpSTVah'],row['Adj Close'],row['ShrtTrmHigh'])
        limit1 = -max(close,stSVPHigh+stSlpSVPHigh)
        sl1 = limit1 + 4
        return (s, limit1, limit2, sl1, sl2,logString)

    def checkLongExit(self,s,row,df,isLastRow, entryPrice,limit1,limit2,sl1,sl2,logString):
        (stSVPHigh,close, ShrtTrmHigh, ShrtTrmLow) = (row['vahShrtTrm'],row['Adj Close'],row['ShrtTrmHigh'], row['ShrtTrmLow'])

        limit1 = -(stSVPHigh)
        sl1 = -(ShrtTrmLow - 4)

        return (s, limit1, limit2, sl1, sl2,logString)

    def checkShortExit(self,s,row,df,isLastRow, entryPrice,limit1,limit2,sl1,sl2,logString):
        (stSVPLow,close, ShrtTrmHigh, ShrtTrmLow) = (row['valShrtTrm'],row['Adj Close'],row['ShrtTrmHigh'], row['ShrtTrmLow'])

        limit1 = stSVPLow
        sl1 = (ShrtTrmHigh + 4)

        return (s, limit1, limit2, sl1, sl2,logString)


