from pygments import highlight
from .SignalGenerator import SignalGenerator

import numpy as np
import pandas as pd
import logging
import cfg
globals().update(vars(cfg))

class MeanRev(SignalGenerator):
    logArray = ['ohlv','svpST']
    
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
        close = row['Adj Close']
        poc = row.pocShrtTrm
        return close < poc and self.meanRevTimeWindow(row) and self.pocIsFlat(row) and self.valNotDiving(row)
        # return row.slpPoc >= 0 and row.slpSTPoc >= 0 and self.meanRevTimeWindow(row) and row['Adj Close'] < row.vah
    
    def OkToEnterShort(self,row):
        close = row['Adj Close']
        poc = row.pocShrtTrm
        return close > poc and self.meanRevTimeWindow(row) and self.pocIsFlat(row) and self.vahNotFlying(row)
        #return row.slpPoc <= 0 and row.slpSTPoc <= 0 and self.meanRevTimeWindow(row) and row['Adj Close'] > row.val
    
    def checkLongEntry(self,s,row,df,isLastRow,limit1,limit2,sl1,sl2,logString):
        (stSVPLow,stSlpSVPLow,close, ShrtTrmLow) = (row['valShrtTrm'],row['slpSTVal'],row['Adj Close'],row['ShrtTrmLow'])
        
        limit1 = min(close,ShrtTrmLow-3)
        sl1 = -(limit1 - 10)
            
        return (s, limit1, limit2, sl1, sl2,logString)

    def checkShortEntry(self,s,row,df,isLastRow,limit1,limit2,sl1,sl2,logString):
        (stSVPHigh,stSlpSVPHigh,close, ShrtTrmHigh) = (row['vahShrtTrm'],row['slpSTVah'],row['Adj Close'],row['ShrtTrmHigh'])
        limit1 = -max(close,ShrtTrmHigh+3)
        sl1 = abs(limit1) + 10
        return (s, limit1, limit2, sl1, sl2,logString)

    def checkLongExit(self,s,row,df,isLastRow, entryPrice,limit1,limit2,sl1,sl2,logString):
        (stSVPHigh,close, ShrtTrmHigh, ShrtTrmLow,poc) = (row['vahShrtTrm'],row['Adj Close'],row['ShrtTrmHigh'], row['ShrtTrmLow'],row['pocShrtTrm'])

        limit1 = -(stSVPHigh)
        sl1 = -(max(ShrtTrmLow,entryPrice) - 10)

        return (s, limit1, limit2, sl1, sl2,logString)

    def checkShortExit(self,s,row,df,isLastRow, entryPrice,limit1,limit2,sl1,sl2,logString):
        (stSVPLow,close, ShrtTrmHigh, ShrtTrmLow,poc) = (row['valShrtTrm'],row['Adj Close'],row['ShrtTrmHigh'], row['ShrtTrmLow'],row['pocShrtTrm'])

        limit1 = stSVPLow
        sl1 = (min(ShrtTrmLow,entryPrice) + 10)

        return (s, limit1, limit2, sl1, sl2,logString)


