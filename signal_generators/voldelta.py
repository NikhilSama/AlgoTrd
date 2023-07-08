from pygments import highlight
from .SignalGenerator import SignalGenerator

import numpy as np
import pandas as pd
import logging
import cfg
globals().update(vars(cfg))

class VolDelta(SignalGenerator):
    logArray = ['ohlv','voldelta']
    
    def __init__(self, limitExitOrders=False, limitEntryOrders=False, slEntryOrders=False, slExitOrders=False,exitStaticBricks=False,useSVPForEntryExitPrices=False,**kwargs):
        super().__init__(**kwargs)
    
    def timeWindow(self,row):
        return True
        
    def priceRange(self,row):
        return True
    #MAIN

    def OkToEnterLong(self,row):
        
        return row.volDelta > row.volDeltaThreshold \
            and row.volDelta > 0.8 * row.maxVolDelta \
                
    
    def OkToEnterShort(self,row):
        return row.volDelta < -row.volDeltaThreshold \
            and row.volDelta < 0.8 * row.minVolDelta \
    
    def checkLongEntry(self,s,row,df,isLastRow,limit1,limit2,sl1,sl2,logString):
        s = 1
        return (s, limit1, limit2, sl1, sl2,logString)

    def checkShortEntry(self,s,row,df,isLastRow,limit1,limit2,sl1,sl2,logString):
        s = -1
        return (s, limit1, limit2, sl1, sl2,logString)

    def checkLongExit(self,s,row,df,isLastRow, entryPrice,limit1,limit2,sl1,sl2,logString, tradeEntry, tradeMax):
        (close, volDelta, volDeltaThreshold) = (row['Adj Close'],row['volDelta'], row['volDeltaThreshold'])
        gainPct = 100 * (close - tradeEntry)/tradeEntry

        exitThreshold = -volDeltaThreshold/2 #if gainPct > 5 else -volDeltaThreshold

        if volDelta < exitThreshold:
            s = -1 if self.OkToEnterShort(row) else 0
        else:
            limit1 = -(tradeEntry*1.02)
            sl1 = -(tradeEntry*0.98)

        return (s, limit1, limit2, sl1, sl2,logString)

    def checkShortExit(self,s,row,df,isLastRow, entryPrice,limit1,limit2,sl1,sl2,logString, tradeEntry,tradeMin):
        (close, volDelta, volDeltaThreshold) = (row['Adj Close'],row['volDelta'], row['volDeltaThreshold'])
        gainPct = 100 * -(close - tradeEntry)/tradeEntry
        
        exitThreshold = volDeltaThreshold/2 #if gainPct > 5 else volDeltaThreshold

        if volDelta > exitThreshold:
            s = 1 if self.OkToEnterLong(row) else 0
        else:
            limit1 = tradeEntry*.98
            sl1 = tradeEntry*1.02   

        return (s, limit1, limit2, sl1, sl2,logString)


