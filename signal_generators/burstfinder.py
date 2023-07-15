from pygments import highlight
from .SignalGenerator import SignalGenerator

import numpy as np
import pandas as pd
import logging
import cfg
globals().update(vars(cfg))

class BurstFinder (SignalGenerator):
    logArray = ['ohlv']
    
    def __init__(self, limitExitOrders=False, limitEntryOrders=False, slEntryOrders=False, slExitOrders=False,exitStaticBricks=False,useSVPForEntryExitPrices=False,useVolDelta=False,**kwargs):
        super().__init__(**kwargs)

    def timeWindow(self,row):
        return True
        
    #MAIN
    def OkToEnterLong(self,row):
        return True

    def OkToEnterShort(self,row):
        return True
    def checkLongEntry(self,s,row,df,tradeHigh,tradeLow,isLastRow,limit1,limit2,sl1,sl2,logString):
       
        close = row['Adj Close']
        sl1 = row.Low+15
        return (s, limit1, limit2, sl1, sl2,logString)

    def checkShortEntry(self,s,row,df,tradeHigh,tradeLow,isLastRow,limit1,limit2,sl1,sl2,logString):
        close = row['Adj Close']
        sl1 = -(row.Low-15)
        return (s, limit1, limit2, sl1, sl2,logString)

    def checkLongExit(self,s,row,df,isLastRow, entryPrice,limit1,limit2,sl1,sl2,logString,
                      tradeEntry,tradeHigh,tradeLow):
        sl1 = -(tradeEntry-15)
        if row.Low<sl1:
            s = 0
            sl1 = float('nan')
            logging.debug(f"tradeEntry is {tradeEntry} and sl1 is {sl1} and row.High is {row.High}")
            logString = "EXIT-SL"

        limit1 = -(tradeEntry+15)
        return (s, limit1, limit2, sl1, sl2,logString)

    def checkShortExit(self,s,row,df,isLastRow, entryPrice,limit1,limit2,sl1,sl2,logString,
                       tradeEntry,tradeHigh,tradeLow):
        
        sl1 = tradeEntry+15
        if row.High>sl1:
            s = 0
            sl1 = float('nan')
            logging.debug(f"tradeEntry is {tradeEntry} and sl1 is {sl1} and row.High is {row.High}")
            logString = "EXIT-SL"

        limit1 = tradeEntry-30
        return (s, limit1, limit2, sl1, sl2,logString)
