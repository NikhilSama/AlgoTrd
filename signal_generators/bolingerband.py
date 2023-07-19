from pygments import highlight
from .SignalGenerator import SignalGenerator

import numpy as np
import pandas as pd
import logging
import cfg
globals().update(vars(cfg))

class BollingerBand (SignalGenerator):
    logArray = ['bb','rsi','ohlv']
    
    def __init__(self, limitExitOrders=False, limitEntryOrders=False, slEntryOrders=False, slExitOrders=False,exitStaticBricks=False,useSVPForEntryExitPrices=False,useVolDelta=False,**kwargs):
        super().__init__(**kwargs)

    def timeWindow(self,row):
        return True
        
    #MAIN
    def OkToEnterLong(self,row):
        return row.ma20_pct_change>0.02 and row.renko_uptrend and row.RSI < 70

    def OkToEnterShort(self,row):
        return row.ma20_pct_change<-0.02 and not row.renko_uptrend and row.RSI > 30
    
    def checkLongEntry(self,s,row,df,prevPosition,tradeHigh,tradeLow,isLastRow,limit1,limit2,sl1,sl2,logString):
        limit1 = row.lower_band #row.ma20 #(row.ma20+row.lower_band)/2#row.lower_band #
        return (s, limit1, limit2, sl1, sl2,logString)

    def checkShortEntry(self,s,row,df,prevPosition,tradeHigh,tradeLow,isLastRow,limit1,limit2,sl1,sl2,logString):
        limit1 = -row.upper_band #row.ma20 #-(row.ma20+row.upper_band)/2#-row.upper_band #
        return (s, limit1, limit2, sl1, sl2,logString)

    def checkLongExit(self,s,row,df,isLastRow, entryPrice,limit1,limit2,sl1,sl2,logString,
                      tradeEntry,tradeHigh,tradeLow):
        limit1 = -(row.upper_band)
        sl1 = -(tradeEntry*.95)#(row.lower_band)
        if row.Low<abs(sl1):
            s = 0
            logString = "LONG-EXIT-PRE-SL"
            sl1 = float('nan')
        return (s, limit1, limit2, sl1, sl2,logString)

    def checkShortExit(self,s,row,df,isLastRow, entryPrice,limit1,limit2,sl1,sl2,logString,
                       tradeEntry,tradeHigh,tradeLow):
        limit1 = row.lower_band
        sl1 = tradeEntry*1.05#row.upper_band
        if row.High > abs(sl1):
            s = 0
            logString = "SHORT-EXIT-PRE-SL"
            sl1 = float('nan')

        return (s, limit1, limit2, sl1, sl2,logString)



# RESULTS
#100 offset .. works better on low value options, possibly even better on offset 50 if possible 



# row.ma20_pct_change>0.02

# Total Return: 214.92%
# Drawdown from Prev Peak: -44.60%
# Sharpe:  10.440956344202105
# Calamar:  -1.3189647664566146
# Num Trades:  76
# Avg Return per day: 0.65%
# Worst Day (2022-05-11): -10.97%
# Best Day (2022-06-22): 49.02%
# No CFG for NIFTYWEEKLYOPTIONCall100. Reverting to default CFG
# End took 53279.12ms


# row.ma20_pct_change>0.015
# Total Return: 402.69%
# Drawdown from Prev Peak: -78.45%
# Sharpe:  7.017966973022232
# Calamar:  -0.6847544044830366
# Num Trades:  218
# Avg Return per day: 1.22%
# Worst Day (2022-06-14): -19.74%
# Best Day (2022-06-22): 67.01%


# row.ma20_pct_change>0.01
# Total Return: 685.64%
# Drawdown from Prev Peak: -70.76%
# Sharpe:  5.235975778458924
# Calamar:  -0.8011089369837766
# Num Trades:  563
# Avg Return per day: 2.05%
# Worst Day (2022-06-14): -24.54%
# Best Day (2022-06-22): 62.07%
# No CFG for NIFTYWEEKLYOPTIONCall100. Reverting to default CFG
# End took 56431.25ms