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
        # Backtest 4/1/2022 - 4/1/2023
        #  All False: 
# Total Return: 1332.56%
# Drawdown from Prev Peak: -267.92%
# Sharpe:  2.9797768868890513
# Calamar:  -0.3462998383863521
# Num Trades:  754
# Avg Return per day: 3.99%    

        # SLExitOrders:
#         Total Return: 1513.45%
# Drawdown from Prev Peak: -235.99%
# Sharpe:  3.546399417087138
# Calamar:  -0.44651128975117554
# Num Trades:  997
# Avg Return per day: 4.53%
# Worst Day (2023-02-08): -107.81%
# Best Day (2022-08-30): 145.78%
# No CFG for NIFTYWEEKLYOPTION. Reverting to default CFG
# End took 229913.71ms
        # SLEXIT + SL Entry Orders:
#         Total Return: 1664.96%
# Drawdown from Prev Peak: -280.58%
# Sharpe:  3.703480180058392
# Calamar:  -0.4113533782779639
# Num Trades:  972
# Avg Return per day: 4.98%
# Worst Day (2022-09-19): -103.04%
# Best Day (2022-08-30): 146.69%
# LimitExitOrders w SLEntry/Exit @ high+1brick w no limitEntry or svp:
#         Total Return: 1428.98%
# Drawdown from Prev Peak: -271.99%
# Sharpe:  3.2509694951617596
# Calamar:  -0.3641962175161762
# Num Trades:  3193
# Avg Return per day: 4.28%
# Worst Day (2023-02-08): -102.59%
# Best Day (2023-02-22): 154.75%
# No CFG for NIFTYWEEKLYOPTION. Reverting to default CFG

# LimitExitOrders w SLEntry/Exit @ high+2xbrick (hits!)w no limitEntry or svp:
# Total Return: 1710.82%
# Drawdown from Prev Peak: -271.13%
# Sharpe:  3.925853394706633
# Calamar:  -0.43741055933826223
# Num Trades:  1070
# Avg Return per day: 5.12%
# Worst Day (2022-09-19): -103.04%
# Best Day (2022-08-30): 111.83%
# No CFG for NIFTYWEEKLYOPTION. Reverting to default CFG
# End took 181827.44ms

# # LimitExitOrders w SLEntry/Exit @ high+2xbrick w limitEntry at brick low (no svp):
# l Return: 1699.99%
# Drawdown from Prev Peak: -234.71%
# Sharpe:  4.020836418708987
# Calamar:  -0.5020782891156211
# Num Trades:  996
# Avg Return per day: 5.09%
# Worst Day (2022-09-19): -83.23%

# # LimitExitOrders w SLEntry/Exit @ high+1.5xbrick w limitEntry at brick low (no svp):
#  SVP SLP only to switch limit orders for market for flying/crashing tickers (slope 1)
# Total Return: 1729.07%
# Drawdown from Prev Peak: -291.07%
# Sharpe:  3.9742024184294245
# Calamar:  -0.41179976648402455
# Num Trades:  1484
# Avg Return per day: 5.18%
# Worst Day (2022-09-19): -95.71%
# Best Day (2023-02-22): 124.41%

# # LimitExitOrders w SLEntry/Exit @ high+1.5xbrick w limitEntry at brick low (no svp):
#  SVP SLP only to skip limit orders for flying/crashing tickers (cfgSVPSlopeThreshold 5), limit exit price 1.4xbrick + brick high, no exit at static candles
# Total Return: 1896.59%
# Drawdown from Prev Peak: -291.25%
# Sharpe:  4.366464139404451
# Calamar:  -0.45141395649617566
# Num Trades:  1494

# CleanDF starts at 9:17
# Total Return: 1879.14%
# Drawdown from Prev Peak: -243.82%
# Sharpe:  4.517621577814576
# Calamar:  -0.5366068774062591
# Num Trades:  1485
# Avg Return per day: 5.63%
# Worst Day (2023-02-08): -93.67%

# 931 start
# Total Return: 1563.08%
# Drawdown from Prev Peak: -203.32%
# Sharpe:  3.913589396728039
# Calamar:  -0.5423920070602197
# Num Trades:  1291
# Avg Return per day: 4.68%
# Worst Day (2023-02-08): -110.16%
# Best Day (2023-02-22): 112.84%
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
        return row.vahShrtTrm if row.slpSTVah < cfgSVPSlopeThreshold else row['Adj Close'] + 10
    def getImmidiateSupport(self,row):
        #Use SVP
        return row.valShrtTrm if row.slpSTVal > -cfgSVPSlopeThreshold else row['Adj Close'] - 10
    def getSLPrice (self,row,type):
        (brickNum,brickSize,brickHigh,brickLow,close) = \
            (row['renko_brick_num'],row['renko_brick_high'] - row['renko_brick_low'],row['renko_brick_high'],row['renko_brick_low'],row['Adj Close'])
        lowEntrySL = brickLow - ((self.getNumBricksForShortTrend(row,type)-abs(brickNum)) * brickSize)
        highEntrySL = brickHigh + ((self.getNumBricksForLongTrend(row, type)-abs(brickNum)) * brickSize)
        lowExitSL = max(brickLow-brickSize,row["ShrtTrmLow"]-(brickSize/2)) if self.useSVPForEntryExitPrices else brickLow-brickSize
        highExitSL = min(brickHigh + brickSize,row["ShrtTrmHigh"] + (brickSize/2)) if self.useSVPForEntryExitPrices else brickHigh+brickSize
        ## SL are not aggressive enough for exit when fllying high above brick_high -- edit those to be val less brickSize, and also dont enter long if close < val (crashing)
        ## also add in some intel about round support levels .. nifty @ 100 multiple etc .. best to exit at 95 on shorts, and 105 on longs, and wait to enter on conclusive break
        
        if type == 'longEntrySL':
            return highEntrySL
        elif type == 'longExitSL':
            return -lowExitSL
        elif type == 'shortEntrySL':
            return -lowEntrySL
        elif type == 'shortExitSL':
            return highExitSL
        else:
            print(f"Unkonwn SL type {type} in getHighSL")
            exit(-1)
    def tickerIsMovingUpFast(self,row):
        return row.slpSTVah >= cfgSVPSlopeThreshold
        return abs(row['renko_brick_num']) < (cfgRenkoNumBricksForTrend + 2)#row.slpPoc >= cfgSVPSlopeThreshold
    def tickerIsMovingDownFast(self,row):   
        return row.slpSTVal <= -cfgSVPSlopeThreshold
        return abs(row['renko_brick_num']) < (cfgRenkoNumBricksForTrend + 2)#row.slpPoc <= -cfgSVPSlopeThreshold
    def getLimit1Price(self,row,type):
        (brickNum,brickSize,brickHigh,brickLow,staticCandles,close) = \
            (row['renko_brick_num'],row['renko_brick_high'] - row['renko_brick_low'],row['renko_brick_high'],row['renko_brick_low'],row['renko_static_candles'],row['Adj Close'])
        h = 1000
        l = 0 
        
        if self.exitStaticBricks and staticCandles >= cfgMinStaticCandlesForMeanRev :
            
            if self.useSVPForEntryExitPrices:
                h = self.getImmidiateResistance(row)
                l = self.getImmidiateSupport(row)
            else:
                h = (max(close,brickHigh) + (0.5*brickSize))
                l = min(close,brickLow - (0.5*brickSize))
        elif type == 'longEntryLimit1':
                if self.tickerIsMovingUpFast(row):
                    l = float('nan')
                else:
                    l = min(brickLow,close) if not self.useSVPForEntryExitPrices else self.getImmidiateSupport(row)
        elif type == 'shortEntryLimit1':
                if self.tickerIsMovingDownFast(row):
                    h = float('nan')
                else:
                    h = max(brickHigh,close) if not self.useSVPForEntryExitPrices else self.getImmidiateResistance(row)
        else: # Exit limit when candles are non-static or we are not using exiStaticBricks
            h = brickHigh + (1.4*brickSize)
            l = brickLow - (1.4*brickSize)
            h = h + brickSize if self.tickerIsMovingUpFast(row) else h
            l = l - brickSize if self.tickerIsMovingDownFast(row) else l

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
    # def isStatic(self,row):
    #     (brickNum,brickSize,brickHigh,brickLow,close,staticCandles) = (row['renko_brick_num'],row['renko_brick_high'] - row['renko_brick_low'],row['renko_brick_high'],row['renko_brick_low'],row['Adj Close'],row['renko_static_candles'])

    #     if  staticCandles >= cfgMinStaticCandlesForMeanRev
    #         return True
    #     else:
    #         if close > brickHigh and staticCandles < 5:
    #             #Newly entered candle, check the static candles of prev brick
    #             prevStaticCandles = row['renko_static_candles_prev']
            
    #MAIN
    def OkToEnterLong(self,row):
        return row['renko_uptrend']
    def OkToEnterShort(self,row):
        return not row['renko_uptrend']
    def checkLongEntry(self,s,row,df,isLastRow,limit1,limit2,sl1,sl2,logString):
        (brickNum,brickSize,brickHigh,brickLow,close) = (row['renko_brick_num'],row['renko_brick_high'] - row['renko_brick_low'],row['renko_brick_high'],row['renko_brick_low'],row['Adj Close'])
        
        # self.wontTrendToday(row,df)
        # print(row)
        if close < row.valShrtTrm:
            logging.info(f"Skipping Renko Long Entry for crashing ticker, as close {close} < valShrtTrm {row.valShrtTrm}") if isLastRow else None
            return (s, limit1, limit2, sl1, sl2,logString)

        if brickNum >= self.getNumBricksForLongTrend(row):
            # if self.meanRevAtStaticCandles and staticCandles > cfgMinStaticCandlesForMeanRev:
            limit1 = self.getLimit1Price(row,'longEntryLimit1') if self.limitEntryOrders else float('nan')
            if np.isnan(limit1):
                s = 1
                logString = "RENKO-LONG-ENTRY"
            # else:
            #     sl1 = min((brickHigh + brickSize),self.getImmidiateResistance(row)+5)
        elif self.slEntryOrders and brickNum >= 1:
            sl1 = self.getSLPrice(row,'longEntrySL') 
        return (s, limit1, limit2, sl1, sl2,logString)

    def checkShortEntry(self,s,row,df,isLastRow,limit1,limit2,sl1,sl2,logString):
        (brickNum,brickSize,brickHigh,brickLow,close) = (row['renko_brick_num'],row['renko_brick_high'] - row['renko_brick_low'],row['renko_brick_high'],row['renko_brick_low'],row['Adj Close'])
        self.wontTrendToday(row,df)

        if close > row.vahShrtTrm:
            logging.info(f"Skipping Renko Short Entry for rising ticker, as close {close} > vahShrtTrm {row.vahShrtTrm}") if isLastRow else None
            return (s, limit1, limit2, sl1, sl2,logString)

        if brickNum <= -self.getNumBricksForShortTrend(row):
            limit1 = self.getLimit1Price(row,'shortEntryLimit1') if self.limitEntryOrders else float('nan')
            if np.isnan(limit1):
                s = -1 
                logString = "RENKO-SHORT-ENTRY"
            # else:
            #     sl1 = -max((brickLow - brickSize),self.getImmidiateSupport(row)-5)

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
