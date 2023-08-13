from pygments import highlight
from .SignalGenerator import SignalGenerator

import numpy as np
import pandas as pd
import logging
import cfg
globals().update(vars(cfg))

class Renko(SignalGenerator):
    logArray = ['voldelta','rsi','RenkoData','svp', 'svpST','bb','ohlv']
    limitExitOrders = False
    limitEntryOrders = False
    slEntryOrders = False
    slExitOrders = False
    exitStaticBricks = False
    useSVPForEntryExitPrices = False
    useVolDelta = False
    def __init__(self, limitExitOrders=False, limitEntryOrders=False, slEntryOrders=False, slExitOrders=False,exitStaticBricks=False,useSVPForEntryExitPrices=False,useVolDelta=False,**kwargs):
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

# with market re-entry for crashing/rising tickers
## Total Return: 2025.79%
# Drawdown from Prev Peak: -249.84%
# Sharpe:  4.817139057949768
# Calamar:  -0.564548048959726
# Num Trades:  1579
# Avg Return per day: 6.07%
# Worst Day (2023-02-08): -101.84%
# Best Day (2023-02-22): 121.66%
# No CFG for NIFTYWEEKLYOPTION. Reverting to default CFG
# End took 121344.2ms
#
# with limit re-entry even when ticker is falling/rising fast -- instead of market re-entry for crashing/rising tickers. Miss a few flyers, but avoid losses, lower returns but better sharpe and calamar
#Total Return: 1988.86%
# Drawdown from Prev Peak: -218.94%
# Sharpe:  4.852557073393357
# Calamar:  -0.6324899147893326
# Num Trades:  1503
# Avg Return per day: 5.95%
# Worst Day (2022-09-19): -94.64%
# Best Day (2023-01-09): 113.31%
# No CFG for NIFTYWEEKLYOPTION. Reverting to default CFG
# End took 147800.63ms

# # with limit re-entry even when ticker is falling/rising fast -- instead of market re-entry for crashing/rising tickers. Miss a few flyers, but avoid losses, lower returns but better sharpe and calamar
## but now limit exit is +1.5xbrick when ticker is flying, and -1.5xbrick when ticker is crashing (instead of 1x)
# Total Return: 2000.08%
# Drawdown from Prev Peak: -218.94%
# Sharpe:  4.87568511020904
# Calamar:  -0.636059832360184
# Num Trades:  1501
# Avg Return per day: 5.99%
# Worst Day (2022-09-19): -94.64%
# Best Day (2023-01-09): 113.31%
# No CFG for NIFTYWEEKLYOPTION. Reverting to default CFG
# End took 145914.35ms

# # with limit re-entry even when ticker is falling/rising fast and +/- 2 (1 is infereor) brickHigh/Low, intead of just brickHigh/Low-- instead of market re-entry for crashing/rising tickers. Miss a few flyers, but avoid losses, lower returns but better sharpe and calamar
## but now limit exit is +1.5xbrick when ticker is flying, and -1.5xbrick when ticker is crashing (instead of 1x)
# Total Return: 2176.62%
# Drawdown from Prev Peak: -216.26%
# Sharpe:  5.185646089502659
# Calamar:  -0.7007580451117692
# Num Trades:  1513
# Avg Return per day: 6.52%
# Worst Day (2023-02-08): -91.13%
# Best Day (2022-08-23): 117.20%
# No CFG for NIFTYWEEKLYOPTION. Reverting to default CFG
# End took 145694.77ms


# Total Return: 2153.78%
# Drawdown from Prev Peak: -203.05%
# Sharpe:  5.2131136630883805
# Calamar:  -0.738512365369062
# Num Trades:  1497
# Avg Return per day: 6.45%
# Worst Day (2022-09-19): -98.88%
# Best Day (2023-01-09): 117.18%
# No CFG for NIFTYWEEKLYOPTION. Reverting to default CFG
# End took 147082.03ms
        self.limitExitOrders = limitExitOrders
        self.limitEntryOrders = limitEntryOrders
        self.slEntryOrders = slEntryOrders
        self.slExitOrders = slExitOrders
        self.exitStaticBricks = exitStaticBricks
        self.useSVPForEntryExitPrices = useSVPForEntryExitPrices
        self.useVolDelta = useVolDelta
        super().__init__(**kwargs)

    def timeToLookForAExtremeExit(self,row,type):
        (close,vah,val,poc, slpPoc, slpVah, slpVal) = row[['Adj Close','vah','val',\
            'poc', 'slpPoc', 'slpVah', 'slpVal']]
        ret = False 
        
        if type == 'longExit':
            if (close > vah and slpVah < 0) or self.getVolDeltaSignal(row,type):
                ret = True
        elif type == 'shortExit':
            if (close < val and slpPoc > 0) or self.getVolDeltaSignal(row,type):
                ret = True          
        return ret
    
    def timeWindow(self,row):
        return True
        return row.name.time().hour > 10
        return row.name.time().hour < 11 or row.name.time().hour > 13

    #Helpers
    
    # def brickHighAtResistance(self,row):
    #     (nifty,brickHigh,brickLow) = row[['nifty','brickHigh','brickLow']]
    #     distFrom100 = nifty%100
        
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
    def noPrevDownTrend(self,row,df):
        df2 = df.loc[:row.name]
        last_negative_value = df2['renko_brick_num'].dropna().astype(int).where(lambda x: x < 0).dropna()
        return last_negative_value.empty
    def getNumBricksForLongTrend(self,row,df):
        noPrevDownTrend = self.noPrevDownTrend(row,df)    
        return cfgRenkoNumBricksForTrend #+ (1 if noPrevDownTrend else 0)#if getSVPquadrant(row) != 'Low' or row['slpPoc'] <= -cfgSVPSlopeThreshold else cfgRenkoNumBricksForTrend-1
    def getNumBricksForShortTrend(self,row):
        return cfgRenkoNumBricksForTrend #if getSVPquadrant(row) != 'High' or row['slpPoc'] >= cfgSVPSlopeThreshold else cfgRenkoNumBricksForTrend-1
    def getImmidiateResistance(self,row):
        #Use SVP
        return row.vahShrtTrm if row.slpSTVah < cfgSVPSlopeThreshold else row['Adj Close'] + 10
    def getImmidiateSupport(self,row):
        #Use SVP
        return row.valShrtTrm if row.slpSTVal > -cfgSVPSlopeThreshold else row['Adj Close'] - 10
        
    def getSLPrice (self,row,type,df,tradeHigh,tradeLow):
        (brickNum,brickSize,brickHigh,brickLow,close) = \
            (row['renko_brick_num'],row['renko_brick_high'] - row['renko_brick_low'],row['renko_brick_high'],row['renko_brick_low'],row['Adj Close'])
        lowEntrySL = brickLow - ((self.getNumBricksForShortTrend(row)-abs(brickNum)) * brickSize)
        highEntrySL = brickHigh + ((self.getNumBricksForLongTrend(row,df)-abs(brickNum)) * brickSize)
        lowEntrySL = min(row.ShrtTrmLow - 5,lowEntrySL)
        highEntrySL = max(row.ShrtTrmHigh + 5,highEntrySL)
        lowExitSL = max(brickLow-brickSize,row["ShrtTrmLow"]-(brickSize/2)) + 1 if self.useSVPForEntryExitPrices else max(brickLow-0.1,close-5)-brickSize
        highExitSL = min(brickHigh + brickSize,row["ShrtTrmHigh"] + (brickSize/2)) - 1 if self.useSVPForEntryExitPrices else min(brickHigh+0.1,close+5)+brickSize
        lowExitSL = min(row.ShrtTrmLow - 2,lowExitSL)
        highExitSL = max(row.ShrtTrmHigh + 2,highExitSL)

        
        # In case we entered brick 2 via stop loss on candle high/Low, but didnt close in the new brick, 
        # we should exit since new brick is not really formed
        # REMOVED -- THESE KICK in too ofetn, just 4 bucks below the ent
        # if brickNum > 0 and brickNum < self.getNumBricksForLongTrend(row,df): 
        #     lowExitSL = brickLow
        # elif brickNum < 0 and abs(brickNum) < self.getNumBricksForShortTrend(row)+1:
        #     highExitSL = brickHigh
        
        #Adjust SL based on VolDeltaSignal
        # if volDeltaSignal == 1:
        #     #Going up, so enter/exit sooner
        #     highEntrySL = max(close,highEntrySL-(brickSize/2))
        #     # highExitSL = max(close,highExitSL - (brickSize/2))
        #     lowEntrySL = min(close,lowEntrySL-(brickSize/2))
        #     # lowExitSL = min(close,lowExitSL - (brickSize/2))

        # elif volDeltaSignal == -1:
        #     #Going Down, so enter/exit sooner
        #     lowEntrySL = min(close,lowEntrySL+(brickSize/2))
        #     # lowExitSL = min(close,lowExitSL + (brickSize/2))
        #     highEntrySL = max(close,highEntrySL+(brickSize/2))
        #     # highExitSL = max(close,highExitSL +(brickSize/2))


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
        return self.marketIsBullish(row) #or (row.slpSTVah > 1 and row.slpPoc > 1)
        return row.ma20_pct_change >= maSlopeThresh or row["MA-FAST-SLP"] >= cfgFastMASlpThresh
        return row.slpSTVah >= cfgSVPSlopeThreshold
        return abs(row['renko_brick_num']) < (cfgRenkoNumBricksForTrend + 2)#row.slpPoc >= cfgSVPSlopeThreshold
    def tickerIsMovingDownFast(self,row):   
        return self.marketIsBearish(row) #or (row.slpSTVal < -1 and row.slpPoc < -1)
        return row.ma20_pct_change <= -maSlopeThresh or row["MA-FAST-SLP"] <= -cfgFastMASlpThresh
        return row.slpSTVal <= -cfgSVPSlopeThreshold
        return abs(row['renko_brick_num']) < (cfgRenkoNumBricksForTrend + 2)#row.slpPoc <= -cfgSVPSlopeThreshold
    def isFirstTradeForTrend (self,row,prevPosition,currTrend):
        if np.isnan(prevPosition) or prevPosition == 0:
            return True
        elif prevPosition == currTrend:
            return False
        else:
            return True
        
    def getLimit1Price(self,row,type,df,tradeEntry,tradeHigh,tradeLow):
        
        # (brickNum,brickSize,brickHigh,brickLow,staticCandles,close, vah, val, slpSTVah, slpSTVal) = \
        #     (row['renko_brick_num'],row['renko_brick_high'] - row['renko_brick_low'],row['renko_brick_high'],row['renko_brick_low'],row['renko_static_candles'],row['Adj Close'],row['vah'],row['val'],row['slpSTVah'],row['slpSTVal'])
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
                    l = min(brickHigh-2,row.Low) # float('nan') #  
                    l = max(row.pocShrtTrm,l)
                else:
                    l = min(brickLow-4,row.Low) if not self.useSVPForEntryExitPrices else self.getImmidiateSupport(row)
                    l = max(row.ShrtTrmLow,l)# l will win on a sudden steep rise, where ShrtTrmLow is too high, and will take a while to catch up
        elif type == 'shortEntryLimit1':
                if self.tickerIsMovingDownFast(row):
                    h = max(brickLow+2,row.High) #float('nan') # 
                    h = min(row.pocShrtTrm,h)
                else:
                    h = max(brickHigh+4,row.High) if not self.useSVPForEntryExitPrices else self.getImmidiateResistance(row)
                    h = min(row.ShrtTrmHigh,h) # h will win on a sudden steep fall, where ShrtTermHigh is too high, and will take a while to catch up
        else: # Exit limit when candles are non-static or we are not using exiStaticBricks
            h = min(brickHigh + (1.4*brickSize), tradeEntry* (1+cfgGoodTradeProfitPct))# if ((close-tradeEntry)/tradeEntry) < .3 else close
            l = max(brickLow - (1.4*brickSize), tradeEntry * (1-cfgGoodTradeProfitPct)) #if ((tradeEntry-close)/tradeEntry) < .3 else close
            h = h + (1.4*brickSize) if self.tickerIsMovingUpFast(row) or row.renko_static_candles <= 4 else h
            l = l - (1.4*brickSize) if self.tickerIsMovingDownFast(row) or row.renko_static_candles <= 4 else l
            h = max(h,row.ShrtTrmHigh+5)
            l = min(l,row.ShrtTrmLow-5)
            h += 15 if row.RSI < 75 else 0
            l -= 15 if row.RSI > 25 else 0

            # In case we entered brick 2 via stop loss on candle high/Low, but didnt close in the new brick, s
            # we shouldnt exit since new brick is not really formed
            if brickNum > 0 and brickNum < self.getNumBricksForLongTrend(row,df): 
                h += brickSize
            elif brickNum < 0 and abs(brickNum) < self.getNumBricksForShortTrend(row)+1:
                l -= brickSize
        
            # if abs(row.poc < 20):
            #     # We can only make new highs and new lows on very bullish or bearish 
            #     # volumes supported by trends.  Exit weak trends at day high/low
            #     h = min(h,max(row.dayHigh,row.ShrtTrmHigh)) if not self.marketIsBullish(row) else h
            #     l = max(l,min(row.dayLow,row.ShrtTrmLow)) if not self.marketIsBearish(row) else l
        
            # we enter longs only if poc slp > .1, if it changes and goes below zero, then exit longs early, at brickHigh
            # conversely we enter shorts w slop < -.1, if it changes and goes above zero, then exit shorts early, at brickLow
            # We have lost the bearish/bullish momentum that we entered with .. so exit early
            
            if row.slpPoc > 0 or self.marketIsBullish(row):
                l = brickLow
                l = max(row.pocShrtTrm,l)
                l = min(l,row.poc)                  
            elif row.slpPoc < 0 or self.marketIsBearish(row):
                h = brickHigh
                h = min(row.pocShrtTrm,h)
                h = max(h,row.poc)
            
                
                
            # if row.pocShrtTrm > row.renko_brick_low and \
            #     row.renko_static_candles >= 10:
            #     l = brickLow+.1
            # elif row.pocShrtTrm < row.renko_brick_high and \
            #     row.renko_static_candles >= 10:
            #     h = brickHigh-.1
            # Early exit at a decent price, when we seem to be clearly marching towards
            # the exit SL
            
            if row.pocShrtTrm > row.renko_brick_high and \
                row.renko_static_candles >= 5 and abs(row.renko_brick_num) <= 3:
                if row.High > row.renko_brick_high:
                    # We are truely close to exiting, poc and low are above brickHigh
                    l = min(brickHigh+.1,row.Low)
                # else: # poc is above brickHigh, but row high isnst
                #     #This happens when it falls really realy fast, so the poc hasnt caught up
                #     No Need to reset llimit value here, since we are already at a good price
                #     l = row.Low
            elif row.pocShrtTrm < row.renko_brick_low and \
                row.renko_static_candles >= 5 and abs(row.renko_brick_num) <= 3:
                if row.Low < row.renko_brick_low:
                    # We are truely close to exiting, poc and low are below brickLow
                    h = max(brickLow-.1,row.High)
                # else: # row Low has caught up w bricks, but poc has not
                #     # this happens when it rises really really fast, so the poc hasnt caught up
                #     No Need to reset llimit value here, since we are already at a good price
                #     h = row.High            
                
            if self.useSVPForEntryExitPrices:
                h = close - 2 if row.slpVah <= 0 and staticCandles > 2 else h
                l = close + 2 if row.slpVal >= 0 and staticCandles > 2 else l
                
            # if self.timeToLookForAExtremeExit(row,'longExit'):
            #     h = max(row.vahShrtTrm,tradeHigh)
            # if self.timeToLookForAExtremeExit(row,'shortExit'):
            #     l = min(row.valShrtTrm,tradeLow)       
                
            if self.timeToLookForAGoodExit(row):
                h = max(row.vahShrtTrm,row.High)
                l = min(row.valShrtTrm,row.Low)
        
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
    # printedDates = set()
    # def wontTrendToday(self,row,df):
    #     (dayHigh,dayLow,voh,vol) = row[['dayHigh','dayLow','vah','val']]
        
    #     if row.name.time() < cfgTimeToCheckDayTrendInfo:
    #         return False
    #     df = df[df.index.date == row.name.date()]
    #     uniqueRenkoBrickNum = df['renko_brick_num'].unique()
    #     maxRenko = np.nanmax(np.abs(uniqueRenkoBrickNum))

    #     if maxRenko < 2:
    #         print(f"{row.name.date()} is not trending") if row.name.date() not in self.printedDates else None
    #         self.printedDates.add(row.name.date())
    #         return True
    #     else:
    #         return False
    # def isStatic(self,row):
    #     (brickNum,brickSize,brickHigh,brickLow,close,staticCandles) = (row['renko_brick_num'],row['renko_brick_high'] - row['renko_brick_low'],row['renko_brick_high'],row['renko_brick_low'],row['Adj Close'],row['renko_static_candles'])

    #     if  staticCandles >= cfgMinStaticCandlesForMeanRev
    #         return True
    #     else:
    #         if close > brickHigh and staticCandles < 5:
    #             #Newly entered candle, check the static candles of prev brick
    #             prevStaticCandles = row['renko_static_candles_prev']

    def trendReversed(self,row,df):
        return False
    
        # on Futures and NIFTY, if we get one big candle in the reverse direction, with at least 3x the average volume of prev 30 candles
    def isBurst(self,type,row):
        if type == 'long' and row.Open - row['Adj Close'] >= 10: 
            return True
        elif type == 'short' and row['Adj Close'] - row.Open >= 10:
            return True
        return False
    #MAIN
    def OkToEnterLong(self,row):
        slp = row.slpPoc #if not np.isnan(row.slpPoc) else row["VWAP-SLP"]*100
        close = row['Adj Close']
        if row['renko_uptrend'] and self.timeWindow(row) and (not self.marketIsBearish(row)) \
            and (slp > 10 or \
                 (slp > 5 and close < row.vah) or \
                 (slp > -25 and close < row.val)):
                return True
        
        return False
        if self.marketIsBearish(row) or not self.timeWindow(row):
            return False # Cant go up with Bearish orderbook imbalance
        
        (uptrend,close) = (row['renko_uptrend'],row['Adj Close'])
        poc = row.poc if 'poc' in row else -1000
        
        if uptrend:
            if self.marketIsBullish(row):
                return True
            elif self.marketIsNeutral(row) and close < poc:
                return True
        return False
        
    def OkToEnterShort(self,row):
        slp = row.slpPoc #if not np.isnan(row.slpPoc) else row["VWAP-SLP"]*100
        close = row['Adj Close']

        if (not row['renko_uptrend']) and self.timeWindow(row) and (not self.marketIsBullish(row)) \
            and (slp < -10 or \
                (slp < -5 and close > row.val) or \
                (slp < 25 and row['Adj Close'] > row.vah)):
                return True
                    
        return False
        if self.marketIsBullish(row) or not self.timeWindow(row):
            return False
        
        (dntrend,close) = (not row['renko_uptrend'],row['Adj Close'])
        poc = row.poc if 'poc' in row else 1000
        
        if dntrend:
            if self.marketIsBearish(row):
                return True
            elif self.marketIsNeutral(row) and close > poc:
                return True
        return False

    def skipLongEntry(self,row,isLastRow,prevPosition):
        (brickNum) = (row['renko_brick_num'])
        # if abs(brickNum) >=10 and row.name.time().hour < 14:
        #     return True
        if self.getVolDeltaSignal(row,'longEntry') == 1:
            return True
        if not self.isFirstTradeForTrend(row,prevPosition,1) and \
            row.pocShrtTrm < row.renko_brick_low and \
                row.Low < row.renko_brick_low:
                #poc and row Low are below the brick_high,we may
                #have just exited the trend early, dont re-enter
                return True

        # (close,vah,poc,val,slpVah) = (row['Adj Close'],row['vah'],row['poc'],row['val'],row['slpVah'])
        # if close > poc and slpVah <= 0 and row.name.time() > cfgTimeToCheckDayTrendInfo:  x
        #     logging.info(f"Skipping Renko Long Entry Close:{close} > poc: {poc} and slpVah:{slpVah}") if isLastRow else None
            # return True
        return False
    def skipShortEntry(self,row,isLastRow,prevPosition):
        (brickNum) = (row['renko_brick_num'])
        # Doesnt work so commmenting out maxBrickThreshold
        # stocks come down the elevator; go up stairs
        # maxBrickThreshold = 20 if self.marketIsBearish(row) else 8
        # if abs(brickNum) >=maxBrickThreshold and row.name.time().hour < 14:
        #     return True
        if self.getVolDeltaSignal(row,'shortEntry') == 1:
            return True
        if not self.isFirstTradeForTrend(row,prevPosition,-1) and \
            row.pocShrtTrm > row.renko_brick_high and \
                row.High > row.renko_brick_high:
                #poc and row high are above the brick_high,we may
                #have just exited the trend early, dont re-enter
                return True
        # if row['Adj Close'] > row.renko_brick_high:
        #     return True
        # (close,vah,poc,val,slpVal) = (row['Adj Close'],row['vah'],row['poc'],row['val'],row['slpVal'])
        # if close < poc and slpVal >= 0 and row.name.time() > cfgTimeToCheckDayTrendInfo:
        #     logging.info(f"Skipping Renko Short Entry Close:{close} < poc: {poc} and slpVal:{slpVal}") if isLastRow else None
        #     return True
        return False
    def checkLongEntry(self,s,row,df,prevPosition,tradeHigh,tradeLow,isLastRow,limit1,limit2,sl1,sl2,logString):
        (brickNum,brickSize,brickHigh,brickLow,close,high) = (row['renko_brick_num'],row['renko_brick_high'] - row['renko_brick_low'],row['renko_brick_high'],row['renko_brick_low'],row['Adj Close'],row.High)
        tradeHigh = max(tradeHigh,row.High)
        tradeLow = min(tradeLow,row.Low)
        brickHurdle = self.getNumBricksForLongTrend(row,df)

        if self.isBurst('long',row):
            s = 1
            logString = "LONG-BURST-ENTRY"
            return (s, limit1, limit2, sl1, sl2,logString)


        if self.skipLongEntry(row,isLastRow,prevPosition):
            logging.info(f"Skipping Renko Long Entry") if isLastRow else None
            return (s, limit1, limit2, sl1, sl2,logString)
        # logging.info("Ticker is Moving Up Fast") if self.tickerIsMovingUpFast(row) and isLastRow else None
        if brickNum >= brickHurdle:
            # if self.meanRevAtStaticCandles and staticCandles > cfgMinStaticCandlesForMeanRev:
            # if close < brickLow, then we did an early SL exit, likely cause we were about to reverse, dont reenter 
            # until we get back up to the brickLow
            limit1 = self.getLimit1Price(row,'longEntryLimit1',df,0,tradeHigh,tradeLow) if self.limitEntryOrders else float('nan')
            # sl1 = self.getSLPrice(row,'longEntrySL',df,tradeHigh,tradeLow)
            if np.isnan(limit1):
                s = 1
                logString = "RENKO-LONG-ENTRY"
                sl1 = self.getSLPrice(row,'longExitSL',df,tradeHigh,tradeLow) 
        elif self.slEntryOrders and brickNum >= (brickHurdle-1):
            sl1 = self.getSLPrice(row,'longEntrySL',df,tradeHigh,tradeLow)
            if tradeHigh >= sl1:
                sl1 = float('nan')
                s = 1
                logString = "RENKO-LONG-ENTRY-SL"
            elif self.marketIsBullish(row):
                limit1 = sl1 - 8
            elif tradeHigh >= sl1 - 6:
                limit1 = max(self.getLimit1Price(row,'longEntryLimit1',df,0,tradeHigh,tradeLow), sl1-16) #self.getLimit1Price(row,'longEntryLimit1',df,0,tradeHigh,tradeLow)
        return (s, limit1, limit2, sl1, sl2,logString)

    def checkShortEntry(self,s,row,df,prevPosition,tradeHigh,tradeLow,isLastRow,limit1,limit2,sl1,sl2,logString):
        (brickNum,brickSize,brickHigh,brickLow,close,low) = (row['renko_brick_num'],row['renko_brick_high'] - row['renko_brick_low'],row['renko_brick_high'],row['renko_brick_low'],row['Adj Close'],row.Low)
        tradeHigh = max(tradeHigh,row.High)
        tradeLow = min(tradeLow,row.Low)
        brickHurdle = -self.getNumBricksForLongTrend(row,df)

        if self.isBurst('short',row):
            s = -1
            logString = "SHORT-BURST-ENTRY"
            return (s, limit1, limit2, sl1, sl2,logString)

        if self.skipShortEntry(row,isLastRow,prevPosition):
            logging.info(f"Skipping Renko Short Entry") if isLastRow else None
            return (s, limit1, limit2, sl1, sl2,logString)
        

        if brickNum <= brickHurdle:
            limit1 = self.getLimit1Price(row,'shortEntryLimit1',df,0,tradeHigh,tradeLow) if self.limitEntryOrders else float('nan')
            if np.isnan(limit1):
                s = -1 
                logString = "RENKO-SHORT-ENTRY"
                sl1 = self.getSLPrice(row,'shortExitSL',df,tradeHigh,tradeLow) 
        elif self.slEntryOrders and brickNum <= brickHurdle+1:
            sl1 = self.getSLPrice(row,'shortEntrySL',df,tradeHigh,tradeLow) 
            if tradeLow <= abs(sl1):
                sl1 = float('nan')
                s = -1
                logString = "RENKO-SHORT-ENTRY-SL"
            elif self.marketIsBearish(row):
                limit1 = -(abs(sl1) + 6)
            elif tradeLow <= (abs(sl1)+6):
                limit1 = max(-(abs(sl1)+9.5),self.getLimit1Price(row,'shortEntryLimit1',df,0,tradeHigh,tradeLow))
        return (s, limit1, limit2, sl1, sl2,logString)

    def checkLongExit(self,s,row,df,isLastRow, entryPrice,limit1,limit2,sl1,sl2,logString,
                      tradeEntry,tradeHigh,tradeLow):
        (brickNum,uptrend,brickSize,brickHigh,brickLow,close) = (row['renko_brick_num'],row['renko_uptrend'],row['renko_brick_high'] - row['renko_brick_low'],row['renko_brick_high'],row['renko_brick_low'],row['Adj Close'])
        tradeHigh = max(tradeHigh,row.High)
        tradeLow = min(tradeLow,row.Low)

        if (not uptrend) or self.trendReversed(row,df):
            if brickNum <= -cfgRenkoNumBricksForTrend:
                s = -1
                logString = "RENKO-LONG-EXIT-AND-SHORT-ENTRY"
            else:
                s = 0
                logString = "RENKO-LONG-EXIT"
                #Set SL for Short Entry
                if self.slEntryOrders and brickNum < 0:
                    sl1 = self.getSLPrice(row,'shortEntrySL',df,tradeHigh,tradeLow) ##CHECK this, tradeHigh/Low is for prev trade, dont think its used within getSLPrice, so ok, but fix
                    if row.Low <= abs(sl1):
                        sl1 = float('nan')
                        s = -1
                        logString = "RENKO-SHORT-ENTRY"

        elif self.getVolDeltaSignal(row,'longExit'):
            limit1 = -max(row.vahShrtTrm,tradeHigh) # We are goiing to re-enter at brickHigh - 2 via limit1 anyway, no point at exiting below this brickHigh+8 level for volDelta signal
        else:
            if self.slExitOrders:
                sl1 = self.getSLPrice(row,'longExitSL',df,tradeHigh,tradeLow) 
                if row.Low <= abs(sl1) and abs(sl1) < brickLow-8: 
                    #Use row.Low, not tradeLow here because tradeLow will contain all the lows from beginning of trade, sl1, moves up as bricks rise
                    #on a biiig up candle, sl can sometimes be too high based on close-5-bricksize, so when it goes up aggressively, it triggers
                    #to prevent this we added a new condition abs(sl1) < brickLow-8 to trigger this one
                    logString = f"RENKO-LONG-EXIT-SL{abs(sl1)}"
                    sl1 = float('nan')
                    s = 0
            if self.limitExitOrders:
                limit1 = self.getLimit1Price(row,'longExitLimit1',df,tradeEntry,tradeHigh,tradeLow) 
            # if tradeLow <= brickLow - 5:
            #     limit1 = -(brickLow)
        return (s, limit1, limit2, sl1, sl2,logString)

    def checkShortExit(self,s,row,df,isLastRow, entryPrice,limit1,limit2,sl1,sl2,logString,
                       tradeEntry,tradeHigh,tradeLow):
        (brickNum,uptrend,brickSize,brickHigh,brickLow,close) = (row['renko_brick_num'],row['renko_uptrend'],row['renko_brick_high'] - row['renko_brick_low'],row['renko_brick_high'],row['renko_brick_low'],row['Adj Close'])
        tradeHigh = max(tradeHigh,row.High)
        tradeLow = min(tradeLow,row.Low)

        if uptrend or self.trendReversed(row,df):
            if brickNum >= cfgRenkoNumBricksForTrend:
                s = 1
                logString = "RENKO-SHORT-EXIT-AND-LONG-ENTRY"
            else:
                s = 0
                logString = "RENKO-SHORT-EXIT"
                #Set SL for Long Entry
                if self.slEntryOrders and brickNum > 0:
                    sl1 = self.getSLPrice(row,'longEntrySL',df,tradeHigh,tradeLow)##CHECK this, tradeHigh/Low is for prev trade, dont think its used within getSLPrice, so ok, but fix
                    if row.High >= sl1:
                        sl1 = float('nan')
                        s = 1
                        logString = "RENKO-LONG-ENTRY-SL"

        elif self.getVolDeltaSignal(row,'shortExit'):
            limit1 = min(row.valShrtTrm,tradeLow)
        else:
            if self.slExitOrders:
                sl1 = self.getSLPrice(row,'shortExitSL',df,tradeHigh,tradeLow) 
                if row.High >= sl1 and sl1 > brickHigh+8:#Use row.High, not tradeHigh here because tradeHigh will contain all the high from beginning of trade, sl1, moves down as bricks fall
                    #on a biiig dn candle, sl can sometimes be too low based on close+5+bricksize, so when it goes dn aggressively, it triggers
                    #to prevent this we added a new condition abs(sl1) > brickHigh+8 to trigger this one
                    logString = "RENKO-SHORT-EXIT-SL"+f"{sl1} -- {row.High}"
                    sl1 = float('nan')
                    s = 0
            if self.limitExitOrders:
                limit1 = self.getLimit1Price(row,'shortExitLimit1',df,tradeEntry,tradeHigh,tradeLow)
            # if tradeHigh >= brickHigh + 5:
            #     limit1 = brickHigh

        return (s, limit1, limit2, sl1, sl2,logString)
