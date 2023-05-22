#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  5 14:26:36 2023

@author: nikhilsama
"""
tickerCfg = {
    'ICICIBANK' : {
        'maLen': 10,
        'bandWidth': 2,
        'fastMALen': 5,
        'atrLen': 14,
        'adxLen': 20,
        'adxThresh': 30,
        'adxThreshYellowMultiplier': 1,
        'numCandlesForSlopeProjection':6,
        'maSlopeThresh': 1.5,
        'maSlopeThreshYellowMultiplier': 0.9,
        'cfgObvMaLen': 20,
        'obvOscThresh': 0.2,
        'obvOscThreshYellowMultiplier': 0.9,
        'bet_size': 10000
        },
    'HDFCBANK' : {
        'includeOptions': True,
        'maLen': 10,
        'bandWidth': 6,
        'fastMALen': 9,
        'atrLen': 14,
        'adxLen': 14,
        'adxThresh': 20,
        'adxThreshYellowMultiplier': 1,
        'numCandlesForSlopeProjection':2,
        'maSlopeThresh': .0002,
        'maSlopeThreshYellowMultiplier': 0.5,
        'obvOscThresh': 1,
        'bet_size': 10000          
        },
    

    'NIFTY' : {
        'maLen': 4,
        'maSlopeThresh': 0.01,
        'adxLen': 5,
        'adxThresh': 20,
        'cfgAdxThreshExitMultiplier': 0.8,
        'numCandlesForSlopeProjection':3,
        'cfgObvMaLen': 4,
        'cfgObvLen':10,
        'obvOscThresh': 0.1,
        'bet_size': 1000
    }, 
        # 'NIFTY' : {
        #     'maLen': 10,
        #     'bandWidth': 1,
        #     'cfgMiniBandWidthMult': 0.75, 
        #     'cfgSuperBandWidthMult':1.25,
        #     'fastMALen': 7,
        #     'atrLen': 14,
        #     'adxLen': 15,
        #     'adxThresh': 25,
        #     'adxThreshYellowMultiplier': 0.9,
        #     'numCandlesForSlopeProjection':2,
        #     'maSlopeThresh': 1,
        #     'maSlopeThreshYellowMultiplier': 0,
        # }, 
        
        

    'BANKNIFTY' : {
        'maLen': 15,
        'bandWidth': 3,
        'fastMALen': 7,
        'atrLen': 20,
        'adxLen': 20,
        'adxThresh': 35,
        'adxThreshYellowMultiplier': 0.7,
        'numCandlesForSlopeProjection':2,
        'maSlopeThresh': 1,
        'maSlopeThreshYellowMultiplier': 0.7,
        }
}


# NIFTY FOLLOW FAST MA
# mysql> select id,num_trades,drawdn,retrn,sharpe,avgRet,stdDev,skew,kurtosis,dayAv,dayShrp,dayStd,maLen,slpThres,adxLen,adxThresh,candles,atrLen from niftystratview where sg='JustFollowMA' order by sharpe desc limit 10;
# +--------+------------+--------+--------+--------+--------+--------+------+----------+-------+---------+--------+-------+----------+--------+-----------+---------+--------+
# | id     | num_trades | drawdn | retrn  | sharpe | avgRet | stdDev | skew | kurtosis | dayAv | dayShrp | dayStd | maLen | slpThres | adxLen | adxThresh | candles | atrLen |
# +--------+------------+--------+--------+--------+--------+--------+------+----------+-------+---------+--------+-------+----------+--------+-----------+---------+--------+
# | 710640 |         52 | -59.40 | 433.91 |   0.33 | 8.34   | 25.44  |  1.3 |      2.7 | 4.93  |    0.26 | 18.62  | 6     |      0.1 | 25     | 40        | 5       | 7      |
# | 706264 |         52 | -59.40 | 433.91 |   0.33 | 8.34   | 25.44  |  1.3 |      2.7 | 4.93  |    0.26 | 18.62  | 6     |      0.1 | 25     | 40        | 5       | 14     |
# | 706816 |         52 | -59.40 | 433.91 |   0.33 | 8.34   | 25.44  |  1.3 |      2.7 | 4.93  |    0.26 | 18.62  | 6     |      0.1 | 25     | 40        | 5       | 21     |
# | 706811 |         55 | -59.39 | 438.99 |   0.32 | 7.98   | 24.72  |  1.4 |        3 | 4.99  |    0.25 | 19.58  | 6     |     0.01 | 25     | 40        | 5       | 21     |
# | 706810 |         55 | -59.39 | 438.99 |   0.32 | 7.98   | 24.72  |  1.4 |        3 | 4.99  |    0.25 | 19.58  | 6     |        0 | 25     | 40        | 5       | 21     |
# | 710635 |         55 | -59.39 | 438.99 |   0.32 | 7.98   | 24.72  |  1.4 |        3 | 4.99  |    0.25 | 19.58  | 6     |     0.01 | 25     | 40        | 5       | 7      |
# | 710634 |         55 | -59.39 | 438.99 |   0.32 | 7.98   | 24.72  |  1.4 |        3 | 4.99  |    0.25 | 19.58  | 6     |        0 | 25     | 40        | 5       | 7      |
# | 706258 |         55 | -59.39 | 438.99 |   0.32 | 7.98   | 24.72  |  1.4 |        3 | 4.99  |    0.25 | 19.58  | 6     |        0 | 25     | 40        | 5       | 14     |
# | 706259 |         55 | -59.39 | 438.99 |   0.32 | 7.98   | 24.72  |  1.4 |        3 | 4.99  |    0.25 | 19.58  | 6     |     0.01 | 25     | 40        | 5       | 14     |
# | 706260 |         55 | -59.40 | 424.09 |   0.31 | 7.71   | 24.89  |  1.4 |        3 | 4.82  |    0.26 | 18.84  | 6     |     0.02 | 25     | 40        | 5       | 14     |
# +--------+------------+--------+--------+--------+--------+--------+------+----------+-------+---------+--------+-------+----------+--------+-----------+---------+--------+

    # 'NIFTY' : {
    #     'maLen': 6,
    #     'maSlopeThresh': 0.1,
    #     'adxLen': 25,
    #     'adxThresh': 40,
    #     'numCandlesForSlopeProjection':5,
    #     }, 

# down period 5/1/2022 to 5/2/2022
# mysql> select id,num_trades,drawdn,retrn,sharpe,avgRet,stdDev,dayAv,dayShrp,maLen,slpThres,adxLen,adxThresh,candles from niftystratview where endTime < '2022-06-01' order by sharpe desc limit 10;
# +--------+------------+--------+--------+--------+--------+--------+-------+---------+-------+----------+--------+-----------+---------+
# | id     | num_trades | drawdn | retrn  | sharpe | avgRet | stdDev | dayAv | dayShrp | maLen | slpThres | adxLen | adxThresh | candles |
# +--------+------------+--------+--------+--------+--------+--------+-------+---------+-------+----------+--------+-----------+---------+
# | 743562 |         17 | -31.77 | 225.39 |   0.31 | 13.26  | 43.00  | 7.51  |    0.23 | 2     |        0 | 5      | 80        | 2       |
# | 743551 |         18 | -31.77 | 248.29 |   0.31 | 13.79  | 43.85  | 8.28  |    0.23 | 4     |     0.01 | 5      | 80        | 2       |
# | 743046 |         18 | -31.77 | 248.29 |   0.31 | 13.79  | 43.85  | 8.28  |    0.23 | 4     |        0 | 5      | 80        | 2       |
# | 743047 |         18 | -31.77 | 248.29 |   0.31 | 13.79  | 43.85  | 8.28  |    0.23 | 4     |     0.01 | 5      | 80        | 2       |
# | 743563 |         17 | -31.77 | 225.39 |   0.31 | 13.26  | 43.00  | 7.51  |    0.23 | 2     |     0.01 | 5      | 80        | 2       |
# | 743054 |         17 | -31.77 | 227.94 |   0.31 | 13.41  | 43.34  | 7.60  |    0.23 | 2     |     0.05 | 5      | 80        | 2       |
# | 743053 |         17 | -31.77 | 225.39 |   0.31 | 13.26  | 43.00  | 7.51  |    0.23 | 2     |     0.01 | 5      | 80        | 2       |
# | 743052 |         17 | -31.77 | 225.39 |   0.31 | 13.26  | 43.00  | 7.51  |    0.23 | 2     |        0 | 5      | 80        | 2       |
# | 743564 |         17 | -31.77 | 227.94 |   0.31 | 13.41  | 43.34  | 7.60  |    0.23 | 2     |     0.05 | 5      | 80        | 2       |
# | 743055 |         17 | -31.77 | 240.96 |   0.31 | 14.17  | 45.17  | 8.03  |    0.23 | 2     |      0.1 | 5      | 80        | 2       |
# +--------+------------+--------+--------+--------+--------+--------+-------+---------+-------+----------+--------+-----------+---------+

    # 'NIFTY' : {
    #     'maLen': 4,
    #     'maSlopeThresh': 0.01,
    #     'adxLen': 5,
    #     'adxThresh': 80,
    #     'numCandlesForSlopeProjection':2,
    #     }, 
