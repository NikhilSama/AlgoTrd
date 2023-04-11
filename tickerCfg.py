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
        'maLen': 20,
        'bandWidth': 1,
        'fastMALen': 7,
        'atrLen': 20    ,
        'adxLen': 20,
        'adxThresh': 30,
        'adxThreshYellowMultiplier': 1,
        'numCandlesForSlopeProjection':6,
        'maSlopeThresh': 1.5,
        'maSlopeThreshYellowMultiplier': 0.9,
        'cfgObvMaLen': 20,
        'obvOscThresh': 0.4,
        'obvOscThreshYellowMultiplier': 1,
        'bet_size': 10000          
        }
}
tickerCfg['NIFTY'] = tickerCfg['HDFCBANK']