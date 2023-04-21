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
        'maSlopeThresh': 1,
        'maSlopeThreshYellowMultiplier': 0.5,
        'bet_size': 10000          
        },
    
    # 'NIFTY' : {
    #     'maLen': 30,
    #     'cfgMiniBandWidthMult': 0.75, 
    #     'bandWidth': 2.5,
    #     'cfgSuperBandWidthMult': 1.25,
    #     'fastMALen': 10,
    #     'atrLen': 14,
    #     'adxLen': 14,
    #     'adxThresh': 40,
    #     'adxThreshYellowMultiplier': 0.7,
    #     'numCandlesForSlopeProjection':2,
    #     'maSlopeThresh': 1.5,
    #     'maSlopeThreshYellowMultiplier': .7,
    #     'bet_size': 100000          
    #     }, 
        'NIFTY' : {
            'maLen': 30,
            'bandWidth': 1.5,
            'fastMALen': 10,
            'atrLen': 14,
            'adxLen': 10,
            'adxThresh': 25,
            'adxThreshYellowMultiplier': 0.7,
            'numCandlesForSlopeProjection':2,
            'maSlopeThresh': 0.01,
            'maSlopeThreshYellowMultiplier': 0,
        }, 

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
        'maSlopeThreshYellowMultiplier': 0,
        }
}


# NIFTY FOLLOW FAST MA
    # 'NIFTY' : {
    #     'maLen': 30,
    #     'bandWidth': 1.5,
    #     'fastMALen': 10,
    #     'atrLen': 14,
    #     'adxLen': 10,
    #     'adxThresh': 25,
    #     'adxThreshYellowMultiplier': 0.7,
    #     'numCandlesForSlopeProjection':2,
    #     'maSlopeThresh': 0.01,
    #     'maSlopeThreshYellowMultiplier': 0,
    #     }, 
