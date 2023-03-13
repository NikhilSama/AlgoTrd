#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 13:39:28 2023

@author: nikhilsama
"""

import kite_init as ki 
import pandas as pd

kite = ki.initKite()
nseInstrumentsDF = pd.DataFrame(kite.instruments(exchange=kite.EXCHANGE_NSE))


ohlc = kite.quote("NSE:RELIANCE")
print(ohlc)

