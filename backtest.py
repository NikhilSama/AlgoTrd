#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  4 20:18:29 2023

@author: nikhilsama
"""

from datetime import date,timedelta
import tickerdata as td
import performance as perf
import numpy as np
import signals as signals
import pandas as pd
import matplotlib.pyplot as plt
import pprint


tickers = td.get_sp500_tickers()
nifty = td.get_nifty_tickers()
index_tickers = td.get_index_tickers()



#td.get_all_ticker_data()
end =date.today() - timedelta(days=2)
start =date.today() - timedelta(days=4)

df = td.get_ticker_data("KOTAKBANK", start,end, incl_options=False)
#ce_ticker = td.get_option_ticker("RELIANCE", df['Adj Close'][-1], 'CE')
#ddf = td.get_ticker_data(ce_ticker, start,end)

# Adding new column
df.insert(0, 'i', range(1, 1 + len(df)))
#ddf.insert(0, 'i', range(1, 1 + len(ddf)))
#signals.eom_effect(df)
#signals.sma50_bullish(df)
#df = signals.bollinger_band_cx(df)

df = signals.bollinger_band_cx_w_flat_superTrend(df)

#df['Open'] = ddf['Open']
#df['High'] = ddf['High']
#df['Low'] = ddf['Low']
#df['Close'] = ddf['Close']
#df['Adj Close'] = ddf['Adj Close']
#df.index[9]
#signals.ema_cx(df)
#signals.mystrat(df)
tearsheet,tearsheetdf = perf.tearsheet(df)

#df[['ma_superTrend', 'ma_slow', 'ma_fast']].plot(grid=True, figsize=(12, 8))
fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(8, 8))

# plot the first series in the first subplot
#ax1.plot(df['i'], df['ma_superTrend'], color='green', linewidth=2)
#ax1.plot(df['i'], df['ma20'], color='gold', linewidth=2)
ax1.plot(df['i'], df['Adj Close'], color='red', linewidth=2)
ax1.plot(df['i'], df['lower_band'], color='grey', linewidth=2)
ax1.plot(df['i'], df['upper_band'], color='grey', linewidth=2)
ax1.plot(df['i'], df['ma_superTrend'], color='orange', linewidth=4)

# plot the second series in the second subplot
ax2.plot(df['i'], df['ma_superTrend_pct_change'], color='red', linewidth=2)
#ax2.plot(df.index, df['superTrend'], color='red', linewidth=2)
#ax2.plot(df['i'], df['Adj Close-P'], color='blue', linewidth=2)
#ax2.plot(df['i'], df['Adj Close-C'], color='red', linewidth=2)

ax3.plot(df['i'], df['ma20_pct_change_ma'], color='green', linewidth=2)
#ax3.plot(df['i'], df['ma20_pct_change_ma'], color='red', linewidth=2)
ax4.plot(df['i'], df['cum_strategy_returns'], color='blue', linewidth=2)
ax5.plot(df['i'], df['position'], color='green', linewidth=2)

#ax3.plot(df.index, df['Volume'], color='red', linewidth=2)
#ax3.plot(df.index, df['obv'], color='green', linewidth=2)
#ax3.plot(df.index, df['ma_obv'], color='black', linewidth=2)

# display the plots
plt.show()
pprint.pprint(tearsheet)

df.to_csv("export.csv")
