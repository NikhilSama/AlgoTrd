import os
import sys
current_dir = os.path.dirname(__file__)
root_dir = os.path.abspath(os.path.join(current_dir, '../'))
sys.path.append(root_dir)
sys.path.append(os.path.join(root_dir, 'connectors'))
import polygonAPI
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import utils
import random

def ATR(DF,n=20):
    df = DF.copy()
    df['H-L'] = df['High'] - df['Low']
    df['H-PC'] = df['High'] - df['Adj Close'].shift(1)
    df['L-PC'] = df['Low'] - df['Adj Close'].shift(1)
    df['TR'] = df[['H-L','H-PC','L-PC']].max(axis=1)
    df['ATR'] = df['TR'].ewm(com=n,min_periods=n).mean()
    return df['ATR']

def addBBStats(df, maLen = 20, cfgMASlopePeriods=3, bandWidth=2):

    # Check if ma20_pct_change is over threshold => uptrend, below negative threshold => downtrend, between positinve and negative threshold => flat
    # for flat market, you can use your logic, downward bias above ma and upward bias below ma
    # for upward market, we always have upward bias - UNLESS  if we are above upper band - that case try no trade, upward bias, downward bias, see what works best
    # vice versa for downward market
    # creating bollinger band indicators
    df['ma20'] = df['Adj Close'].shift(1).rolling(window=maLen).mean()
    df['ma20_pct_change'] = df['ma20'].pct_change(periods=cfgMASlopePeriods)/3
    df['std'] = df['Adj Close'].shift(1).rolling(window=maLen).std()
    df['upper_band'] = df['ma20'] + (bandWidth * df['std'])
    df['lower_band'] = df['ma20'] - (bandWidth * df['std'])
    df['atr'] = ATR(df,maLen)
    return df

def getSlopeThreshold(df,percentile=20):
    avSlp = df['ma20_pct_change'].mean()
    avABSSlp = df['ma20_pct_change'].abs().mean()
    medABSSlp = df['ma20_pct_change'].abs().median()
    # Drop NaN values before calculating percentile
    percentileThreshold = np.percentile(df['ma20_pct_change'].abs().dropna(), percentile)
    
    stdSlp = df['ma20_pct_change'].std()
    
    print(f"Average Slope: {avABSSlp:.5f} Std Slope: {stdSlp:.5f} Median Slope: {medABSSlp:.5f} 20% Slope: {percentileThreshold:.5f} ATR: {df['atr'].iloc[-1]:.2f}")
    return percentileThreshold

def get_market_regime(row,th=0.000055):
    
    if row['ma20_pct_change']>th:
        market_regime = "Uptrend"
    elif row['ma20_pct_change']<-th:
        market_regime = "Downtrend"
    else:
        market_regime ="Flat"
    return market_regime

def getSignal(row, th, window=3):
    return random.choice([-1, 0, 1])
    market_regime = get_market_regime(row, th)  # Pass the 'th' parameter to get_market_regime
    if market_regime == "Uptrend":
        Signal = 1
    elif market_regime == "Downtrend":
        Signal = -1
    else:
        Signal = 0
    return Signal

def isProfitable(s,strike,close,high,low,atr):
    pnl = 0
    threshold = atr/2.5
    if s == 1:
        if close > strike or high > strike+threshold:
            pnl = 1
        elif close < strike:
            pnl = -1
    elif s == -1:
        if close < strike or low < strike-threshold:
            pnl = 1
        elif close > strike:
            pnl = -1
    return pnl
def pnl(df, th=0.00005, num=1):
    df['Strike_Price'] = 0  # Initialize the 'Strike_Price' column
    df['Signal'] = 0  # Initialize the 'Signal' column
    prevIndex = None
    # 1:profitable trade, -1:Loss trade, 0:No trade
    for index, row in df.iterrows():
        Signal = getSignal(row,th)
        if Signal == 1:  # Prediction is "market will go up"
            Signal = 1
            Strike_Price = (np.floor(row['Open'] / num) * num)  # Selecting the lower strike price
            df.at[index, 'Strike_Price'] = Strike_Price
        elif Signal == -1 :  # Prediction is "Market will go down"
            Signal = -1
            Strike_Price = (np.ceil(row['Open'] / num) * num)  # Selecting the upper strike price
            df.at[index, 'Strike_Price'] = Strike_Price
        else:
            Signal = 0  # No Prediction
            Strike_Price = 0
        df.at[index, 'Signal'] = Signal
        df.at[index, 'rawPnL'] = isProfitable(Signal,Strike_Price,row['Adj Close'],row.High,row.Low,row.atr)
        # if (not prevIndex is None) and (df.at[prevIndex, 'rawPnL'] == -1):
        #     df.at[index, 'PnL'] = 0
        # else:
        #     df.at[index, 'PnL'] = df.at[index, 'rawPnL']
        df.at[index, 'PnL'] = df.at[index, 'rawPnL']
        prevIndex = index
        
    df.to_csv('df_5min_candle_prediction.csv')
    return df    

def printResults(df):
    pnl_counts = df['PnL'].value_counts()
    pnl_percent = df['PnL'].value_counts(normalize=True) * 100

    print("PnL counts in absolute numbers:\n", pnl_counts)
    print("\nPnL counts as percentage:\n", pnl_percent)
    if not pnl_counts.empty and 1 in pnl_counts.keys() and -1 in pnl_counts.keys() and  (pnl_counts[-1] + pnl_counts[1] > 0):
        print(f"profitable to lossy ratio: {pnl_counts[1]/pnl_counts[-1]} percent profitable:{pnl_counts[1]/(pnl_counts[-1] + pnl_counts[1]):.2%}")

def groupAndPrint(df):
    df['ma20_pct_change_abs'] = df['ma20_pct_change'].abs()
    bins = [0, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006]
    labels = ['0-0.001', '0.001-0.002', '0.002-0.003', '0.003-0.004', '0.004-0.005', '0.005-0.006']
    df['ma20_pct_change_group'] = pd.cut(df['ma20_pct_change_abs'], bins=bins, labels=labels)
    for name, group in df.groupby('ma20_pct_change_group'):
        print(f"RESULTS FOR MA SLOPE: {name}")
        printResults(group)

def getSignalForToday(df,th):
    signal = getSignal(df.iloc[-1],th)
    strike = exitPrice = 0
    atr  = df.iloc[-1].atr
    exitThresh = atr/2.5
    if signal == 1:
        strike = np.floor(df.iloc[-1]['Open'])
        exitPrice = strike + exitThresh
    elif signal == -1:
        strike = np.ceil(df.iloc[-1]['Open'])
        exitPrice = strike - exitThresh
    return (signal,strike,exitPrice)


# th is the percentile for slope threshold
def backtest(t,th=20):
    e = datetime.now() + timedelta(days=1)
    s = e - timedelta(days=6000)
    df = polygonAPI.polygonGet(t,s,e,"day")
    df = addBBStats(df)
    th = getSlopeThreshold(df,th)
    df= pnl(df, th=th, num=1)
    # groupAndPrint(df)
    printResults(df)

# th is the percentile for slope threshold
def predict(t,open, th=20):
    e = datetime.now() + timedelta(days=1)
    s = e - timedelta(days=6000)
    df = polygonAPI.polygonGet(t,s,e,"day")
    today_4am = datetime.now().replace(hour=4, minute=0, second=0, microsecond=0)
    if df.iloc[-1].name.date() < today_4am.date():
        #insert new row for today
        new_row = pd.Series({'Date': today_4am, 'Open': open})
        df = df.append(new_row, ignore_index=True)
    df = addBBStats(df)
    th = getSlopeThreshold(df,th)
    (signal,strike,exitPrice) = getSignalForToday(df,th)

    if signal == 1:
        print(f"Bull Call Spread {strike}/{strike-1} for $0.5")
        print(f"Exit spot at : {exitPrice}")
    elif signal == -1:
        print(f"Bear Put Spread {strike}/{strike+1} for $0.5")
        print(f"Exit spot at : {exitPrice}")
    else:
        print("No Trade Today")
    print(f"Open: {open} Slp: {df.iloc[-1]['ma20_pct_change']:.5f}")


def doAll(fn,th=20,open=None):
    tickers = utils.get_ndx_tickers()
    for t in tickers:
        print(f"Applying {t}")
        if open is None:
            fn(t,th)
        else:
            fn(t,open,th)
if __name__ == '__main__':
    slopePercentile = 1
    backtest("SPY", th=slopePercentile)
    # predict("SPY",447.24, th=slopePercentile)
    # doAll(backtest,th=slopePercentile)    
    # doAll(predict,th=slopePercentile,)