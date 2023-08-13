
import numpy as np
import pandas as pd
import math 
from datetime import date,timedelta,timezone
import datetime
import pytz
import pickle
import os

# signalGenerator = svb()
#cfg has all the config parameters make them all globals here
import cfg
globals().update(vars(cfg))

# set timezone to IST
ist = pytz.timezone('Asia/Kolkata')


## CORE ANALYTICS FUNCTIONS ##
def candleStickPatterns(df):
    candle_names = ta.get_function_groups()['Pattern Recognition']
    avVol = df.Volume.mean()
    for candle in candle_names:
        # below is same as;
        # df["CDL3LINESTRIKE"] = talib.CDL3LINESTRIKE(op, hi, lo, cl)
        df[candle] = getattr(ta, candle)(df['Open'], df['High'], df['Low'], df['Adj Close'])

    df['candlestick_pattern'] = np.nan
    df['candlestick_match_count'] = np.nan
    df['candlestick_signal'] = np.nan
    for index, row in df.iterrows():

        # no pattern found
        if row.Volume < 1.2*avVol:
            df.loc[index,'candlestick_pattern'] = "NO_VOLUME"
        elif len(row[candle_names]) - sum(row[candle_names] == 0) == 0:
            df.loc[index,'candlestick_pattern'] = "NO_PATTERN"
            df.loc[index, 'candlestick_match_count'] = 0
        # single pattern found
        elif len(row[candle_names]) - sum(row[candle_names] == 0) == 1:
            # bull pattern 100 or 200
            if any(row[candle_names].values > 0):
                pattern = list(compress(row[candle_names].keys(), row[candle_names].values != 0))[0] + '_Bull'
                df.loc[index, 'candlestick_pattern'] = pattern
                df.loc[index, 'candlestick_match_count'] = 1
                df.loc[index, 'candlestick_signal'] = 1
            # bear pattern -100 or -200
            else:
                pattern = list(compress(row[candle_names].keys(), row[candle_names].values != 0))[0] + '_Bear'
                df.loc[index, 'candlestick_pattern'] = pattern
                df.loc[index, 'candlestick_match_count'] = 1
                df.loc[index, 'candlestick_signal'] = -1

        # multiple patterns matched -- select best performance
        else:
            # filter out pattern names from bool list of values
            patterns = list(compress(row[candle_names].keys(), row[candle_names].values != 0))
            container = []
            for pattern in patterns:
                if row[pattern] > 0:
                    container.append(pattern + '_Bull')
                    df.loc[index, 'candlestick_signal'] = 1
                else:
                    container.append(pattern + '_Bear')
                    df.loc[index, 'candlestick_signal'] = -1
            rank_list = [candle_rankings[p] for p in container]
            if len(rank_list) == len(container):
                rank_index_best = rank_list.index(min(rank_list))
                df.loc[index, 'candlestick_pattern'] = container[rank_index_best]
                df.loc[index, 'candlestick_match_count'] = len(container)
    # clean up candle columns
    df.drop(candle_names, axis = 1, inplace = True)

    # hanging_man = ta.CDLHANGINGMAN(df['Open'], df['High'], df['Low'], df['Adj Close'])
    # return (hanging_man)
def MACD(DF,f=20,s=50):
    df = DF.copy()
    df["ma_fast"] = df["Adj Close"].ewm(span=f,min_periods=f).mean()
    df["ma_slow"] = df["Adj Close"].ewm(span=s,min_periods=s).mean()
    df["macd"] = df["ma_fast"] - df["ma_slow"]
    df["macd_signal"] = df["macd"].ewm(span=c, min_periods=c).mean()
    df['macd_his'] = df['macd'] = df['macd_signal']
    return(df['macd_signal'],df['macd_his'])

def OBV(df):
    df = df.copy()
    obv = ta.OBV(df['Adj Close'], df['Volume'])
    obv_pct_chang = obv.pct_change(periods=cfgObvLen).clip(-.1, .1)
    obv_osc = obv/obv.mean() - 1
    obv_osc_pct_chang = obv_osc.diff(cfgObvLen)/cfgObvLen
    return (obv_osc, obv_osc_pct_chang, obv, obv_pct_chang)
    # calculate the OBV column
    df['change'] = df['Adj Close'] - df['Open']
    df['direction'] = df['change'].apply(lambda x: 1 if x > 0 else -1 if x < 0 else 0)
    df['obv'] = df['direction'] * df['Volume']
    df['obv'] = df['obv'].rolling(window=cfgObvLen, min_periods=cfgObvLen).sum() # instead of cumsum; this restricts it to historical candles spec in cfg
    df['ma_obv'] = df['obv'].rolling(window=cfgObvMaLen, min_periods=2).mean()
    df['ma_obv_diff'] = df['ma_obv'].diff(5)
    
    #OBV-Diff Max/Min diff should only look at previous candles, not future candles
    #Also restrict the lookback to cfgMaxLookbackCandles, to keep backtest results consistent
    #apples to apples with live trading
    
    df['ma_obv_diff_max'] = df['ma_obv_diff'].rolling(window=cfgObvLen, min_periods=cfgObvLen).max()
    df['ma_obv_diff_min'] = df['ma_obv_diff'].rolling(window=cfgObvLen, min_periods=cfgObvLen).min()
    df['obv_osc'] = df['ma_obv_diff'] / (df['ma_obv_diff_max'] - df['ma_obv_diff_min'])
    df['obv_osc_pct_change'] = df['obv_osc'].diff(2)/2
    df['obv_trend'] = np.where(df['obv_osc'] > obvOscThresh,1,0)
    df['obv_trend'] = np.where(df['obv_osc'] < -obvOscThresh,-1,df['obv_trend'])
    
    # CLIP extreme
    df['obv_osc'] = df['obv_osc'].clip(lower=-1, upper=1)
    # df.to_csv("obv1.csv")
    # exit(0)
    return (df['ma_obv'],df['obv_osc'],df['obv_trend'],df['obv_osc_pct_change'])
def renko(DF):

    from stocktrends import Renko
    
    "function to convert ohlc data into renko bricks"
    df = DF.copy()
    df.reset_index(inplace=True)
    df = df.rename(columns= {'index': 'date'}) if 'index' in df.columns else df
    df = df.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Adj Close': 'close'})
    # df.columns.values[:10] = ["date", "i", "symbol", "open", "high", "low", "close", "volume", "signal", "ATR"]
    # df.columns = ["date","i","symbol","open","high","low","close","volume","signal","ATR"] if 'Volume-P' not in df.columns else ["date","i","open","high","low","close","volume","symbol","Open-P","High-P","Low-P","Adj Close-P","Volume-P","Strike-P","Open-C","High-C","Low-C","Adj Close-C","Volume-C","Strike-C","signal","ATR"]
    df = df.fillna(method='ffill')

    df2 = Renko(df)
    df2.brick_size = cfgRenkoBrickSize
    renko_df = df2.get_ohlc_data() #if using older version of the library please use get_bricks() instead
    renko_df["bar_num"] = np.where(renko_df["uptrend"]==True,1,np.where(renko_df["uptrend"]==False,-1,0))

    for i in range(1,len(renko_df["bar_num"])):
        if renko_df["bar_num"][i]>0 and renko_df["bar_num"][i-1]>0:
            renko_df["bar_num"][i]+=renko_df["bar_num"][i-1]
        elif renko_df["bar_num"][i]<0 and renko_df["bar_num"][i-1]<0:
            renko_df["bar_num"][i]+=renko_df["bar_num"][i-1]
    renko_df.drop_duplicates(subset="date",keep="last",inplace=True)
    renko_df["brick_size"] = cfgRenkoBrickSize
    
    renko_df.to_csv("renko.csv")
    # change leading uptrent to nan until a real trend is found
    mask = renko_df['bar_num'] == 1
    # Find the index where bar_num changes to a value other than 1
    change_idx = np.argmax(renko_df['bar_num'].values != 1)
    # Replace the initial rows with NaN values
    renko_df.loc[mask & (renko_df.index < change_idx), ['bar_num', 'uptrend']] = np.nan
    
    # Find the index where the first non-NaN negative integer appears in renko_brick_num
    first_negative_idx = renko_df[renko_df['bar_num'].notnull() & (renko_df['bar_num'] < 0)].index.min()
    # Reduce non-NaN leading positive integer values of renko_brick_num by 1
    mask = (renko_df.index < first_negative_idx) & renko_df['bar_num'].notnull() & (renko_df['bar_num'] > 0) if not np.isnan(first_negative_idx) else renko_df['bar_num'].notnull()
    renko_df.loc[mask, 'bar_num'] -= 1
    
    # renko_df['uptrend_change'] = renko_df['uptrend'].shift(-1) != renko_df['uptrend']
    # max_min_values = renko_df.loc[renko_df['uptrend_change'], 'bar_num'].tolist()

    # # The first element is not valid because it compares with NaN
    # max_min_values = max_min_values[1:]

    # print(max_min_values)
    # print(renko_df)

    return renko_df

def vwap(df):
    df['typical_price'] = (df['High'] + df['Low'] + df['Adj Close']) / 3
    df['VWAP'] = np.cumsum(df['typical_price'] * df['Volume']) / np.cumsum(df['Volume'])
    df['VWAP-SLP'] = df['VWAP'].diff(1)
    
    # Calculate Standard Deviation (SD) bands
    num_periods = 20  # Number of periods for SD calculation
    df['VWAP_SD'] = df['VWAP'].rolling(num_periods).std()  # Rolling standard deviation
    df['VWAP_upper'] = df['VWAP'] + 1 * df['VWAP_SD']  # Upper band (2 SD above VWAP)
    df['VWAP_lower'] = df['VWAP'] - 1 * df['VWAP_SD']  # Lower band (2 SD below VWAP)
    
# Function to calculate POC, VAH, VAL
def session_volume_profile (DF, type='normal', start_time=datetime.time(9, 15)):
    # calculate volume distribution
    if DF.empty:
        return (np.nan, np.nan, np.nan, np.nan, np.nan)
    if isinstance(DF.index, pd.DatetimeIndex):
        df = DF[DF.index.time > start_time].copy() # remove pre-market candles as they are huge volume (OI growth) and therefore have an undue influence on SVP
    else:
        df = DF.copy()
    if df.empty:
        return (np.nan, np.nan, np.nan, np.nan, np.nan)

    df['svp_price'] = (df['High']+df['Low']+df['Adj Close'])/3   
    volume_distribution = df.groupby('svp_price').sum()['Volume']
    # POC: price with max volume
    poc = volume_distribution.idxmax()
    
    # VAH, VAL: prices at 70% of volume
    volume_sorted = volume_distribution.sort_index()
    volume_cumsum = volume_sorted.cumsum()
    volume_total = volume_cumsum.iloc[-1]
    (t1,t2) = (0.15,0.85) if type == 'normal' else (0.1,0.9)
    if volume_total > 0: 
        val = volume_sorted[volume_cumsum <= volume_total * 0.15].index.max()  # 15% from top
        val = volume_sorted.index[0] if np.isnan(val) else val
        vah = volume_sorted[volume_cumsum >= volume_total * 0.85].index.min()  # 15% from bottom
        vah = volume_sorted.index[-1] if np.isnan(vah) else vah
    
        poc_val = volume_sorted[volume_cumsum <= volume_total * 0.495].index.max()  # 49% from top
        poc_vah = volume_sorted[volume_cumsum >= volume_total * 0.495].index.min()  # 49% from bottom
        poc = (poc_val + poc_vah)/2
    else:
        val = vah = poc = float('nan')
    dayHigh = volume_sorted.index.max()
    dayLow  = volume_sorted.index.min()
    return (poc, vah, val,dayHigh,dayLow)

def tenAMToday (now=0):
    if (now == 0):
        now = datetime.datetime.now(ist)

    # Set the target time to 8:00 AM
    tenAM = datetime.time(hour=10, minute=0, second=0, tzinfo=ist)
    
    # Combine the current date with the target time
    tenAMToday = datetime.datetime.combine(now.date(), tenAM)
    return tenAMToday

def ATR(DF,n=atrLen, type='option'):
    df = DF.copy()
    df['H-L'] = (df['niftyHigh'] - df['niftyLow']) if type == 'nifty' else (df['High'] - df['Low'])
    df['H-PC'] = (df['niftyHigh'] - df['nifty'].shift(1)) if type == 'nifty' else (df['High'] - df['Adj Close'].shift(1))
    df['L-PC'] = (df['niftyLow'] - df['nifty'].shift(1)) if type == 'nifty' else (df['Low'] - df['Adj Close'].shift(1))

    df['TR'] = df[['H-L','H-PC','L-PC']].max(axis=1)
    df['ATR'] = df['TR'].ewm(com=n,min_periods=cfgMinCandlesForMA).mean()
    return df['ATR']

def supertrend(df, multiplier=3, atr_period=10, type='option'):
    high = df['niftyHigh'] if type ==  'nifty' else df['High']
    low = df['niftyLow'] if type ==  'nifty' else df['Low']
    close = df['nifty'] if type ==  'nifty' else df['Adj Close']
    atr = df['niftyATR'] if type ==  'nifty' else df['ATR']
    # HL2 is simply the average of high and low prices
    hl2 = (high + low) / 2
    # upperband and lowerband calculation
    # notice that final bands are set to be equal to the respective bands
    final_upperband = upperband = hl2 + (multiplier * atr)
    final_lowerband = lowerband = hl2 - (multiplier * atr)
    # initialize Supertrend column to True
    supertrend = [True] * len(df)
    supertrend_signal = np.full(len(df), np.nan)
    
    for i in range(1, len(df.index)):
        curr, prev = i, i-1
        # if current close price crosses above upperband
        if close[curr] > final_upperband[prev]:
            supertrend[curr] = True
        # if current close price crosses below lowerband
        elif close[curr] < final_lowerband[prev]:
            supertrend[curr] = False
        # else, the trend continues
        else:
            supertrend[curr] = supertrend[prev]
            # adjustment to the final bands
            if supertrend[curr] == True and final_lowerband[curr] < final_lowerband[prev]:
                final_lowerband[curr] = final_lowerband[prev]
            if supertrend[curr] == False and final_upperband[curr] > final_upperband[prev]:
                final_upperband[curr] = final_upperband[prev]

        # to remove bands according to the trend direction
        if supertrend[curr] == True:
            final_upperband[curr] = np.nan
        else:
            final_lowerband[curr] = np.nan
        
        if supertrend[curr] and (not supertrend[prev]):
            supertrend_signal[curr] = 1
            
            # get an extra candle for the signal
            #cause our code is hacky.  If you use one signal to exit, then the next one need to also
            #contain the signal for us to enter
            if curr+2 < len(df):
                supertrend_signal[curr+1] = 1
            
        elif (not supertrend[curr]) and supertrend[prev]:
            supertrend_signal[curr] = -1 
            if curr+2 < len(df):
                supertrend_signal[curr+1] = -1  # get an extra candle for the signal
    return supertrend_signal,supertrend, final_upperband,final_lowerband


def ADX(DF, n=adxLen):
    adx = ta.ADX(DF['High'], DF['Low'], DF['Adj Close'], timeperiod=n)
    adx_pct_change = adx.diff(2)/2
    return (adx,adx_pct_change)

def RSI(DF, n=14):
    df = DF.copy()
    df["change"] = df["Adj Close"] - df["Adj Close"].shift(1)
    df["gain"] = np.where(df["change"]>=0, df["change"], 0)
    df["loss"] = np.where(df["change"]<0, -1*df["change"], 0)
    df["avgGain"] = df["gain"].ewm(alpha=1/n, min_periods=n).mean()
    df["avgLoss"] = df["loss"].ewm(alpha=1/n, min_periods=n).mean()
    df["rs"] = df["avgGain"]/df["avgLoss"]
    df["rsi"] = 100 - (100/ (1 + df["rs"]))
    return df["rsi"]

def addSMA(df,fast=8,slow=26,superTrend=200):
    
    period = df.index[1]- df.index[0]
    period_hours = period.total_seconds() / 3600
    
    df['ma_superTrend'] = df['Adj Close'].ewm(com=superTrend, min_periods=superTrend).mean()
    df['ma_superTrend_pct_change'] = df['ma_superTrend'].pct_change()
    df['ma_superTrend_pct_change_ma'] = df['ma_superTrend_pct_change'].ewm(com=fast, min_periods=5).mean()
    df['ma_superTrend_pct_change_ma_per_hr'] = df['ma_superTrend_pct_change_ma']/period_hours
    df['superTrend'] = np.where(df['ma_superTrend_pct_change_ma_per_hr'] > .002,1,0)
    df['superTrend'] = np.where(df['ma_superTrend_pct_change_ma_per_hr'] < -.002,-1, df['superTrend'] )

    
    df['ma_slow'] = df['Adj Close'].ewm(com=slow, min_periods=slow).mean()
    df['ma_fast'] = df['Adj Close'].ewm(com=fast, min_periods=fast).mean()

def addBBStats(df):
    # creating bollinger band indicators
    df['ma_superTrend'] = df['Adj Close'].ewm(com=superLen, min_periods=cfgMinCandlesForMA).mean()
    df['ma_superTrend_pct_change'] = 10000*df['ma_superTrend'].pct_change(periods=3)
    df['ma20'] = df['Adj Close'].rolling(window=maLen).mean()
    df['MA-FAST'] = df['Adj Close'].rolling(window=fastMALen).mean()
    df['MA-FAST-SLP'] = df['MA-FAST'].pct_change(periods=3)
    df['MA-FAST-SLP'] = df['MA-FAST-SLP'].clip(lower=-0.1, upper=0.1)
    df['ma20_pct_change'] = df['ma20'].pct_change(periods=cfgMASlopePeriods)
    df['ma20_pct_change_ma'] = df['ma20_pct_change'].ewm(com=5, min_periods=1).mean()
    df['ma20_pct_change_ma_sq'] = df['ma20_pct_change_ma'].pct_change()
    # df['ma20_pct_change_ma_sq'] = df['ma20_pct_change_ma_sq'].ewm(com=maLen, min_periods=maLen).mean()
    # df['ma20_pct_change_ma_sq'] = df['ma20_pct_change_ma_sq'].clip(lower=-0.5, upper=0.5)
    df['std'] = df['Adj Close'].rolling(window=maLen).std()
    df['upper_band'] = df['ma20'] + (bandWidth * df['std'])
    df['lower_band'] = df['ma20'] - (bandWidth * df['std'])
    df['mini_upper_band'] = df['ma20'] + (cfgMiniBandWidthMult*bandWidth * df['std'])
    df['mini_lower_band'] = df['ma20'] - (cfgMiniBandWidthMult*bandWidth * df['std'])
    df['super_upper_band'] = df['ma20'] + (cfgSuperBandWidthMult*bandWidth * df['std'])
    df['super_lower_band'] = df['ma20'] - (cfgSuperBandWidthMult*bandWidth * df['std'])
    #df.drop(['Open','High','Low'],axis=1,inplace=True,errors='ignore')
    #df.tail(5)
    slopeStdDev = df['ma20_pct_change_ma'].rolling(window=cfgMaxLookbackCandles,min_periods=maLen).std()
    slopeMean = df['ma20_pct_change_ma'].rolling(window=cfgMaxLookbackCandles,min_periods=maLen).mean()
    # df['SLOPE-OSC'] = (df['ma20_pct_change_ma'] - slopeMean)/slopeStdDev
    # df['SLOPE-OSC-SLOPE'] = df['SLOPE-OSC'].diff(2)/2
    
    df['SLOPE-OSC'] = df['ma20_pct_change']
    df['SLOPE-OSC-SLOPE'] = df['ma20_pct_change_ma_sq']
    return df
        
## END OF CORE ANALYTICS FUNCTIONS ##

## POPULATE DATA ANALYICS ##
####### POPULATE FUNCTIONS #######
# Functions that populate the dataframe with the indicators
def populateBB (df):
    addBBStats(df)

def populateATR(df):
    df['ATR'] = ATR(df,atrLen)
    # df['niftyATR'] = ATR(df,atrLen,'nifty')
    
def populateADX (df):
    (df['ADX'],df['ADX-PCT-CHNG']) = ADX(df,adxLen)

def populateRSI (df):
    df['RSI'] = RSI(df)
def populateOBV (df):
    # if (df['Volume'].max() == 0):
    #     return False # Index has no volume data so skip it
    (df['OBV-OSC'],df['OBV-OSC-PCT-CHNG'], df['OBV'], df['OBV-PCT-CHNG']) = OBV(df)
def populateSuperTrend (df):
    (df['SuperTrend'],df['SuperTrendDirection'],df['SuperTrendUpper'],df['SuperTrendLower']) = supertrend(df)

def populateCandleStickPatterns(df):
    #(df['HANGINGMAN']) = 
    candleStickPatterns(df)
def populateRenko(df):
    renkoDF = renko(df)
    renkoDF.columns = ["Date","open","renko_brick_high","renko_brick_low","close","uptrend","bar_num","brick_size"]
    df["Date"] = df.index
    # exit(0)
    # print(f"Max: {renkoDF['bar_num'].max()} Min: {renkoDF['bar_num'].min()}")
    df_renko_ohlc = df.merge(renkoDF.loc[:,["Date","uptrend","renko_brick_high","renko_brick_low","bar_num","brick_size"]],how="outer",on="Date")
    df_renko_ohlc["uptrend"].fillna(method='ffill',inplace=True)
    df_renko_ohlc["bar_num"].fillna(method='ffill',inplace=True)
    df_renko_ohlc["renko_brick_high"].fillna(method='ffill',inplace=True)
    df_renko_ohlc["renko_brick_low"].fillna(method='ffill',inplace=True)
    df_renko_ohlc.set_index('Date', drop=True, inplace=True)
    df['renko_uptrend'] = df_renko_ohlc['uptrend']
    df['renko_brick_num'] = df_renko_ohlc['bar_num']
    
    diff = df['renko_brick_num'].diff()
    df['renko_brick_diff'] = diff

    df['unique_brick_id'] = np.where(df['renko_brick_diff'] != 0, 
                                    df.index.astype(str) + df['renko_brick_num'].astype(str), 
                                    np.nan)
    df['unique_brick_id'].fillna(method='ffill', inplace=True)  # Forward fill the NaN values

    df['renko_brick_volume'] = df.groupby('unique_brick_id')['Volume'].transform('sum')
    df['renko_brick_volume_osc'] = df['renko_brick_volume']/np.mean(df['renko_brick_volume'].unique())
    # df = df.drop(columns='unique_brick_id')
    # df['renko_brick_vol'] = df_renko_ohlc['brick_volume']
    
    # Step 1: Filtering
    filtered_series = df.loc[df['renko_brick_diff'] == 0, 'renko_brick_diff']

    # Step 2: Grouping
    groups = (df['renko_brick_diff'] != 0).cumsum()

    # Step 3: Cumulative Counting
    cumulative_counts = filtered_series.groupby(groups).cumcount()

    # Step 4: Applying Conditions
    static_candles = cumulative_counts.where(df['renko_brick_diff'] == 0).fillna(0)

    # Assigning back to DataFrame
    df['renko_static_candles'] = static_candles

    df['renko_static_candles'].fillna(0)
    df['renko_static_candles'] = df['renko_static_candles'].where(df['renko_static_candles'] >= 1, 0)

    same_sign = np.sign(df['renko_brick_num']) == np.sign(df['renko_brick_num'].shift(1))
    renko_brick_diff = diff.where(same_sign, 0)
    df['renko_brick_diff'] = renko_brick_diff

    df['renko_brick_high'] = df_renko_ohlc['renko_brick_high']
    df['renko_brick_low'] = df_renko_ohlc['renko_brick_low']
    
    df.drop(["Date"],inplace=True,axis=1)
    df.to_csv('renko.csv')
    
def populateSVP(df):
    window_size = cfgFastSVPWindowSize  # 15-minute window size in terms of number of rows
    df['pocShrtTrm'] = df['vahShrtTrm'] = df['valShrtTrm'] = df['poc'] = df['vah'] = df['val']  = \
        df['slpPoc'] = df['slpVah'] = df['slpVal'] = df['slpSTPoc'] = df['slpSTVah'] = df['slpSTVal'] = \
            df['dayHigh']  = df['dayLow'] = df['ShrtTrmHigh']  = df['ShrtTrmLow'] = df['slpSTHigh']  = \
                df['slpSTLow'] = np.nan

    for end in range(window_size, len(df)):
        start = end-window_size
        window_df = df.iloc[start:end]
        (poc, vah, val,dayHigh,dayLow) = session_volume_profile(window_df,type='normal')
        df.iloc[end, df.columns.get_loc('pocShrtTrm')] = poc
        df.iloc[end, df.columns.get_loc('vahShrtTrm')] = vah
        df.iloc[end, df.columns.get_loc('valShrtTrm')] = val
        df.iloc[end, df.columns.get_loc('ShrtTrmHigh')] = dayHigh
        df.iloc[end, df.columns.get_loc('ShrtTrmLow')] = dayLow
        df.iloc[end, df.columns.get_loc('slpSTVah')] = (vah - df.iloc[end-3, df.columns.get_loc('vahShrtTrm')])#.rolling(window=5, min_periods=2).mean()
        df.iloc[end, df.columns.get_loc('slpSTVal')] = (val - df.iloc[end-3, df.columns.get_loc('valShrtTrm')])#.rolling(window=5, min_periods=2).mean()
        df.iloc[end, df.columns.get_loc('slpSTPoc')] = (poc - df.iloc[end-3, df.columns.get_loc('pocShrtTrm')])#.rolling(window=5, min_periods=2).mean()
        df.iloc[end, df.columns.get_loc('slpSTHigh')] = (dayHigh - df.iloc[end-2, df.columns.get_loc('ShrtTrmHigh')])#.rolling(window=5, min_periods=2).mean()
        df.iloc[end, df.columns.get_loc('slpSTLow')] = (dayLow - df.iloc[end-2, df.columns.get_loc('ShrtTrmLow')])#.rolling(window=5, min_periods=2).mean()

    for end in range(5, len(df)):
        start = 0
        window_df = df.iloc[start:end]
        (poc, vah, val,dayHigh,dayLow) = session_volume_profile(window_df,start_time=datetime.time(9,31)) 
        df.iloc[end, df.columns.get_loc('poc')] = poc
        df.iloc[end, df.columns.get_loc('vah')] = vah
        df.iloc[end, df.columns.get_loc('val')] = val   
        
        slpCandles = cfgSVPSlopeCandles
        pocSlpCandles = 2*slpCandles

        if np.isnan(df.iloc[end-pocSlpCandles, df.columns.get_loc('poc')]): 
            slpCandles = 1
            pocSlpCandles = slpCandles
            
        df.iloc[end, df.columns.get_loc('slpPoc')] = (poc - df.iloc[end-pocSlpCandles, df.columns.get_loc('poc')])/pocSlpCandles
        df.iloc[end, df.columns.get_loc('slpVah')] = (vah - df.iloc[end-slpCandles, df.columns.get_loc('vah')])/slpCandles
        df.iloc[end, df.columns.get_loc('slpVal')] = (val - df.iloc[end-slpCandles, df.columns.get_loc('val')])/slpCandles
        
        df.iloc[end, df.columns.get_loc('dayHigh')] = dayHigh
        df.iloc[end, df.columns.get_loc('dayLow')] = dayLow
        # print(f"start={start}, end={end}, poc={poc}, vah={vah}, val={val}")
    df['slpPoc'] = df['slpPoc'].ewm(span=cfgSVPSlopeCandles,min_periods=1).mean() * 100
    df['slpVah'] = df['slpVah'].ewm(span=cfgSVPSlopeCandles,min_periods=1).mean() * 100
    df['slpVal'] = df['slpVal'].ewm(span=cfgSVPSlopeCandles,min_periods=1).mean() * 100
    df['slpSTPoc'] = df['slpSTPoc'].rolling(window=10).mean()
    df['slpSDSTPoc'] = df['slpSTPoc'].rolling(window=10).std()
    df['slpSTPoc'].clip(lower=-3, upper=3, inplace=True)
    # df['slpVah'].clip(lower=0, upper=2, inplace=True)
    # df['slpVal'].clip(lower=-2, upper=0, inplace=True)

def populateVolDelta(df):
    if 'sellVol' not in df.columns:
        return # first df, does not have vol data
    df['maSellVol'] = df['sellVol'].rolling(window=1000,min_periods=1).mean()
    df['maBuyVol'] = df['buyVol'].rolling(window=1000,min_periods=1).mean()
    df['volDeltaThreshold'] = (df['maSellVol'] + df['maBuyVol']) * cfgVolDeltaThresholdMultiplier
    df['maVolDelta'] = df['volDelta'].rolling(window=5,min_periods=1).mean()
    df['stCumVolDelta'] = df['volDelta'].rolling(window=5,min_periods=1).sum()
    df['stMaxVolDelta'] = df['volDelta'].rolling(window=5,min_periods=1).max()
    df['stMinVolDelta'] = df['volDelta'].rolling(window=5,min_periods=1).min()
    df['cumVolDelta'] = df['volDelta'].cumsum()
    df.drop(['maSellVol','maBuyVol'], axis=1, inplace=True)
    


## END POPULATE DATA ANALYICS ##


#### ANALYTICS CACHE ####

def cacheAnalytics(df):
    fname = f"Data/analyticsCache/renko-{cfgRenkoNumBricksForTrend}-{df['symbol'][0]}-{df.index[0]}-{df.index[-1]}.pickle"
    with open(fname,"wb") as f:
        pickle.dump(df,f)

def hasCachedAnalytics(df):
    # return False
    fname = f"Data/analyticsCache/renko-{cfgRenkoNumBricksForTrend}-{df['symbol'][0]}-{df.index[0]}-{df.index[-1]}.pickle"
    if os.path.exists(fname):
        return True
    else:
        return False
def getCachedAnalytics(df):
    fname = f"Data/analyticsCache/renko-{cfgRenkoNumBricksForTrend}-{df['symbol'][0]}-{df.index[0]}-{df.index[-1]}.pickle"
    with open(fname, "rb") as f:
        df = pickle.load(f)
        print("Getting analytics from cache")
    return df


### END ANALYTICS CACHE ###

## MAIN ANALYTICS FUNCTIONS ##

def genAnalyticsForDay(df_daily,analyticsGenerators): 
    if df_daily.empty or len(df_daily) < 2:
        return df_daily
    for analGen in analyticsGenerators:
        analGen(df_daily)
    
    return df_daily

def generateAnalyticsForFreq(analyticsGenerators,df,freq):
    df.index = pd.to_datetime(df.index, utc=True)
    # Convert timezone-aware datetime index from UTC to IST
    df.index = df.index.tz_convert(ist)
    
    # Assuming the input dataframe is 'df' with a datetime index
    # 1. Split the dataframe into separate dataframes for each day
    daily_dataframes = [group for _, group in df.groupby(pd.Grouper(freq=freq))]

    # 2. Run the 'genAnalyticsForDay' function on each day's dataframe
    daily_analytics = [genAnalyticsForDay(day_df, analyticsGenerators) for day_df in daily_dataframes]

    # 3. Combine the resulting pandas series from the analytics function
    combined_analytics = pd.concat(daily_analytics)

    # 4. Merge the combined series with the original dataframe 'df'
    df_with_analytics = df.merge(combined_analytics, left_index=True, right_index=True)
    return combined_analytics

def generateAnalytics(analyticsGenerators,df):
    if not analyticsGenerators['nofreq']:
        dfWithAnalytics = generateAnalyticsForFreq(analyticsGenerators['hourly'],df,'H')
        dfWithAnalytics = generateAnalyticsForFreq(analyticsGenerators['daily'],dfWithAnalytics,'D')
    else:
        dfWithAnalytics = genAnalyticsForDay(df,analyticsGenerators['nofreq'])
    return dfWithAnalytics

## END MAIN ANALYTICS FUNCTIONS ##