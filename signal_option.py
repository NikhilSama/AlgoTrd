import signals
import numpy as np
import pandas as pd
import cfg
import utils
import pytz
from DatabaseLogin import DBBasic
import math
db = DBBasic() 

globals().update(vars(cfg))

# set timezone to IST
ist = pytz.timezone('Asia/Kolkata')

# Ticker Status 
tickerStatus = {}

def initTickerStatus(ticker):
#    if ticker not in tickerStatus.keys():
    tickerStatus[ticker] = {'position':float("nan"), 
                            'legs':[],
                            }     
def tickerHasNoPosition(t):
    return np.isnan(tickerStatus[t]['position']) or \
        tickerStatus[t]['position'] == 0
def getTickerPosition(ticker):
    if ticker not in tickerStatus:
        return float ('nan')
    elif 'position' not in tickerStatus[ticker]:
        return float ('nan')
    else:
        return tickerStatus[ticker]['position']
def getTickerPrevPosition(ticker):
    if ticker not in tickerStatus:
        return float ('nan')
    elif 'prevPosition' not in tickerStatus[ticker]:
        return float ('nan')
    else:
        return tickerStatus[ticker]['prevPosition']
def tickerHasPosition(ticker):
    return (not np.isnan(getTickerPosition(ticker))
                and getTickerPosition(ticker) != 0)
def tickerHasLongPosition(ticker):
    return getTickerPosition(ticker) == 1
def tickerHasShortPosition(ticker):
    return getTickerPosition(ticker) == -1
def getDefaultTickerMaxPrice():
    return int(0)
def getDefaultTickerMinPrice():
    return int(10000000)
def setTickerMaxPriceForTrade(ticker,entry_price,high,low,signal):
    default_min = getDefaultTickerMinPrice()
    default_max = getDefaultTickerMaxPrice()
    if 'max_price' not in tickerStatus[ticker]:
        tickerStatus[ticker]['max_price'] = default_max
        tickerStatus[ticker]['min_price'] = default_min
    if np.isnan(signal):
        tickerStatus[ticker]['max_price'] = round(max(abs(high),tickerStatus[ticker]['max_price']))
        tickerStatus[ticker]['min_price'] = round(min(abs(low),tickerStatus[ticker]['min_price']))
    else: # signal is 1/-1/0, start or end of a new trade
        tickerStatus[ticker]['max_price'] = tickerStatus[ticker]['min_price'] = round(abs(entry_price),1)
    
def setTickerPosition(ticker,signal, entry_price,high, low,strategy,opt1,opt1_signal,opt1_price,opt2,opt2_signal,opt2_price,opt3,opt3_signal,opt3_price,opt4,opt4_signal,opt4_price):
    # logging.info(f"setting ticker position for {ticker} to {signal} entry{entry_price} pos:{tickerStatus[ticker]['position']}")
    setTickerMaxPriceForTrade(ticker,entry_price,high,low,signal)
    # (tickerStatus[ticker]['limit1'], tickerStatus[ticker]['limit2'], tickerStatus[ticker]['sl1'], tickerStatus[ticker]['sl2']) = (limit1, limit2, sl1, sl2)
    #logging.info(f"signal: {signal} entry: {tickerStatus[ticker]['entry_price']} max: {tickerStatus[ticker]['max_price']} min: {tickerStatus[ticker]['min_price']}")
    if np.isnan(signal):
        return # nan signal does not change position
    if ticker not in tickerStatus:
        tickerStatus[ticker] = {}
    if tickerStatus[ticker]['position'] != signal:
        if tickerStatus[ticker]['position'] != 0:
            # We had a position, likely now exiting with signal = 0
            print("Setting opt tickers")
            print(f"signal: {signal} position: {tickerStatus[ticker]['position']}")
            #Store last long or short position
            tickerStatus[ticker]['prevPosition'] = tickerStatus[ticker]['position']
        if signal == 0:
            tickerStatus[ticker]['entry_price'] = float('nan')
            tickerStatus[ticker]['strategy'] = float('nan')
            tickerStatus[ticker]['opt1'] = float('nan')
            tickerStatus[ticker]['opt1_signal'] = float('nan')
            tickerStatus[ticker]['opt1_price'] = float('nan')
            tickerStatus[ticker]['opt2'] = float('nan')
            tickerStatus[ticker]['opt2_signal'] = float('nan')
            tickerStatus[ticker]['opt2_price'] = float('nan')
            tickerStatus[ticker]['opt3'] = float('nan')
            tickerStatus[ticker]['opt3_signal'] = float('nan')
            tickerStatus[ticker]['opt3_price'] = float('nan')
            tickerStatus[ticker]['opt4'] = float('nan')
            tickerStatus[ticker]['opt4_signal'] = float('nan')
            tickerStatus[ticker]['opt4_price'] = float('nan')

        else:
            tickerStatus[ticker]['entry_price'] = round(entry_price,1)
            tickerStatus[ticker]['strategy'] = strategy
            tickerStatus[ticker]['opt1'] = opt1
            tickerStatus[ticker]['opt1_signal'] = opt1_signal
            tickerStatus[ticker]['opt1_price'] = opt1_price
            tickerStatus[ticker]['opt2'] = opt2
            tickerStatus[ticker]['opt2_signal'] = opt2_signal
            tickerStatus[ticker]['opt2_price'] = opt2_price
            tickerStatus[ticker]['opt3'] = opt3
            tickerStatus[ticker]['opt3_signal'] = opt3_signal
            tickerStatus[ticker]['opt3_price'] = opt3_price
            tickerStatus[ticker]['opt4'] = opt4
            tickerStatus[ticker]['opt4_signal'] = opt4_signal
            tickerStatus[ticker]['opt4_price'] = opt4_price


        tickerStatus[ticker]['position'] = signal

def getTickerEntryPrice(ticker):
    return round(abs(tickerStatus[ticker]['entry_price'])) if 'entry_price' in tickerStatus[ticker] and not np.isnan(tickerStatus[ticker]['entry_price']) else 0
def getTickerMaxPrice(ticker):
    return abs(tickerStatus[ticker]['max_price'])
def getTickerMinPrice(ticker):
    return abs(tickerStatus[ticker]['min_price'])

def logSignal(row,msg):
    print(f'{row.name.strftime("%d/%m %I:%M")}: {msg}')
    
## END OF TICKER STATUS
def getShortStraddle(row):
    close = round(row['Adj Close']/100)*100
    putStrike = close
    callStrike = close
    putTicker = utils.getOptionTicker(row.name,'NIFTY',putStrike,'PE')
    putPrice = db.getNiftyOptionPrice(putTicker,row.name)
    callTicker = utils.getOptionTicker(row.name,'NIFTY',callStrike,'CE')
    callPrice = db.getNiftyOptionPrice(callTicker,row.name)

    return (putTicker,putPrice,-1,callTicker,callPrice,-1)
    
def isWeeklyExpiryTime(d):
    if d.date() == utils.getWeeklyOptionExpiryDate(d,minDaysToExpiry=0):
        if d.time() > datetime.datetime.strptime("15:14+05:30", "%H:%M%z").time():
            return True
    return False

def getVerticalSpreadStrikes(row):
    (ma,maSlp,atr) = (row['ma20'],row['ma20_pct_change'],row.ATR)
    expiry = utils.getWeeklyOptionExpiryDate(row.name,minDaysToExpiry=2)
    daysToExpiry = (expiry - row.name).days
    atr = 200 if np.isnan(atr) else atr
    maSlp = maSlp/3 if not np.isnan(maSlp) else 0# convert 3 period slope to 1 period slope
    projectedGrowth = ((1+maSlp)**daysToExpiry)
    projectedClose = projectedGrowth*row.Open
    projectedClose = row.Open
    upper = projectedClose #+ atr
    lower = projectedClose #- atr
    # upper = max(upper,ma)
    # lower = min(lower,ma)
    upper = math.ceil(upper/50)*50
    lower = math.floor(lower/50)*50
    print(f"open:{row.Open} ma:{ma} maSlp:{maSlp} daysToExpiry:{daysToExpiry} atr:{atr} projectedGrowth:{projectedGrowth} projectedClose:{projectedClose} upper:{upper} lower:{lower}")
    return (upper,lower)

def getVerticalSpread(row,sellStrike,buyStrike,type):
    sellTicker = utils.getOptionTicker(row.name,'NIFTY',sellStrike,type,minDaysToExpiry=2)
    sellPrice = db.getNiftyOptionPrice(sellTicker,row.name)
    buyTicker = utils.getOptionTicker(row.name,'NIFTY',buyStrike, type,minDaysToExpiry=2)
    buyPrice = db.getNiftyOptionPrice(buyTicker,row.name)

    return (sellTicker,sellStrike,sellPrice,-1,buyTicker,buyStrike,buyPrice,1)

def getUnderlyingSignal(row):
    slp = row['ma20_pct_change']
    superTrend = row.SuperTrendDirection
    weekday = row.name.weekday()
    if slp > 0 and superTrend and weekday == 2:
        return 1
    elif slp < 0 and not superTrend and weekday == 2:
        return -1
    else:
        return float('nan')

def verticalSpread(type,s,isLastRow,row,df):
    signal = getUnderlyingSignal(row)
    strategy = "Vertical Spread"
    opt1 = opt1_strike = opt1_signal = opt2 = opt2_strike = opt2_signal = \
        opt3 = opt3_strike = opt3_signal = opt4 = opt4_strike = opt4_signal = \
            opt1_price = opt2_price = opt3_price = opt4_price = float("nan")
    row_date = row.name.date() if isinstance(row.name, pd.DatetimeIndex) else row.name
    position = getTickerPosition(row.symbol)
    entryPrice = getTickerEntryPrice(row.symbol)
    close = row['Adj Close']

    if isWeeklyExpiryTime(row.name):
        if tickerHasPosition(row.symbol):
            msg = "Close Position - Expiry"
            s = 0
            exitPriceType = 'Close'
        else:
            msg = 'No Position at expirty'
    elif tickerHasNoPosition(df['symbol'][0]):
        (upper,lower) = getVerticalSpreadStrikes(row)
        if (signal == 1):#Bullish
            strategy = "Bull Put Spread"
            (opt1,opt1_strike,opt1_price,opt1_signal, opt2, opt2_strike, opt2_price, opt2_signal) = getVerticalSpread(row,lower,lower-100,'PE')
            s = 1
        elif signal == -1: #Bearish
            strategy = "Bear Call Spread"
            (opt1,opt1_strike,opt1_price,opt1_signal, opt2, opt2_strike, opt2_price, opt2_signal) = getVerticalSpread(row,upper,upper+100,'CE')
            s = -1
        else:
            strategy = "None"
        msg = f"New position - {strategy} close:{row['Adj Close']} opt1:{opt1} opt2:{opt2}"
    elif not np.isnan(signal):
        # Ticker has a position, and we not at expiry
        # and we have a non-nan signal
        
        if position != signal:
            msg = f"Close Position - Signal Changed Close{close}"
            s = 0
            exitPriceType = 'Close'
        else:
            msg = "continue"
    else: # ticker has a position, we not at expiry, but no signal
        if position == 1 and (close - entryPrice) >= 100:
            msg = f"Close Position - Take Profit close:{close}"
            s = 0 # take profit and re-enter
            exitPriceType = 'Close'
        # elif position == -1 and (entryPrice - close) >= 100:
        #     msg = f"Close Position - Take Profit close:{close}"
        #     s = 0
        #     exitPriceType = 'Close'
        else:
            msg = "continue"
    
    
    if s == 0: 
        opt1 =  tickerStatus[row.symbol]['opt1']
        opt2 =  tickerStatus[row.symbol]['opt2']
        opt1_price = db.getNiftyOptionPrice(opt1,row.name,exitPriceType)
        opt2_price = db.getNiftyOptionPrice(opt2,row.name)
        opt1_signal = opt2_signal = opt3_signal = opt4_signal = 0
        msg += f" opt1:{opt1} opt2:{opt2}"
        
    logSignal(row,msg)
    return (s,strategy,opt1,opt1_strike,opt1_signal,opt1_price,opt2,opt2_strike,opt2_signal,opt2_price,opt3,opt3_strike,opt3_signal,opt3_price,opt4,opt4_strike,opt4_signal,opt4_price) # Outside of trading hours EXIT ALL POSITIONS

def shortStraddle(type,s,isLastRow,row,df):
    (close,t) = (row['Adj Close'],row.symbol)
    strategy = "Short Straddle"
    opt1 = opt1_signal = opt2 = opt2_signal = \
        opt3 = opt3_signal = opt4 = opt4_signal = \
            opt1_price = opt2_price = opt3_price = opt4_price = float("nan")
    if tickerHasNoPosition(df['symbol'][0]):
        logSignal("New position")
        (opt1,opt1_price,opt1_signal,opt2,opt2_price,opt2_signal) = getShortStraddle(row)
        s = -1
    elif abs(getTickerEntryPrice(t) - close) > 100 or isWeeklyExpiryTime(row.name):
        logSignal("closing position")
        (opt1,opt2) = (tickerStatus[row.symbol]['opt1'],tickerStatus[row.symbol]['opt2'])
        (opt1_price,opt2_price) = (db.getNiftyOptionPrice(opt1,row.name),db.getNiftyOptionPrice(opt2,row.name))
        (opt1_signal,opt2_signal) = (0,0)
        s = 0
    return (s,strategy,opt1,opt1_signal,opt1_price,opt2,opt2_signal,opt2_price,opt3,opt3_signal,opt3_price,opt4,opt4_signal,opt4_price) # Outside of trading hours EXIT ALL POSITIONS

def weCanEnterNewTrades(row):
    return True
def getSignal(row,signalGenerators, df):
    s = strategy = opt1 = opt1_strike = opt1_signal = opt2 = opt2_strike = opt2_signal = \
        opt3 = opt3_strike = opt3_signal = opt4 = opt4_strike = opt4_signal = \
            opt1_price = opt2_price = opt3_price = opt4_price =  float("nan")
    isLastRow = (row.name == df.index[-1]) or cfgIsBackTest
   
    #Return nan if its not within trading hours
    if (not isinstance(df.index, pd.DatetimeIndex)) or row.name.time() >= cfgStartTimeOfDay:
            if weCanEnterNewTrades(row):
                type = 1 # Entry or Exit
            elif row.name.time() < cfgEndExitTradesOnlyTimeOfDay:
                # Last time period before intraday exit; only exit positions
                # No new psitions will be entered
                type = 0 
            else:
                # Exit at end of day
                return (0,strategy,opt1,opt1_signal,opt1_price,opt2,opt2_signal,opt2_price,opt3,opt3_signal,opt3_price,opt4,opt4_signal,opt4_price) # Outside of trading hours EXIT ALL POSITIONS

            for sigGen in signalGenerators:
                # these functions can get the signal for *THIS* row, based on the
                # what signal Generators previous to this have done
                # they cannot get or act on signals generated in previous rows
                # signal s from previous signal generations is passed in as an 
                # argument

                result = sigGen(type,s, isLastRow, row, df)
                (s,strategy,opt1,opt1_strike,opt1_signal,opt1_price,opt2,opt2_strike,opt2_signal,opt2_price,opt3,opt3_strike,opt3_signal,opt3_price,opt4,opt4_strike,opt4_signal,opt4_price)  = result if \
                    isinstance(result, tuple) else (result,strategy,opt1,opt1_strike,opt1_signal,opt1_price,opt2,opt2_strike,opt2_signal,opt2_price,opt3,opt3_strike,opt3_signal,opt3_price,opt4,opt4_strike,opt4_signal,opt4_price)
            # logging.info(f"trade price is {trade_price}")
            setTickerPosition(row.symbol, s, row['Adj Close'],row.High, row.Low,strategy,opt1,opt1_signal,opt1_price,opt2,opt2_signal,opt2_price,opt3,opt3_signal,opt3_price,opt4,opt4_signal,opt4_price)
    else:
        #reset at start of day
        return (s,strategy,opt1,opt1_strike,opt1_signal,opt1_price,opt2,opt2_strike,opt2_signal,opt2_price,opt3,opt3_strike,opt3_signal,opt3_price,opt4,opt4_strike,opt4_signal,opt4_price) # Outside of trading hours EXIT ALL POSITIONS
    # if isLastRow:
    #     logTickerStatus(row.symbol)
    return (s,strategy,opt1,opt1_strike,opt1_signal,opt1_price,opt2,opt2_strike,opt2_signal,opt2_price,opt3,opt3_strike,opt3_signal,opt3_price,opt4,opt4_strike,opt4_signal,opt4_price) # Outside of trading hours EXIT ALL POSITIONS



def applyOptionStrategy(df,analyticsGenerators, signalGenerators, tradeStartTime=None):      
    initTickerStatus(df['symbol'][0])

    # Resample the datasframe to daily frequency before passing it to getAnalytics
    df_daily = df.resample('D').agg({'i': 'last','Open': 'first', 'High': 'max', 'Low': 'min', 'Adj Close': 'last', 'Volume': 'sum', 'symbol': 'last'})
    df_daily.index = df_daily.index.date
    # Drop all rows in df_daily where 'Adj Close' is nan
    df_daily = df_daily.dropna(subset=['Adj Close'])
    
    df_daily = signals.getAanalytics(df_daily,analyticsGenerators)
    
    #Analytics is based on Adj Close price of this day, we need to shift it up once
    # so that today we are using yesterdays analytics, and taking position today based on that 
    df_daily = df_daily.shift(1)
    
    maxpain = db.getNiftyLTOptionPainAndPCR(df_daily.index[0],df_daily.index[-1])
    print("duplicated")
    print(maxpain.index.duplicated().sum())  # Should be 0

    columns_to_copy = ['maxPain', 'wmaxPain', 'pcr', 'atm_pcr']
    df_daily[['maxpain', 'wmaxpain', 'pcr', 'atm_pcr']] = maxpain[columns_to_copy]
    # Convert Daily analytics back to hourly for DF
    # Create a new 'date' column in df by extracting the date part from the index
    df['tdt'] = df.index.date
    # Identify intersecting columns
    intersecting_columns = [col for col in df_daily.columns if col in df.columns]

    # Drop intersecting columns from df_daily, except for 'date', which is needed for merging
    df_daily = df_daily.drop(columns=[col for col in intersecting_columns if col != 'date'])

    # Reset the index for df_daily and name the new column 'date'
    df_daily.reset_index(inplace=True)
    df_daily.rename(columns={'index': 'tdt'}, inplace=True)

    # Merge the data frames on the 'date' column
    result = pd.merge(df, df_daily, on='tdt', suffixes=('', '_daily'))
    result.drop(columns=['tdt'], inplace=True)
    result.set_index(df.index, inplace=True)
    df = result
    ## END OF RESAMPLE BACK TO HOURLY 
    
    x = df.apply(getSignal, 
        args=(signalGenerators, df), axis=1)
    (df['signal'],df['strategy'], df['opt1'],df['opt1_strike'],df['opt1_signal'], df['opt1_price'], df['opt2'], \
        df['opt2_strike'], df['opt2_signal'], df['opt2_price'], df['opt3'],df['opt3_strike'],df['opt3_signal'], df['opt3_price'],\
            df['opt4'],df['opt4_strike'],df['opt4_signal'],df['opt4_price']) = zip(*x)

    return df
