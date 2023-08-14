

import signals
import numpy as np
import cfg
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
    return np.isnan(tickerStatus[t]['position'])
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
    
def setTickerPosition(ticker,signal, entry_price,high, low):
    # logging.info(f"setting ticker position for {ticker} to {signal} entry{entry_price} pos:{tickerStatus[ticker]['position']}")
    setTickerMaxPriceForTrade(ticker,entry_price,high,low,signal)
    (tickerStatus[ticker]['limit1'], tickerStatus[ticker]['limit2'], tickerStatus[ticker]['sl1'], tickerStatus[ticker]['sl2']) = (limit1, limit2, sl1, sl2)
    #logging.info(f"signal: {signal} entry: {tickerStatus[ticker]['entry_price']} max: {tickerStatus[ticker]['max_price']} min: {tickerStatus[ticker]['min_price']}")
    if np.isnan(signal):
        return # nan signal does not change position
    if ticker not in tickerStatus:
        tickerStatus[ticker] = {}
    if tickerStatus[ticker]['position'] != signal:
        if tickerStatus[ticker]['position'] != 0:
            #Store last long or short position
            tickerStatus[ticker]['prevPosition'] = tickerStatus[ticker]['position']
        tickerStatus[ticker]['position'] = signal
        if signal == 0:
            tickerStatus[ticker]['entry_price'] = float('nan')
        else:
            tickerStatus[ticker]['entry_price'] = round(entry_price,1)
## END OF TICKER STATUS
def getShortStraddle(row):
    close = round(row['Adj Close']/100)*100
    putStrike = close+100
    callStrike = close-100
    putTicker = getPutTicker(row,putStrike)
    callTicker = getCallTicker(row,callStrike)
    return (putTicker,-1,callTicker,-1)
    

def shortStraddle(type,s,isLastRow,row,df):
    strategy = "Short Straddle"
    opt1 = opt1_signal = opt2 = opt2_signal = \
        opt3, opt3_signal, opt4, opt4_signal = float("nan")
    if tickerHasNoPosition(df['symbol'][0]):
        (opt1,opt1_signal,opt2,opt2_signal) = getShortStraddle(row)
    return (s,strategy,opt1,opt1_signal,opt2,opt2_signal,opt3,opt3_signal,opt4,opt4_signal) # Outside of trading hours EXIT ALL POSITIONS

def weCanEnterNewTrades(row):
    return True
def getSignal(row,signalGenerators, df):
    s = strategy = opt1 = opt1_signal = opt2 = opt2_signal = \
        opt3, opt3_signal, opt4, opt4_signal = float("nan")
    isLastRow = (row.name == df.index[-1]) or cfgIsBackTest
    row_time = row.name.time()
   
    #Return nan if its not within trading hours
    if signals.weShouldTrade(row):
            if weCanEnterNewTrades(row):
                type = 1 # Entry or Exit
            elif row.name.time() < cfgEndExitTradesOnlyTimeOfDay:
                # Last time period before intraday exit; only exit positions
                # No new psitions will be entered
                type = 0 
            else:
                return (0,strategy,opt1,opt1_signal,opt2,opt2_signal,opt3,opt3_signal,opt4,opt4_signal) # Outside of trading hours EXIT ALL POSITIONS

            for sigGen in signalGenerators:
                # these functions can get the signal for *THIS* row, based on the
                # what signal Generators previous to this have done
                # they cannot get or act on signals generated in previous rows
                # signal s from previous signal generations is passed in as an 
                # argument

                result = sigGen(type,s, isLastRow, row, df)
                (s,strategy,opt1,opt1_signal,opt2,opt2_signal,opt3,opt3_signal,opt4,opt4_signal)  = result if \
                    isinstance(result, tuple) else (result,strategy,opt1,opt1_signal,opt2,opt2_signal,opt3,opt3_signal,opt4,opt4_signal)
   
            # logging.info(f"trade price is {trade_price}")
            setTickerPosition(row.symbol, s, row['Adj Close'],row.High, row.Low)
    else:
        #reset at start of day
        initTickerStatus(row.symbol)
        return (s,strategy,opt1,opt1_signal,opt2,opt2_signal,opt3,opt3_signal,opt4,opt4_signal) # Outside of trading hours EXIT ALL POSITIONS
    # if isLastRow:
    #     logTickerStatus(row.symbol)
    return (s,strategy,opt1,opt1_signal,opt2,opt2_signal,opt3,opt3_signal,opt4,opt4_signal) # Outside of trading hours EXIT ALL POSITIONS



def applyOptionStrategy(df,analyticsGenerators, signalGenerators):

    df = signals.getAanalytics(df,analyticsGenerators)    
    
    x = df.apply(getSignal, 
        args=(signalGenerators, df), axis=1)
    (df['signal'],df['strategy'], df['opt1'],df['opt1_signal'], df['opt2'], \
        df['opt2_signal'], df['opt3'],df['opt3_signal'],\
            df['opt4'],df['opt4_signal']) = zip(*x)

    return df
