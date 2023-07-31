import sys
import os
import re
import pandas as pd
import numpy as np
from datetime import datetime
import time
import tickerCfg
import calendar


#cfg has all the config parameters make them all globals here
import cfg
globals().update(vars(cfg))

    
def timeToString(t,date=False,time=True):
    if date and time:
        return t.strftime("%d-%m %I:%M %p")
    elif date:
        return t.strftime("%d-%m")
    else:
        return t.strftime("%I:%M:%S %p")

def fileNameFromArgs(prefix=''):
    args = sys.argv[1:]
    fname = prefix
    for arg in args:
        key, value = arg.split(':')
        if key in ['zerodha_access_token','dbuser','dbpass','cacheTickData', 'dbname', 'dbhost']:
            continue
        fname = fname + '-' + value
    fname = fname + '.csv'
    return fname
def fileExists(fname):
    return os.path.isfile(fname)
def tickerIsFutOrOption(t):
    return tickerIsFuture(t) or isOption(t)
def tickerIsFuture(t):
    return re.match(r'^[a-zA-Z]+\d+[a-zA-Z]{3}FUT$', t)
def getUnderlyingTickerForFuture(t):
    match = re.match(r'^([a-zA-Z]+)\d+[a-zA-Z]{3}FUT$', t)
    if match:
        uderlying_ticker = match.group(1)
        return uderlying_ticker
    else:
        return None
def explodeOptionTicker(t):
    if t.endswith(".NFO"):
        t = t[:-4]
    pattern = r'^([A-Z]+)\d+(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)?\d+((?:PE|CE))$'
    match = re.match(pattern, t)
    if match:
        return (match.group(1), match.group(2))
    else:
        return False
def optionTypeFromTicker(t):
    if explodeOptionTicker(t):
        return explodeOptionTicker(t)[1]
def optionUnderlyingFromTicker(t):
    if explodeOptionTicker(t):
        return explodeOptionTicker(t)[0]
    else:
        return t
def isPutOption(t):
    return optionTypeFromTicker(t) == 'PE'
def isCallOption(t):
    return optionTypeFromTicker(t) == 'CE'
def isOption(t):
    return isPutOption(t) or isCallOption(t)
def isNotAnOption(t):
    return not isOption(t)
def convertPEtoCEAndViceVersa(t):
    if optionTypeFromTicker(t):     
        return t[:-2] + ('CE' if t[-2] == 'P' else 'PE')
    else:
        return t
def getTickerCfg(ticker):
    if explodeOptionTicker(ticker):
        ticker = explodeOptionTicker(ticker)[0]
    else:
        futTicker = getUnderlyingTickerForFuture(ticker)
        if futTicker is not None:
            ticker = futTicker
    if ticker in tickerCfg.tickerCfg:
        #print(f"Applying CFG for {ticker}")
        return tickerCfg.tickerCfg[ticker]
    else:
        print(f"No CFG for {ticker}. Reverting to default CFG")
        return cfgDict
    
    
def getNSEHolidays():
    nse_holidays = [
            "January 26, 2018", "February 13, 2018", "March 02, 2018", "March 29, 2018",
        "March 30, 2018", "April 30, 2018", "May 01, 2018", "August 15, 2018",
        "August 22, 2018", "September 13, 2018", "September 20, 2018", "October 02, 2018",
        "October 18, 2018", "November 07, 2018", "November 08, 2018", "November 23, 2018",
        "December 25, 2018", "March 04, 2019", "March 21, 2019", "April 17, 2019",
        "April 19, 2019", "May 01, 2019", "May 06, 2019", "June 05, 2019",
        "August 12, 2019", "August 15, 2019", "August 23, 2019", "October 02, 2019",
        "October 08, 2019", "October 21, 2019", "November 12, 2019", "December 25, 2019",
        
        "January 26, 2019",  # Republic Day
        "February 4, 2019",  # Maha Shivratri
        "March 29, 2019",  # Holi
        "April 14, 2019",  # Dr. Babasaheb Ambedkar Jayanti
        "April 29, 2019", # Good Friday
        "May 1, 2019",  # May Day
        "May 12, 2019",  # Mahavir Jayanti
        "August 15, 2019",  # Independence Day
        "October 2, 2019",  # Gandhi Jayanti
        "December 25, 2019",  # Christmas
        "January 26, 2020",  # Republic Day
        "March 25, 2020",  # Holi
        "April 10, 2020",  # Good Friday
        "April 14, 2020",  # Dr. Babasaheb Ambedkar Jayanti
        "May 1, 2020",  # May Day
        "May 12, 2020",  # Mahavir Jayanti
        "August 15, 2020",  # Independence Day
        "October 2, 2020",  # Gandhi Jayanti
        "December 25, 2020",  # Christmas
        "September 2, 2019",#ganesh chaturthi
        "September 10, 2019",#muharram
        "October 28, 2019",#diwali pre-diwali
        "January 26, 2020", "February 21, 2020", "March 10, 2020", "March 25, 2020",
        "April 02, 2020", "April 06, 2020", "April 10, 2020", "May 01, 2020",
        "May 25, 2020", "October 02, 2020", "November 16, 2020", "November 30, 2020",
        "December 25, 2020", "26 January, 2021", "March 11, 2021", "March 29, 2021", "April 02, 2021",
        "April 14, 2021", "April 21, 2021", "April 28, 2021", "May 13, 2021", "July 21, 2021",
        "August 19, 2021", "September 10, 2021", "October 15, 2021", "November 04, 2021",
        "November 05, 2021", "November 19, 2021", "December 06, 2021", "December 25, 2021",
        "January 26, 2022", "March 01, 2022", "March 18, 2022", "April 14, 2022", "April 15, 2022",
        "May 03, 2022", "August 09, 2022", "August 15, 2022", "August 31, 2022", "October 05, 2022", 
        "October 24, 2022", "October 26, 2022", "November 08, 2022",
        "26-Jan-2023","07-Mar-2023","30-Mar-2023","04-Apr-2023","07-Apr-2023","14-Apr-2023",
        "01-May-2023","28-Jun-2023", "15-Aug-2023", "19-Sep-2023", "02-Oct-2023", "24-Oct-2023",
        "14-Nov-2023", "27-Nov-2023", "25-Dec-2023"
    ]
    nse_holidays = pd.to_datetime(nse_holidays)
    nse_holidays = [h.date() for h in nse_holidays]
    return nse_holidays

def cleanDF(df):
    # Kite can sometimes return junk data before 915 or 1530, wich very 
    # low or zero volume.  These set the min/max values for OBV and 
    # affect our analytics and signals for a long time.  So we filter
    # fileter out these junk values

    df = df.between_time('09:17:00+05:30', '15:29:00+05:30')    
    df = df[df.index.weekday<5] # remove weekends
    
    # Remove holiday data from the DataFrame
    df = df[~df.index.isin(getNSEHolidays())]
    return df

def isTradingDay(date):
    is_weekend = date.weekday() >= 5
    is_holiday = date.date() in getNSEHolidays()
    return not (is_weekend or is_holiday)

def priceWithSlippage(slPrice, type='longEntry'):
    if type == 'longEntry' or type == 'shortExit':
        return slPrice*(1+cfgSlippage)
    elif type == 'shortEntry' or type == 'longExit':
        return slPrice*(1-cfgSlippage)
    
def get_last_thursday(year, month):
    """
    This function returns the last Thursday of the given month and year.
    """
    last_day_of_month = calendar.monthrange(year, month)[1]
    last_day_date = datetime.date(year, month, last_day_of_month)
    while last_day_date.weekday() != calendar.THURSDAY:
        last_day_date -= datetime.timedelta(days=1)
    return last_day_date

def getImmidiateFutureTicker(index='NIFTY'):
    """
    This function generates the next month NIFTY Future ticker.
    """
    # Get current date
    today = datetime.date.today()
    
    # Get last Thursday of the current month
    last_thursday = get_last_thursday(today.year, today.month)
    
    # Check if the current date is before the last Thursday of the month
    if today < last_thursday:
        year = today.year
        month = today.month
    else:
        # If it's on or after the last Thursday, use the next month
        if today.month == 12:
            year = today.year + 1
            month = 1
        else:
            year = today.year
            month = today.month + 1
    
    # Format year, month and index name into the desired string format
    month_str = calendar.month_name[month][:3].upper()
    year_str = str(year)[2:]
    ticker = f"{index}{year_str}{month_str}FUT"
    
    return ticker
