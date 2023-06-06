#!/usr/bin/env python3

import pandas as pd
import warnings
warnings.filterwarnings("ignore")
# Code in separate file DatabaseLogin.py to login to kite connect 
from DatabaseLogin import DBBasic
import kite_init as ki 
import DownloadHistorical as downloader
import datetime
import math 

db = DBBasic()

def getNiftyPrice(zgetTO):
    zgetFROM = zgetTO - datetime.timedelta(minutes=1)
    df = downloader.zget(zgetFROM,zgetTO,'NIFTY 50','minute',includeOptions=False)
    return df.iloc[0]['Adj Close'] # 10 am price

def getStrike():
    niftyOpen = getNiftyPrice(datetime.datetime.combine(datetime.date.today(), datetime.time(10)))
    strikeFloor = math.floor(niftyOpen/100)*100
    strikeCiel = math.ceil(niftyOpen/100)*100
    #print(f"Strike Floor: {strikeFloor}, Strike Ciel: {strikeCiel}")
    return (strikeFloor,strikeCiel)
    
def getTickers():
    (strikeFloor,strikeCiel) = getStrike()
    (itmCall,lot,tick,strike) = db.get_option_ticker('NIFTY 50',0,'CE',None,strike=strikeFloor)
    (otmCall,lot,tick,strike) = db.get_option_ticker('NIFTY 50',0,'CE',None,strike=strikeCiel)
    (otmPut,lot,tick,strike) = db.get_option_ticker('NIFTY 50',0,'PE',None,strike=strikeFloor)
    (itmPut,lot,tick,strike) = db.get_option_ticker('NIFTY 50',0,'PE',None,strike=strikeCiel)
#    print(f"ITM Call: {itmCall}, OTM Call: {otmCall}, OTM Put: {otmPut}, ITM Put: {itmPut}")
    return(itmCall,otmCall,otmPut,itmPut)

def getDataForOptions():
    zgetFROM = datetime.datetime.combine(datetime.date.today(), datetime.time(9))
    zgetTO = datetime.datetime.combine(datetime.date.today(), datetime.time(16))

    (itmCall,otmCall,otmPut,itmPut) = getTickers()

    itmCallDF = downloader.zget(zgetFROM,zgetTO,itmCall,'minute',includeOptions=False)
    itmCallDF = itmCallDF.drop(['i'],axis=1)
    itmPutDF = downloader.zget(zgetFROM,zgetTO,itmPut,'minute',includeOptions=False)
    otmCallDF = downloader.zget(zgetFROM,zgetTO,otmCall,'minute',includeOptions=False)
    otmPutDF = downloader.zget(zgetFROM,zgetTO,otmPut,'minute',includeOptions=False)
        
    return(itmCallDF,otmCallDF,otmPutDF,itmPutDF)

def saveTodayOptionsDataToDb():
    (itmCallDF,otmCallDF,otmPutDF,itmPutDF) = getDataForOptions()
    db.toDB('niftyITMCall',itmCallDF)
    db.toDB('niftyOTMCall',otmCallDF)
    db.toDB('niftyITMPut',itmPutDF)
    db.toDB('niftyOTMPut',otmPutDF)
       
saveTodayOptionsDataToDb()