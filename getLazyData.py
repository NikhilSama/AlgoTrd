import datetime
from DatabaseLogin import DBBasic
import requests
import time
import pandas as pd
import json

# from selenium import webdriver
# from selenium.webdriver.chrome.options import Options
# from selenium.webdriver.common.by import By
# from selenium.webdriver.support.ui import WebDriverWait
# from selenium.webdriver.support import expected_conditions as EC

import cfg
globals().update(vars(cfg))
url_indices="https://www.nseindia.com/api/allIndices"
url_oc= "https://www.nseindia.com/option-chain"
url_nf='https://www.nseindia.com/api/option-chain-indices?symbol=NIFTY'
headers={
        'user-agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36 Edg/115.0.1901.203',
        'Accept-Encoding':'gzip, deflate, br',
        'Accept-Language':'en-US,en;q=0.9'
        }

db = DBBasic() 

sess = requests.Session()
cookies = dict()

# chrome_driver_path = './Data/chromedriver' # Replace with the path to your ChromeDriver executable
def set_cookie():
    request = sess.get(url_oc, headers=headers, timeout=5)
    cookies = dict(request.cookies)
def get_data(url):
    set_cookie()
    response = sess.get(url, headers=headers, timeout=5, cookies=cookies)
    if(response.status_code==401):
        set_cookie()
        response = sess.get(url_nf, headers=headers, timeout=5, cookies=cookies)
    if(response.status_code==200):
        return response.text
    return ""

def getNiftyLTP():
    response_text = get_data(url_indices)
    data = json.loads(response_text)
    for index in data["data"]:
        if index["index"]=="NIFTY 50":
            nf_ltp = index["last"]
    return nf_ltp

def getPCR():
    ltp = getNiftyLTP()    
    url='https://www.nseindia.com/api/option-chain-indices?symbol=NIFTY'
    session=requests.Session()
    request=session.get(url,headers=headers)
    time.sleep(1)
    cookies=dict(request.cookies)
    time.sleep(1)
    response=session.get(url,headers=headers,cookies=cookies).json()
    rawdata=pd.DataFrame(response)
    rawop=pd.DataFrame(rawdata['filtered']['data']).fillna(0)    
    data=[]
    for i in range(0,len(rawop)):
        calloi=callcoi=cltp=cIV=putoi=putcoi=pltp=pIV=0
        stp=rawop['strikePrice'][i]
        expiry=datetime.datetime.strptime(rawop['expiryDate'][i], '%d-%b-%Y')
        if abs(stp - round(ltp)) > 200:
            # print(f"skipping {stp}")
            #Skip Deep OTM strikes
            continue
        today = datetime.datetime.now()
        if (expiry - today).days > 7:
            # skip long dated expiries
            continue
        
        # print(expiry)
        # print(type(expiry))       
        if(rawop['CE'][i]==0):
            calloi=callcoi=0
        else:
            calloi=rawop['CE'][i]['openInterest']
            callcoi=rawop['CE'][i]['changeinOpenInterest']
            cltp=rawop['CE'][i]['lastPrice']
            cIV=rawop['CE'][i]['impliedVolatility']
        if(rawop['PE'][i]==0):
            putoi=putcoi=0
        else:
            putoi=rawop['PE'][i]['openInterest']
            putcoi=rawop['PE'][i]['changeinOpenInterest']
            pltp=rawop['PE'][i]['lastPrice']
            pIV=rawop['PE'][i]['impliedVolatility']
        opdata={
            'CALL OI':calloi,'CALL CHNG OI':callcoi,'CALL LTP':cltp,'STRIKE PRICE':stp,'CALL IV':cIV,
            'PUT OI':putoi,'PUT  CHNG OI':putcoi,'PUT  LTP':pltp,'STRIKE PRICE':stp,'PUT  IV':pIV
            }
        # print(f"{calloi} : {cStrike} : {putoi} : {pStrike}")
        data.append(opdata)
    optionchain=pd.DataFrame(data)
    totalcalloi=optionchain['CALL OI'].sum()
    totalputoi=optionchain['PUT OI'].sum()
    callVol = putVol= 0
    pcr = totalputoi/totalcalloi
    # print(f"{totalcalloi},{callVol},{totalputoi},{putVol}")

    optionchain['callpain'] = optionchain.apply(lambda row: row['CALL OI'] * max(row['STRIKE PRICE'] - ltp, 0), axis=1)
    optionchain['putpain'] = optionchain.apply(lambda row: row['PUT OI'] * max(ltp - row['STRIKE PRICE'], 0), axis=1)
    optionchain['totalpain'] = optionchain['callpain'] + optionchain['putpain']
    # print(optionchain)
    top_3_strikes = optionchain.nsmallest(3, 'totalpain')
    maxpain = top_3_strikes['STRIKE PRICE'].iloc[0]
    
    inverse_weights = 1 / top_3_strikes['totalpain']
    normalized_weights = inverse_weights / inverse_weights.sum()
    weightedAvMaxPain = (top_3_strikes['STRIKE PRICE'] * normalized_weights).sum()
    weightedAvMaxPain = round(weightedAvMaxPain,0)
    # print(f"Max Pain: {maxpain}")
    # print(f"Weighted Avg Strike Price: {weightedAvMaxPain}")
    # print(optionchain)
    # print(top_3_strikes)
    return (round(pcr,2),maxpain,weightedAvMaxPain)

def getSpaceMVolDelta(t):
    # Measures vol delta of intra-second up or down candles   
    # HFT Marker
    # I think its When derivative price has deviated too far from underlying; typically marks end of trend .. at least short term
    # but many times long term too

    eTime = datetime.datetime.now()
    sTime = eTime - datetime.timedelta(seconds=180)
    q = f'select sum(up_h),sum(dn_h) from voldelta where ticker = "{t}" and t >= "' + \
        sTime.strftime("%Y-%m-%d %H:%M:%S") + \
            '" and t <= "' + \
            eTime.strftime("%Y-%m-%d %H:%M:%S") + '";'
    data = db.fromDBResult(q)
    for row in data:
        upVol = row[0] if row[0] is not None else 0
        dnVol = row[1] if row[1] is not None else 0

    return (upVol,dnVol)

(pcr,maxpain,wmaxpain) = getPCR()
(niftyUpVol,niftyDnVol) = getSpaceMVolDelta('NIFTY')
(futUpVol,futDnVol) = getSpaceMVolDelta('NIFTY1!')

print(f"{pcr},{niftyUpVol},{niftyDnVol},{futUpVol},{futDnVol},{maxpain},{wmaxpain}")