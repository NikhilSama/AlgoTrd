import datetime
from DatabaseLogin import DBBasic
import sys
# from selenium import webdriver
# from selenium.webdriver.chrome.options import Options
# from selenium.webdriver.common.by import By
# from selenium.webdriver.support.ui import WebDriverWait
# from selenium.webdriver.support import expected_conditions as EC

import cfg
globals().update(vars(cfg))

db = DBBasic() 
# chrome_driver_path = './Data/chromedriver' # Replace with the path to your ChromeDriver executable


def getPCR():
    return 1
    chrome_options = Options()
    chrome_options.add_argument('--headless')
    browser = webdriver.Chrome(executable_path=chrome_driver_path, options=chrome_options)
    browser.get('https://www.nseindia.com/option-chain')
    
    wait_time = 30
    element = WebDriverWait(browser, wait_time).until(
        EC.presence_of_element_located((By.ID, "equityOptionChainTotalRow-CE-totOI"))
    )

    callOI = element.text

    element = browser.find_element_by_id("equityOptionChainTotalRow-CE-totVol")
    callVol = element.text
    
    element = browser.find_element_by_id("equityOptionChainTotalRow-PE-totOI")
    putOI = element.text
    
    element = browser.find_element_by_id("equityOptionChainTotalRow-PE-totVol")
    putVol = element.text

    callOI = int(callOI.replace(',',''))
    callVol = int(callVol.replace(',',''))
    putOI = int(putOI.replace(',',''))
    putVol = int(putVol.replace(',',''))

    print(f"{callOI},{callVol},{putOI},{putVol}")

def getSpaceMVolDelta(t):
    # Measures vol delta of intra-second up or down candles   
    # HFT Marker
    # I think its When derivative price has deviated too far from underlying; typically marks end of trend .. at least short term
    # but many times long term too

    eTime = datetime.datetime.now()
    sTime = eTime - datetime.timedelta(seconds=120)
    q = f'select sum(up_h),sum(dn_h) from voldelta where ticker = "{t}" and t >= "' + \
        sTime.strftime("%Y-%m-%d %H:%M:%S") + \
            '" and t <= "' + \
            eTime.strftime("%Y-%m-%d %H:%M:%S") + '";'
    data = db.fromDBResult(q)
    for row in data:
        upVol = row[0] if row[0] is not None else 0
        dnVol = row[1] if row[1] is not None else 0

    return (upVol,dnVol)

pcr = getPCR()
(niftyUpVol,niftyDnVol) = getSpaceMVolDelta('NIFTY')
(futUpVol,futDnVol) = getSpaceMVolDelta('NIFTY1!')

print(f"{pcr},{niftyUpVol},{niftyDnVol},{futUpVol},{futDnVol}")