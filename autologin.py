#!/usr/bin/env python

from kiteconnect import KiteConnect
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import time
import os
from pyotp import TOTP
import sys 

chrome_driver_path = './Data/chromedriver' # Replace with the path to your ChromeDriver executable

file = open("./Data/cred.txt", "r")
keys = file.read().split()  # Get a List of keys

api_key = keys[0]
key_secret = keys[1]
userID = keys[2]
pwd = keys[3]
totp_key = keys[4]

kite = KiteConnect(api_key=api_key)

# Create Chrome options and set them to run in headless mode
chrome_options = Options()
chrome_options.add_argument('--disable-blink-features=AutomationControlled')

if 'headless' in sys.argv:
    chrome_options.add_argument('--headless')

browser = webdriver.Chrome(executable_path=chrome_driver_path, options=chrome_options)
browser.get(kite.login_url())
browser.implicitly_wait(5)
username = browser.find_element("xpath", '/html/body/div[1]/div/div[2]/div[1]/div/div/div[2]/form/div[1]/input')
password = browser.find_element("xpath", '/html/body/div[1]/div/div[2]/div[1]/div/div/div[2]/form/div[2]/input') 

username.send_keys(userID)
password.send_keys(pwd)

print ("SEnd usernmae and pass")
# Click Login Button
browser.find_element("xpath", '/html/body/div[1]/div/div[2]/div[1]/div/div/div[2]/form/div[4]/button').click()
time.sleep(2)
print ("Asking for TOTP")

pin = browser.find_element("xpath", '/html/body/div[1]/div/div[2]/div[1]/div[2]/div/div[2]/form/div[1]/input')
totp = TOTP(totp_key)
token = totp.now()
print(f"Got totp {token}")
pin.send_keys(token)

#browser.find_element("xpath", '/html/body/div[1]/div/div[2]/div[1]/div/div/div[2]/form/div[3]/button').click()
time.sleep(2)
temp_token=browser.current_url.split('request_token=')[1][:32]
# Save in Database or text File
print('got temp_token', temp_token)
kite = KiteConnect(api_key=api_key)
data = kite.generate_session(temp_token, api_secret=key_secret)
access_token = data["access_token"]
print("got access_token", access_token)

with open('./Data/zerodha_kite_accesstoken.txt', "w") as f:
    f.write(access_token)
browser.quit()