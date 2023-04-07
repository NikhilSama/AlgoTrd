#!/usr/bin/env python

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys

import time
import os
import sys 

chrome_driver_path = '/Users/nikhilsama/Dropbox/Coding/AlgoTrading/Data/chromedriver' # Replace with the path to your ChromeDriver executable

# Create Chrome options and set them to run in headless mode
chrome_options = Options()
chrome_options.add_argument('--disable-blink-features=AutomationControlled')

browser = webdriver.Chrome(executable_path=chrome_driver_path, options=chrome_options)
browser.get("https://www.youtube.com/")
browser.implicitly_wait(15)
search_input = browser.find_element("xpath", '/html/body/ytd-app/div[1]/div/ytd-masthead/div[4]/div[2]/ytd-searchbox/form/div[1]/div[1]/input')
search_input.send_keys("Never gonna give you up")
search_input.send_keys(Keys.RETURN)

video = browser.find_element('xpath', '/html/body/ytd-app/div[1]/ytd-page-manager/ytd-search/div[1]/ytd-two-column-search-results-renderer/div/ytd-section-list-renderer/div[2]/ytd-item-section-renderer/div[3]/ytd-video-renderer[1]/div[1]/ytd-thumbnail/a/yt-image/img')
video.click()

time.sleep(200)


