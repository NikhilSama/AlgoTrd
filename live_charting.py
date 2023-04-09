from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import os
import subprocess
from selenium.webdriver.chrome.options import Options
import socket

# Display charts of current positions on Trading View
chrome_driver_path = '/Users/nikhilsama/Dropbox/Coding/AlgoTrading/Data/chromedriver' # Replace with the path to your ChromeDriver executable

tv_first_url = "https://www.tradingview.com/accounts/signin/"
tv_chart_url = "https://in.tradingview.com/chart/t6kqQsqt/?symbol=NSE%3AHDFCBANK"


xpaths = {
    "login_icon" : "/html/body/div[3]/div/div[2]/div/div/div/div/div/div/div[1]/div[4]/div/span",
    "username" : "/html/body/div[3]/div/div[2]/div/div/div/div/div/div/form/div[1]/div[1]/input",
    "pwd" : "/html/body/div[3]/div/div[2]/div/div/div/div/div/div/form/div[2]/div[1]/input",
    "signinbutton": "/html/body/div[3]/div/div[2]/div/div/div/div/div/div/form/div[5]/div[2]/button",
    "chart" : ["/html/body/div[2]/div[1]/div[2]",
              "/html/body/div[2]/div[1]/div[4]",
              "/html/body/div[2]/div[1]/div[3]",
              "/html/body/div[2]/div[1]/div[5]"
             ],
    "ticker_top_left" : "/html/body/div[2]/div[3]/div/div/div[3]/div[1]/div/div/div/div/div[2]/button[1]/div",
    "ticker_input_dialog" : "/html/body/div[6]/div/div/div[2]/div/div[2]/div[1]/input",
    "chart_layout_selector" : "/html/body/div[2]/div[3]/div/div/div[3]/div[1]/div/div/div/div/div[14]/button/span",
    "four_chart_layout_selector": "/html/body/div[6]/div/span/div[1]/div/div/div/div[1]/div/div[4]/div[1]/div[2]/div[1]/div/span"   
}
displayCharts = ['','','','']
lastChartIndex = 3
chartNum = 0

def initTradingView():
    global browser
    
    # Create Chrome options and hide the address bar
    chrome_options = Options()
    chrome_options.add_argument(f"--app={tv_first_url}")
    chrome_options.add_argument("--disable-extensions")
    chrome_options.add_argument("--disable-infobars")
    chrome_options.add_argument('--disable-blink-features=AutomationControlled')

    browser = webdriver.Chrome(executable_path=chrome_driver_path, \
                            options=chrome_options)

    file = open("/Users/nikhilsama/Dropbox/Coding/AlgoTrading/Data/tv-cred.txt", "r")
    keys = file.read().split()  # Get a List of keys

    user = keys[0]
    pwd = keys[1]
    browser.get(tv_first_url)
    browser.implicitly_wait(5)
    login_icon = browser.find_element_by_xpath(xpaths['login_icon'])
    login_icon.click()
    browser.implicitly_wait(2)
    print("Entering username and password")
    username = browser.find_element("xpath", xpaths['username'])
    password = browser.find_element("xpath", xpaths['pwd'])
    print("Sending username and password")
    username.send_keys(user)
    password.send_keys(pwd)
    print("Sent username and password")
    signinbutton = browser.find_element_by_xpath(xpaths['signinbutton'])
    signinbutton.click()
    browser.implicitly_wait(30)
    input("Proceed ? ")

def init4ChartLayout():
    browser.get(tv_chart_url)
    browser.implicitly_wait(30)
    input("Proceed ? ")

    try:
        chart_layout_selector = browser.find_element("xpath", xpaths['chart_layout_selector'])
        chart_layout_selector.click()
        browser.implicitly_wait(10)
        four_chart_layout_selector = browser.find_element("xpath", xpaths['four_chart_layout_selector'])
        four_chart_layout_selector.click()
    except:
        print("didnt find four chart layout selector")
    browser.implicitly_wait(10)
def alreadyDisplayed(ticker):
    for i in range(0,lastChartIndex):
        if displayCharts[i] == ticker:
            return True
    return False

def loadChart(ticker):
    global chartNum
    if alreadyDisplayed(ticker):
        return
    try:
        toClick = browser.find_element("xpath", xpaths['chart'][chartNum])
        toClick.click()
        ticker_top_left = browser.find_element("xpath", xpaths['ticker_top_left'])
        ticker_top_left.click()
        browser.implicitly_wait(2)
        ticker_input_dialog = browser.find_element_by_xpath("//div[@data-name='symbol-search-items-dialog']//input[@type='text']")
        ticker_input_dialog.clear()
        ticker_input_dialog.send_keys(ticker)
        ticker_input_dialog.send_keys(Keys.RETURN)
        displayCharts[chartNum] = ticker
    except: 
        print("didnt find four chart layout selector")

    if chartNum == lastChartIndex:
        chartNum = 0
    else:
        chartNum = chartNum + 1
    
def init():
    initTradingView()
    init4ChartLayout()

def listen():

    # create a socket object
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # bind the socket to a public host and port
    server_socket.bind(('localhost', 8000))

    # listen for incoming connections
    server_socket.listen(1)

    print('Listening for incoming messages...')

    while True:
        # wait for a connection
        client_socket, client_address = server_socket.accept()

        # receive the message
        message = client_socket.recv(1024)

        # process the message
        print('Received message: ', message.decode())

        # close the client connection
        client_socket.close()

def show(ticker):
    # Define the IP address and port to connect to
    ip_address = '192.168.0.134'
    port = 8000
    print(f"Sending message to {ip_address}:{port}...{ticker}")
    # Create a socket object
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # Connect to the socket
    client_socket.connect((ip_address, port))

    # Send a message to the server
    message = ticker
    client_socket.send(message.encode())
    print(f"Sent message to {ip_address}:{port}...{ticker}")

    # Close the socket
    client_socket.close()


if __name__ == '__main__':
    listen()
