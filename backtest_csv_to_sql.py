import os
import csv
import mysql.connector
from datetime import datetime, timedelta
import time

# set up database connection
mydb = mysql.connector.connect(
    host="NikhilSama.mysql.pythonanywhere-services.com",
    user="NikhilSama",
    password="Xyz",
    database="NikhilSama$backtest"
)
cursor = mydb.cursor()

# define function to parse CSV files
def parse_csv(file_path):
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        header = next(reader)  # skip header row
        data = next(reader)  # get data row
        data.pop(0)  # remove first column
        data.append(os.path.basename(file_path))  # add source filename
        data.append(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))  # add created_time
        return tuple(data)

# define function to insert data into database
def insert_data(data):
    sql = "INSERT INTO backtest (trading_days, days_in_trade, num_trades, num_winning_trades, num_losing_trades, win_pct, ret, ret_per_day_in_trade, annualized_ret, avg_per_trade_return, average_of_per_ticker_std_dev_across_trades, skewness_pertrade_return, kurtosis_pertrade_return, wins, loss, std_dev_across_stocks, kurtosis_across_stocks, skewness_across_stocks, maLen, bandWidth, fastMALen, adxLen, adxThresh, adxThreshYellowMultiplier, numCandlesForSlopeProjection, atrLen, superLen, superBandWidth, adxSlopeThresh, maSlopeThresh, maSlopeThreshYellowMultiplier, maSlopeSlopeThresh, obvOscThresh, obvOscThreshYellowMultiplier, obvOscSlopeThresh, overrideMultiplier, source_filename, created_time) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"
    cursor.execute(sql, data)
    mydb.commit()

# set up loop to check for new files every 5 minutes
while True:
    for file_name in os.listdir("/home/NikhilSama/algo/AlgoTrd/Data/backtest/nifty"):
        if file_name.endswith(".csv"):
            file_path = os.path.join("/home/NikhilSama/algo/AlgoTrd/Data/backtest/nifty", file_name)
            # check if file has already been processed
            sql = "SELECT * FROM backtest WHERE source_filename = %s"
            cursor.execute(sql, (file_name,))
            result = cursor.fetchone()
            if result is None:
                # file has not been processed, so parse and insert data
                data = parse_csv(file_path)
                insert_data(data)
    # wait 5 minutes before checking again
    time.sleep(300)
