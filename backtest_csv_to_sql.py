import os
import csv
import mysql.connector
from datetime import datetime, timedelta

# MySQL database connection parameters
host = 'NikhilSama.mysql.pythonanywhere-services.com'
user = 'NikhilSama'
password = 'trading123'
database = 'NikhilSama$default'

# Directory to watch for new CSV files
csv_dir = '/home/NikhilSama/algo/AlgoTrd/Data/backtest/nifty'

# Connect to MySQL database
cnx = mysql.connector.connect(user=user, password=password, host=host, database=database)
cursor = cnx.cursor()

# Loop forever
while True:
    # Get list of CSV files in directory
    csv_files = os.listdir(csv_dir)
    # Loop through each file
    for csv_file in csv_files:
        # Check if file has already been processed
        query = "SELECT COUNT(*) FROM backtest WHERE source_filename = %s"
        cursor.execute(query, (csv_file,))
        result = cursor.fetchone()
        if result[0] > 0:
            continue
        # Parse data from CSV file
        with open(os.path.join(csv_dir, csv_file)) as f:
            reader = csv.DictReader(f)
            row = next(reader)
            # Prepare data for insertion into MySQL table
            data = {
                'trading_days': row['trading_days'],
                'days_in_trade': row['days_in_trade'],
                'num_trades': row['num_trades'],
                'num_winning_trades': row['num_winning_trades'],
                'num_losing_trades':
