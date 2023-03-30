import os
import pandas as pd
import MySQLdb
from datetime import datetime, timedelta
import time

# Connect to the MySQL database
db = MySQLdb.connect(host="NikhilSama.mysql.pythonanywhere-services.com",
                     user="NikhilSama",
                     passwd="Xyz",
                     db="NikhilSama$backtest")

# Define the table name and column names
table_name = "backtest"
column_names = ["trading_days", "days_in_trade", "num_trades", "num_winning_trades", "num_losing_trades",
                "win_pct", "ret", "ret_per_day_in_trade", "annualized_ret", "avg_per_trade_return",
                "avg_of_per_ticker_std_dev_across_trades", "skewness_pertrade_return", "kurtosis_pertrade_return",
                "wins", "loss", "std_dev_across_stocks", "kurtosis_across_stocks", "skewness_across_stocks",
                "maLen", "bandWidth", "source_filename", "created_time"]

# Define the directory to search for new files
directory = "/home/NikhilSama/algo/AlgoTrd/Data/backtest/nifty"

while True:
    # Get a list of all CSV files in the directory
    csv_files = [f for f in os.listdir(directory) if f.endswith(".csv")]

    # Loop through each CSV file
    for csv_file in csv_files:
        print("Migrating to SQL for file: " + csv_file)
              
        # Check if the file has already been added to the database
        cursor = db.cursor()
        cursor.execute(f"SELECT COUNT(*) FROM {table_name} WHERE source_filename = '{csv_file}'")
        result = cursor.fetchone()[0]
        if result > 0:
            # The file has already been added, so skip it
            print("Skipping, already in DB")
            continue

        # The file has not been added, so read in the data
        file_path = os.path.join(directory, csv_file)
        df = pd.read_csv(file_path)

        # Add the source filename and created time columns to the DataFrame
        df["source_filename"] = csv_file
        df["created_time"] = datetime.now()

        # Convert the DataFrame to a list of tuples
        data = [tuple(row) for row in df.to_numpy()]

        # Insert the data into the MySQL database
        cursor.executemany(f"INSERT INTO {table_name} ({', '.join(column_names)}) VALUES ({', '.join(['%s'] * len(column_names))})", data)
        db.commit()

    # Wait for five minutes before checking for new files again
    time.sleep(300)

# Close the MySQL connection
db.close()
