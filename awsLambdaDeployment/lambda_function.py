import json
import mysql.connector
from datetime import datetime
import pytz

print('Loading function')


def lambda_handler(event, context):
    split_values = event['body'].split("|")
    
    t = split_values[0]
    utc_datetime = datetime.strptime(t, '%Y-%m-%dT%H:%M:%SZ')
    ist_timezone = pytz.timezone('Asia/Kolkata')
    ist_datetime = utc_datetime.replace(tzinfo=pytz.utc).astimezone(ist_timezone)
    t = ist_datetime.strftime('%Y-%m-%d %H:%M:%S')

    dn = {
        'o': int(round(float(split_values[1]))),
        'h': int(round(float(split_values[2]))),
        'l': int(round(float(split_values[3]))),
        'c': int(round(float(split_values[4])))
    }
    up = {
        'o': int(round(float(split_values[5]))) - dn['c'] ,
        'h': int(round(float(split_values[6]))) - dn['c'],
        'l': int(round(float(split_values[7]))) - dn['c'],
        'c': int(round(float(split_values[8]))) - dn['c']
    }
    
    ticker = split_values[9]
    
    print(f"SpaceManVolDelta: {ticker}: {t}: {up['o']}, {up['h']}, {up['l']}, {up['c']}, {dn['o']}, {dn['h']}, {dn['l']}, {dn['c']}")

    # Define the database connection details
    host = 'trading.ca6bwmzs39pr.ap-south-1.rds.amazonaws.com'
    database = 'trading'
    username = 'trading'
    password = 'trading123'
    
    # Establish a connection to the database
    connection = mysql.connector.connect(
        host=host,
        database=database,
        user=username,
        password=password
    )
    
    # Create a cursor object to interact with the database
    cursor = connection.cursor()
    
    # Construct the query to insert the row into the table
    query = "INSERT INTO voldelta (t, up_o, up_h, up_l, up_c, dn_o, dn_h, dn_l, dn_c,ticker) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s,%s)"
    values = (t, up['o'], up['h'], up['l'], up['c'], dn['o'], dn['h'], dn['l'], dn['c'],ticker)
    
    # Execute the query
    cursor.execute(query, values)
    
    # Commit the changes to the database
    connection.commit()
    
    # Close the cursor and connection
    cursor.close()
    connection.close()

    # print("value1 = " + event['key1'])
    # print("value2 = " + event['key2'])
    # print("value3 = " + event['key3'])
    # return event['key1']  # Echo back the first key value
    # #raise Exception('Something went wrong')
    return 'ok'

