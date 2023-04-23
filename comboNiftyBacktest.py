from multiprocessing import Pool,cpu_count
import subprocess
import os
import pandas as pd
import sys
import itertools 
import time 
import mysql.connector
import datetime
import socket 

mydb = None
monthlyStartDates = ['2022-05-01','2022-06-01', '2022-07-01', '2022-08-01', '2022-09-01', '2022-10-01', '2022-11-01', '2022-12-01', '2023-01-01', '2023-02-01', '2023-03-01', '2023-04-01', '2023-05-01', '2023-06-01', '2023-07-01', '2023-08-01', '2023-09-01', '2023-10-01', '2023-11-01', '2023-12-01', '2024-01-01', '2024-02-01', '2024-03-01']
def generate_days(start_date, end_date, target_day):
    days = {
        "Monday": 0,
        "Tuesday": 1,
        "Wednesday": 2,
        "Thursday": 3,
        "Friday": 4,
        "Saturday": 5,
        "Sunday": 6
    }
    
    if target_day not in days:
        raise ValueError("Invalid day specified. Please provide a valid day of the week.")
    
    target_day_num = days[target_day]

    start = datetime.datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.datetime.strptime(end_date, "%Y-%m-%d")
    
    # Find the first target_day after the start_date
    start += datetime.timedelta(days=(target_day_num - start.weekday()) % 7)
    
    # Iterate through the target days and store them as strings in the 'result_days' list
    result_days = []
    while start <= end:
        if start.weekday() == target_day_num:
            result_days.append(start.strftime("%Y-%m-%d"))
        start += datetime.timedelta(days=7)

    return result_days



def printSQLToGetRESULTS():
    start_date = "2022-05-01"
    end_date = "2023-03-31"

    startTimes = generate_days(start_date, end_date, "Monday")
    for startDate in startTimes:
        print(f"select id,t,startTime,endTime,num_trades,drawdn,retrn,sharpe,avgRet, stdDev,dayAv,dayShrp,maLen,slpThres,adxLen,adxThresh,candles from niftystratview where startTime = '{startDate}' order by sharpe desc, cast(retrn as decimal) desc limit 10;")

def connect_to_db():
    global mydb
    # Connect to the MySQL database
    mydb = mysql.connector.connect(
        host="trading.ca6bwmzs39pr.ap-south-1.rds.amazonaws.com",
        user="trading",
        password="trading123",
        database="trading"
)
def close_db():
    mydb.close()

def getFnameForLastTask():
    connect_to_db()
    q = "SELECT task_name FROM tasks ORDER BY created_time DESC LIMIT 1"
    mycursor = mydb.cursor(buffered=True)
    mycursor.execute(q)
    (fname,) = mycursor.fetchone()
    mycursor.close()
    close_db()
    return fname
def getCfgFromFname(fname):
    pairs = fname.split()
    cfg = {}
    for pair in pairs:
        key,value = pair.split(':')
        if not any(c.isalpha() for c in value):
            if '.' in value:
                # value has a decimal, so convert to float
                value = float(value)
            else:
                # value has no decimal, so convert to int
                value = int(value)

        cfg[key] = value
    return cfg
    
# Add a task name to the tasks table
def add_task(task_name):
    mycursor = mydb.cursor()
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    hostname = socket.gethostname()
    num_cpus = cpu_count()

    sql = "INSERT INTO tasks (task_name,created_time,hostname,num_cpu) VALUES (%s,%s,%s,%s)"
    val = (task_name,now,hostname,num_cpus)
    mycursor.execute(sql, val)
    mydb.commit()

# Check if a task is being worked on by another computer
def is_task_in_progress(task_name):
    mycursor = mydb.cursor(buffered=True)
    mycursor.execute("SELECT * FROM tasks WHERE task_name = %s", (task_name,))
    result = mycursor.fetchone()
    mycursor.close()

    if result is not None:
        return True
    else:
        return False

def mark_task_complete(task_name):
    mycursor = mydb.cursor()
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sql = "UPDATE tasks SET status = 1, completed_time = %s WHERE task_name = %s"
    val = (now,task_name)
    mycursor.execute(sql, val)
    mydb.commit()
perfTIME = time.time()    

def sqlFetchone(q):
    connect_to_db()
    mycursor = mydb.cursor(buffered=True)
    mycursor.execute(q)
    res = mycursor.fetchone()
    mycursor.close()
    close_db()
    return res

def getTopParamsForWeekStarting(strt):
    q = f"select id,maLen,slpThres,adxLen,adxThresh,candles from niftystratview where startTime = '{strt}' and num_trades between 10 and 50 order by cast(retrn as decimal)/cast(drawdn as decimal) asc limit 10;"
    return sqlFetchone(q)
def previous_monday(date_str):
    dt = datetime.datetime.strptime(date_str, '%Y-%m-%d')

    weekday = dt.weekday()
    
    if weekday == 0: # If the given date is a Monday, we want the Monday from the last week
        days_to_subtract = 7
    else:
        days_to_subtract = weekday+7
    
    previous_monday_date = dt - datetime.timedelta(days=days_to_subtract)
    return previous_monday_date.strftime('%Y-%m-%d')

def applyConfigForDate(strt):
    prevMonday = previous_monday(strt)
    (id,maLen,slepThres,adxLen,adxThresh,candles) = getTopParamsForWeekStarting(prevMonday)
    print(f"monday:{prevMonday} id:{id} maLen:{maLen} slepThres:{slepThres} adxLen:{adxLen} adxThresh:{adxThresh} candles:{candles}")

cloud_args=''
#if 'cloud' in sys.argv:
cloud_args = 'cacheTickData:True zerodha_access_token:uzifIMYFPPnuRFLWVe3vM89jYKDHMM4V dbhost:trading.ca6bwmzs39pr.ap-south-1.rds.amazonaws.com dbuser:trading dbpass:trading123 dbname:trading'
iter = 0

def run_instance(args):
    argString = ''
    try: 
        arg_dict = {}
        arg_array = []
        for arg in args.split():
            key, value = arg.split(':')
            arg_dict[key] = value
            arg_array.append(arg)
            if key in ["cacheTickData","zerodha_access_token","dbhost","dbuser", "dbpass" ,"dbname"]:
                continue
            argString = argString + ' ' + arg
        argString = argString.strip()
        
        global iter
        
        iter = iter + 1
        # connect_to_db()
        print(f'{iter}: Runing: {argString}')
        # if is_task_in_progress(argString):
        #     print(f'SKIP - Already running task {argString}')
        #     return
        # add_task(argString)
        # close_db()
        subprocess.call(f'python3 backtest_combinator.py {args}', shell=True) 
    except Exception as e:
        print('Error in run_instance():', e)
        
def argGenerator():
    ## Varaibles for Orig Strategy
    # ma_lens = [10,15,20] # 5/50 absolutely sucks, 10 seems to be the best sharpe, 15 a close second with better returns than 10, but more variance, and 20 isnt bad 
    # band_widths = [1,1.5,2,2.5] # bw 2, w mini at .5, and supebw at 1 is the best
    # cfgMiniBandWidthMults = [0.5,0.75,1]
    # cfgSuperBandWidthMults = [1,1.25,1.5]
    # fast_ma_lens = [5,7] # 5 seems better, 7 not bad; used for exit, Is this a sign we should be exiting earlir ? try 2 here ? or just exit when trend following stops ?
    # adx_lens = [10,15,20,25] # 25 is the best, 20 is a far second.  Try 30 ?
    # adx_thresholds = [10,15,20,25,30,40,80] # 30 firmly the best(may change for longer adx len)
    
    #Variables for justFollowMA strategy
    
    start_date = "2022-05-01"
    end_date = "2023-03-31"

    startTimes = monthlyStartDates #generate_days(start_date, end_date, "Monday")
    ma_lens = [2,4,6,8,10,12] 
    band_widths = [1] 
    cfgMiniBandWidthMults = [0.5]
    cfgSuperBandWidthMults = [1]
    fast_ma_lens = [5] 
    adx_lens = [5,10,15,20,25,30] # 25 is the best, 20 is a far second.  Try 30 ?
    adx_thresholds = [10,15,20,25,30,40,80] # 30 firmly the best(may change for longer adx len)
    adx_thresh_yellow_multipliers = [0.6]
    num_candles_for_slope_proj = [2,5,7,9] # 2 or 5 works 
    atr_lens = [7,14,21]
    cfgTickers = ['NIFTYWEEKLYOPTION']
    
    param_iterator = itertools.product(ma_lens, band_widths, fast_ma_lens, adx_lens, adx_thresholds, adx_thresh_yellow_multipliers, num_candles_for_slope_proj,
                                atr_lens, cfgMiniBandWidthMults, cfgSuperBandWidthMults, cfgTickers, startTimes)

    # startCFG = getCfgFromFname(getFnameForLastTask())

    # if (startCFG is not None) and len(startCFG):
    #     start_index = list(itertools.product(ma_lens, band_widths, fast_ma_lens, adx_lens, adx_thresholds, adx_thresh_yellow_multipliers, num_candles_for_slope_proj, 
    #                                         atr_lens, cfgMiniBandWidthMults, cfgSuperBandWidthMults, cfgTickers)).index((startCFG['maLen'], startCFG['bandWidth'], 
    #                                         startCFG['fastMALen'], startCFG['adxLen'], startCFG['adxThresh'], startCFG['adxThreshYellowMultiplier'], startCFG['numCandlesForSlopeProjection'], 
    #                                         startCFG['atrLen'], startCFG['cfgMiniBandWidthMult'], startCFG['cfgSuperBandWidthMult'], startCFG['cfgTicker']))

    #     # Slice the iterator to start at the desired point -- where we left off at the last run 
    #     sliced_iterator = itertools.islice(param_iterator, start_index, None)
    # else: 
    sliced_iterator = param_iterator # start afresh if no last task found
        
    for params in sliced_iterator:
        ma_len, band_width, fast_ma_len, adx_len, adx_thresh, adx_thresh_yellow_multiplier, num_candles, atr_len, cfgMiniBandWidthMult, cfgSuperBandWidthMult, cfgTicker, startTime, \
        = params
        #check w db to see if this combination has been run before or is currently running
        # if not then mark it as running
        # do it
        # check that csv exists and mark it as done in db 
        
        argString = f"maLen:{ma_len} bandWidth:{band_width} fastMALen:{fast_ma_len} adxLen:{adx_len} adxThresh:{adx_thresh} adxThreshYellowMultiplier:{adx_thresh_yellow_multiplier} numCandlesForSlopeProjection:{num_candles} atrLen:{atr_len} cfgMiniBandWidthMult:{cfgMiniBandWidthMult} cfgSuperBandWidthMult:{cfgSuperBandWidthMult} cfgTicker:{cfgTicker} cfgZGetStartDate:{startTime}"
                
        # will run 3^5=243 * 2^3=8 = 1944 times == approx 20K min @ 10 min per run 
        # 100 paraaZllel cpu = 200 min = 3.2 hrs * $7/hr = $12.8
        # 64 cpu 20000/64 = 312.5 min = 5.2 hrs * $4/hr = $20.8
        
        
        # create view top_performers as select num_trades,FORMAT(max_drawdown_from_peak*100,2) as percentage,FORMAT(`return`*100,2) as percentage,round(sharpe_ratio,2),maLen,bandWidth,fastMALen,adxLen,adxThresh,adxThreshYellowMultiplier,numCandlesForSlopeProjection,atrLen,ma_slope_thresh,ma_slope_thresh_yellow_multiplier,obv_osc_thresh,obv_osc_thresh_yellow_multiplier from performance order by sharpe_ratio desc limit 10;

        yield f'{cloud_args} {argString}'
    

def argGeneratorTest():
    ma_lens = [5, 10, 15, 16, 17,18,19,20, 25]
    band_widths = [2]
    cfgTickers = ['NIFTY2341317700CE','BANKNIFTY2341341500CE', 'NIFTY23APRFUT']

    for params in itertools.product(ma_lens, band_widths, cfgTickers):
        ma_len, band_width, cfgTicker = params
        argString = f"maLen:{ma_len} bandWidth:{band_width} cfgTicker:{cfgTicker}"
        # if is_task_in_progress(argString):
        #     print(f'SKIP - Already running task {argString}')
        #     continue
        # add_task(argString)
        
        # will run 3^8 * 4 = 26.2K times == 10 parallel on pc, 8 on mac, 8 more on a amy mac
        #so 26 in parallel .. 1000 parallel runs will do it         
        print(f'{cloud_args} {argString}')
        yield f'{cloud_args} {argString}'
        
    # for maLen in [20,30]:
    #     for bandWidth in [2,4]:
    #         print(f"{cloud_args} maLen:{maLen} bandWidth:{bandWidth}")
    #         yield f'{cloud_args} maLen:{maLen} bandWidth:{bandWidth}'

    
if __name__ == '__main__':
    # Create a Pool object with number of processes equal to number of CPU cores
    pool = Pool(cpu_count())

    # Execute instances in parallel using the Pool object
    pool.map(run_instance, argGenerator())

    # Close the Pool object to free resources
    pool.close()
    pool.join()
    print (f"COMBINED took {round((time.time() - perfTIME)*1000,2)}ms")
    exit(0)
    
    
    # Done with multiprocessing
    # now merge the csv files into one

    # Directory containing CSV files
    directory = 'Data/backtest/nifty/'

    # List of CSV files in directory
    csv_files = [f for f in os.listdir(directory) if f.endswith('.csv')]

    # List of DataFrames loaded from CSV files
    dfs = []
    df_combined = pd.DataFrame()
    for file in csv_files:
        df = pd.read_csv(os.path.join(directory, file))
        df_combined = pd.concat([df_combined, df], axis=0)
    
#        dfs.append(df)

    # Concatenate DataFrames into single DataFrame
 #   df_combined = pd.concat(dfs, axis=0)
 #   df_combined = df_combined.T.reset_index().drop_duplicates(subset='index').set_index('index')
    #print (df_combined)
    # Write combined DataFrame to CSV file
    df_combined.to_csv(directory+'combined.csv', index=False)



# CREATE VIEW pview AS 
# SELECT num_trades,
#        FORMAT(max_drawdown_from_peak*100,2) AS drawdn,
#        FORMAT(`return`*100,2) AS retrn,
#        ROUND(sharpe_ratio,2) AS sharpe,
#         FORMAT(average_per_trade_return*100,2) AS avgRet,
#         FORMAT(std_dev_pertrade_return*100,2) AS stdDev,
#         ROUND(skewness_pertrade_return,1) AS skew,
#         ROUND(kurtosis_pertrade_return,1) AS kurtosis,
#         FORMAT(avg_daily_return*100,2) AS dayAv,
#         ROUND(sharpe_daily_return,2) AS dayShrp,
#         FORMAT(std_daily_return*100,2) AS dayStd,
#         ROUND(skew_daily_return,1) AS daySkew,
#         ROUND(kurtosis_daily_return,1) AS dayKurt,
#         maLen as maLen,
#        bandWidth as bw,
#        fastMALen as fstMA,
#        adxLen,
#        adxThresh,
#        adxThreshYellowMultiplier as axdMult,
#        numCandlesForSlopeProjection as candles,
#        atrLen,
#        ma_slope_thresh as slpThres,
#        ma_slope_thresh_yellow_multiplier as slpMult,
#        obv_osc_thresh as obvThres,
#        obv_osc_thresh_yellow_multiplier as obvMult
# FROM performancev2

