from multiprocessing import Pool
import subprocess
import os
import pandas as pd
import sys
import itertools 
import time 
perfTIME = time.time()    

cloud_args=''
if 'cloud' in sys.argv:
    cloud_args = 'cacheTickData:True zerodha_access_token:7Xzo3aGAQ06z2cvT8AyppbT2Uh6BD4Ru dbhost:NikhilSama.mysql.pythonanywhere-services.com dbuser:NikhilSama dbpass:trading123 dbname:NikhilSama\$default'
def run_instance(args):
    try: 
        arg_dict = {}
        arg_array = []
        for arg in args.split():
            key, value = arg.split(':')
            arg_dict[key] = value
            arg_array.append(arg)

        subprocess.call(f'python3.8 backtest.py {args}', shell=True) 
    except Exception as e:
        print('Error in run_instance():', e)

def argGenerator():

    ma_lens = [10, 20, 30]
    band_widths = [2, 2.5, 3]
    fast_ma_lens = [5, 7, 10]
    adx_lens = [10, 14, 20]
    adx_thresholds = [20, 30, 40]
    adx_thresh_yellow_multipliers = [0.5, 0.7, 0.9]
    num_candles_for_slope_proj = [1, 2, 4, 6,]
    atr_lens = [8,14,20]
    super_lens = [200]
    super_bandWidths = [2.5]
    adx_slope_threshes = [0.02, 0.06, 0.10]
    
    for params in itertools.product(ma_lens, band_widths, fast_ma_lens, adx_lens, adx_thresholds, adx_thresh_yellow_multipliers, num_candles_for_slope_proj,
                                    atr_lens, super_lens, super_bandWidths, adx_slope_threshes):
        ma_len, band_width, fast_ma_len, adx_len, adx_thresh, adx_thresh_yellow_multiplier, num_candles, atr_len, super_len, super_bandWidth, adx_slope_thresh, \
        = params
        #check w db to see if this combination has been run before or is currently running
        # if not then mark it as running
        # do it
        # check that csv exists and mark it as done in db 
        
        
        # will run 3^8 * 4 = 26.2K times == 10 parallel on pc, 8 on mac, 8 more on a amy mac
        #so 26 in parallel .. 1000 parallel runs will do it 

        print(f'{cloud_args} maLen:{ma_len} bandWidth:{band_width} fastMALen:{fast_ma_len} adxLen:{adx_len} adxThresh:{adx_thresh} adxThreshYellowMultiplier:{adx_thresh_yellow_multiplier}')
        yield f'{cloud_args} maLen:{ma_len} bandWidth:{band_width} fastMALen:{fast_ma_len} adxLen:{adx_len} adxThresh:{adx_thresh} adxThreshYellowMultiplier:{adx_thresh_yellow_multiplier} numCandlesForSlopeProjection:{num_candles} atrLen:{atr_len} superLen:{super_len} superBandWidth:{super_bandWidth} adxSlopeThresh:{adx_slope_thresh}'
    # for maLen in [10,20,30]:
    #     for bandWidth in [2,2.5,3,3.5,4]:
    #         for fastMALen in [5,7,10]:
    #             for adxLen in [10,14,20]:
    #                 for adxThresh in [20,25,30,35,40]:
    #                     for adxThreshYellowMultiplier in [0.5,0.6,0.7,0.8,0.9]:
    #                         for numCandlesForSlopeProjection in [1,2.3,4,5,6,7,8,9,10]:
    #                             print(f'{cloud_args} maLen:{maLen} bandWidth:{bandWidth} fastMALen:{fastMALen} adxLen:{adxLen} adxThresh:{adxThresh} adxThreshYellowMultiplier:{adxThreshYellowMultiplier} ')
    #                             yield f'{cloud_args} maLen:{maLen} bandWidth:{bandWidth} fastMALen:{fastMALen} adxLen:{adxLen} adxThresh:{adxThresh} adxThreshYellowMultiplier:{adxThreshYellowMultiplier} '
    

def argGeneratorTest():
    ma_lens = [10, 20, 25, 30]
    band_widths = [2, 2.5, 3]
    for params in itertools.product(ma_lens, band_widths):
        ma_len, band_width = params
        print(f"{cloud_args} maLen:{ma_len} bandWidth:{band_width}")
        yield f'{cloud_args} maLen:{ma_len} bandWidth:{band_width}'

    # for maLen in [20,30]:
    #     for bandWidth in [2,4]:
    #         print(f"{cloud_args} maLen:{maLen} bandWidth:{bandWidth}")
    #         yield f'{cloud_args} maLen:{maLen} bandWidth:{bandWidth}'

    
if __name__ == '__main__':
    # List of argument strings to pass to instances
    args_list = ['maLen:20 cacheTickData:True', 'maLen:200 cacheTickData:True', 'maLen:8 cacheTickData:True']

    # Create a Pool object with number of processes equal to number of CPU cores
    pool = Pool(processes=8)

    # Execute instances in parallel using the Pool object
    pool.map(run_instance, argGeneratorTest())

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
