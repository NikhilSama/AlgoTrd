from multiprocessing import Pool
import subprocess
import os
import pandas as pd
import sys

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

        subprocess.call(f'python3.8 nifty_backtest.py {args}', shell=True) 
    except Exception as e:
        print('Error in run_instance():', e)

def argGenerator():
    for maLen in [10,20,30]:
        for bandWidth in [2,2.5,3,3.5,4]:
            for fastMALen in [5,7,10]:
                for adxLen in [10,14,20]:
                    for adxThresh in [20,25,30,35,40]:
                        for adxThreshYellowMultiplier in [0.5,0.6,0.7,0.8,0.9]:
                            for adxThreshYellowMultiplier in [0.5,0.6,0.7,0.8,0.9]:
                                print(f'{cloud_args} maLen:{maLen} bandWidth:{bandWidth} fastMALen:{fastMALen} adxLen:{adxLen} adxThresh:{adxThresh} adxThreshYellowMultiplier:{adxThreshYellowMultiplier} ')
                                yield f'{cloud_args} maLen:{maLen} bandWidth:{bandWidth} fastMALen:{fastMALen} adxLen:{adxLen} adxThresh:{adxThresh} adxThreshYellowMultiplier:{adxThreshYellowMultiplier} '
    

def argGeneratorTest():
    for maLen in [20,30]:
        for bandWidth in [2,4]:
            print(f"{cloud_args} maLen:{maLen} bandWidth:{bandWidth}")
            yield f'{cloud_args} maLen:{maLen} bandWidth:{bandWidth}'

    
if __name__ == '__main__':
    # List of argument strings to pass to instances
    args_list = ['maLen:20 cacheTickData:True', 'maLen:200 cacheTickData:True', 'maLen:8 cacheTickData:True']

    # Create a Pool object with number of processes equal to number of CPU cores
    pool = Pool()

    # Execute instances in parallel using the Pool object
    pool.map(run_instance, argGeneratorTest())

    # Close the Pool object to free resources
    pool.close()
    pool.join()

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
