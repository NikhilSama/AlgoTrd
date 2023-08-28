import datetime
from DatabaseLogin import DBBasic
import pandas as pd
import utils
frm = datetime.datetime(2022,5,1,9,15) 
to = datetime.datetime(2022,5,30,15,30)

db = DBBasic()

# Receive a df for a single expiry
# Calculate option pain for each strike
# in that expiry and return a df
def getPainByStrike(df,spot_price):
    print(f"Spot Price: {spot_price}")
    df['Pain'] = 0
    df = df[(df['strike'].astype(float) < spot_price + 400) & (df['strike'].astype(float) > spot_price - 400)]    
    for index, row in df.iterrows():
        oi = float(row['Open_Interest'])
        strike = float(row['strike'])
        if row['type'] == 'CE':
            if strike > spot_price:
                df.at[index, 'Pain'] = oi * (strike - spot_price)
        elif row['type'] == 'PE':
            if strike < spot_price:
                df.at[index, 'Pain'] = oi * (spot_price- strike)
                
    new_df = df.groupby('strike').agg({'Pain': ['sum']})
    new_df = new_df[new_df[('Pain', 'sum')] != 0]
    
    top_3_strikes = new_df.nsmallest(3, ('Pain', 'sum'))
    if len(top_3_strikes) == 0:
        return (0,0)
    maxpain = top_3_strikes.index[0]    
    inverse_weights = 1 / top_3_strikes[('Pain', 'sum')]
    normalized_weights = inverse_weights / inverse_weights.sum()
    weightedAvMaxPain = (top_3_strikes.index.astype(float) * normalized_weights).sum()
    weightedAvMaxPain = round(weightedAvMaxPain,0)
    print(f"Max Pain: {maxpain} Weighted Avg Strike Price: {weightedAvMaxPain}")
    return (maxpain,weightedAvMaxPain)

def getPCR(df,spot_price):
    pe_open_interest_sum = df.loc[df['type'] == 'PE', 'Open_Interest'].sum()
    ce_open_interest_sum = df.loc[df['type'] == 'CE', 'Open_Interest'].sum()
    pcr = pe_open_interest_sum / ce_open_interest_sum

    atm_df = df.loc[(df['strike'].astype(int) >= spot_price - 200) & (df['strike'].astype(int) <= spot_price + 200)]
    atm_pe_open_interest_sum = atm_df.loc[atm_df['type'] == 'PE', 'Open_Interest'].sum()
    atm_ce_open_interest_sum = atm_df.loc[atm_df['type'] == 'CE', 'Open_Interest'].sum()
    atm_pcr = atm_pe_open_interest_sum / atm_ce_open_interest_sum
    return (pcr, atm_pcr)

def getLTPain(d):
    optionPain = pd.DataFrame()
    df = db.getOptionChain(d)
    earliest_expiry = df['expiry'].min()
    second_earliest_expiry = df[df['expiry'] > earliest_expiry]['expiry'].min()

    # Convert both dates to date format (without time) to calculate the difference in days correctly
    target_expiry = earliest_expiry if (earliest_expiry.date() - d.date()).days > 2 else second_earliest_expiry
    df = df[df['expiry'] == target_expiry]

    # get a snapshot at the first time of day
    min_time = df['Time'].min()
    df = df[df['Time'] == min_time]
    dt = df['Date'].iloc[0]
    dt = pd.to_datetime(dt) + pd.to_timedelta(min_time)
    ltp = db.getNiftyPrice(dt)
    # Access expiry and time variables here
    (maxPain, wmaxPain) = getPainByStrike(df, ltp)   
    (pcr, atm_pcr) = getPCR(df, ltp)
    optionPain = pd.concat([optionPain, pd.DataFrame({'expiry': target_expiry, 'time': dt, 'maxPain': maxPain, 'wmaxPain': wmaxPain, 'pcr': pcr, 'atm_pcr': atm_pcr}, index=[0])], ignore_index=True)

    print(f"Expiry: {target_expiry} Time: {min_time} Date: {dt} Max Pain: {maxPain}, W Max Pain: {wmaxPain}, {pcr}, {atm_pcr}")

    # now we have the option chain snapshot for target expiry at the first time of today
    
    return optionPain
    
def getPainForDay(d):
    optionPain = pd.DataFrame()
    df = db.getOptionChain(d)
    earliest_expiry = df['expiry'].min()
    second_earliest_expiry = df[df['expiry'] > earliest_expiry]['expiry'].min()

    df = df[df['expiry'] == earliest_expiry]
        
    grouped_df = df.groupby(['expiry', 'Time'])
    for (expiry, time), group in grouped_df:
        # print(group)
        dt = group['Date'].iloc[0]
        # Combine date and time to form a datetime variable called dt
        dt = pd.to_datetime(dt) + pd.to_timedelta(time)
        if dt.time() < datetime.time(9,30) or dt.time() > datetime.time(15,30):
            continue
        ltp = db.getNiftyPrice(dt)
        # Access expiry and time variables here
        (maxPain, wmaxPain) = getPainByStrike(group, ltp)   
        (pcr, atm_pcr) = getPCR(group, ltp)
        
        print(f"Expiry: {expiry} Time: {time} Date: {dt} Max Pain: {maxPain}, W Max Pain: {wmaxPain}, {pcr}, {atm_pcr}")
        optionPain = pd.concat([optionPain, pd.DataFrame({'expiry': expiry, 'time': dt, 'maxPain': maxPain, 'wmaxPain': wmaxPain, 'pcr': pcr, 'atm_pcr': atm_pcr}, index=[0])], ignore_index=True)
    return (optionPain)

def saveMaxPain():
    frm = datetime.datetime(2022,5,1,9,15) 
    to = datetime.datetime(2023,3,30,15,30)

    day = frm
    
    while day <= to:
        if utils.isTradingDay(day):
            df = getPainForDay(day)
            print(df)
            db.toDB('maxpain',df)
        else:
            print(f"Skipping {day}")
        day = day + datetime.timedelta(days=1)

def saveLTMaxPain():
    frm = datetime.datetime(2022,5,1,9,15) 
    to = datetime.datetime(2023,3,30,15,30)

    day = frm
    
    while day <= to:
        if utils.isTradingDay(day):
            df = getLTPain(day)
            print(df)
            db.toDB('ltmaxpain',df)
        else:
            print(f"Skipping {day}")
        day = day + datetime.timedelta(days=1)


                
saveMaxPain() 