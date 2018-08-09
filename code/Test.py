import pandas as pd
import numpy as np
import pickle
import os.path
from sklearn.ensemble import RandomForestRegressor
from dateutil.relativedelta import relativedelta

def get_holidays(fpath):
    # holidays are from http://www.timeanddate.com/holidays/us/ , holidays and some observances
    
    f = open(fpath)
    lines = f.readlines()
    lines = [line.split(" ")[:3] for line in lines]
    lines = ["{} {} {}".format(line[0], line[1], line[2]) for line in lines]
    lines = pd.to_datetime(lines)
    return pd.DataFrame({"date2":lines})

def get_holiday_names(fpath):
    # holiday_names are holidays + around Black Fridays
    
    f = open(fpath)
    lines = f.readlines()
    lines = [line.strip().split(" ")[:4] for line in lines]
    lines_dt = ["{} {} {}".format(line[0], line[1], line[2]) for line in lines]
    lines_dt = pd.to_datetime(lines_dt)
    lines_hol = [line[3] for line in lines]
    return pd.DataFrame({"date2":lines_dt, "holiday_name":lines_hol})

def to_float(series, replace_value_for_M, replace_value_for_T):
    series = series.map(lambda s : s.strip())
    series[series == 'M'] = replace_value_for_M
    series[series == 'T'] = replace_value_for_T
    return series.astype(float)

def preprocess(_df, is_train):
    
    df = _df.copy()

    # log1p
    if is_train: 
        df['log1p'] = np.log(df['units'] + 1)

    # date
    df['date2'] = pd.to_datetime(df['date'])

    # weather features
    wtr['date2'] = pd.to_datetime(wtr.date)
    wtr["preciptotal2"] = to_float(wtr["preciptotal"], 0.00, 0.005)
    wtr["preciptotal_flag"] = np.where(wtr["preciptotal2"] > 0.2, 1.0, 0.0)

    wtr["depart2"] = to_float(wtr.depart, np.nan, 0.00)
    wtr["depart_flag"] = 0.0
    wtr["depart_flag"] = np.where(wtr["depart2"] < -8.0, -1, wtr["depart_flag"])
    wtr["depart_flag"] = np.where(wtr["depart2"] > 8.0 ,  1, wtr["depart_flag"])
    df = pd.merge(df, key, on='store_nbr')
    df = pd.merge(df, wtr[["date2", "station_nbr", "preciptotal_flag", "depart_flag"]], 
                      on=["date2", "station_nbr"])
    
    # weekday
    df['weekday'] = df.date2.dt.weekday
    df['is_weekend'] = df.date2.dt.weekday.isin([5,6])
    df['is_holiday'] = df.date2.isin(holidays.date2)
    df['is_holiday_weekday'] = df.is_holiday & (df.is_weekend == False)
    df['is_holiday_weekend'] = df.is_holiday &  df.is_weekend

    # bool to int (maybe no meaning)
    df.is_weekend = np.where(df.is_weekend, 1, 0)
    df.is_holiday = np.where(df.is_holiday, 1, 0)
    df.is_holiday_weekday = np.where(df.is_holiday_weekday, 1, 0)
    df.is_holiday_weekend = np.where(df.is_holiday_weekend, 1, 0)
    
    # day, month, year
    df['day'] = df.date2.dt.day
    df['month'] = df.date2.dt.month
    df['year'] = df.date2.dt.year
    
    # around BlackFriday
    df = pd.merge(df, holiday_names, on='date2', how = 'left')
    df.loc[df.holiday_name.isnull(), "holiday_name"] = ""

    around_BlackFriday = ["BlackFridayM3", "BlackFridayM2", "ThanksgivingDay", "BlackFriday",
                          "BlackFriday1", "BlackFriday2", "BlackFriday3"]
    df["around_BlackFriday"] = np.where(df.holiday_name.isin(around_BlackFriday), 
                                        df.holiday_name, "Else")

    return df

# read dataframes
key = pd.read_csv("data/key.csv")
wtr = pd.read_csv("data/weather.csv")
holidays = get_holidays("data/holiday.txt")
holiday_names = get_holiday_names("data/holidaynames.txt")

store_item_nbrs_path = 'store_item_nbrs.csv'
store_item_nbrs = pd.read_csv(store_item_nbrs_path)
valid_store_items = set(zip(store_item_nbrs.store_nbr, store_item_nbrs.item_nbr))

# preprocess 
df_test = pd.read_csv("data/test.csv")
mask_test = [(sno_ino in valid_store_items) for sno_ino in zip(df_test['store_nbr'], df_test['item_nbr']) ]
df_test2 = df_test[mask_test].copy()

df_preprocessed = preprocess(df_test2, False)
df_feature_vector = df_preprocessed[[ 'store_nbr' , 'item_nbr'  , 'station_nbr' ,
                                     'preciptotal_flag' , 'depart_flag',  'weekday',  'is_weekend',  'is_holiday',
                                    'is_holiday_weekday' , 'is_holiday_weekend',  'day',  'month',  'year']]

store_item_nbrs_path2 = 'invalid_store_item_nbrs.csv'
store_item_nbrs2 = pd.read_csv(store_item_nbrs_path2)
valid_store_items2 = set(zip(store_item_nbrs2.store_nbr, store_item_nbrs2.item_nbr))
df_test_indices_left=[(sno_ino in valid_store_items2) for sno_ino in zip(df_test['store_nbr'], df_test['item_nbr']) ]    
df_test3 = df_test[df_test_indices_left].copy()

df_rollingmean = pd.read_pickle('weather_ml/df_rollingmean.pkl')
df_pre = df_preprocessed.merge(df_rollingmean[['item_nbr', 'store_nbr', 'date2', 'rmean', 'include1', 'include2']],
                      how = 'left',
                      on = ['item_nbr', 'store_nbr', 'date2'])
df_pre = df_pre.reset_index(drop = True)
df_feature_vector = df_pre[[ 'store_nbr' , 'item_nbr'  , 'station_nbr' ,
                                     'preciptotal_flag' , 'depart_flag',  'weekday',  'is_weekend',  'is_holiday',
                                    'is_holiday_weekday' , 'is_holiday_weekend',  'day',  'month',  'year','rmean']]


# predicting using random forest for non-stationary store and item combinations
f=open('models/rf.pkl','rb')
rf=pickle.load(f)
predicted_log=rf.predict(df_feature_vector)
predicted_units=np.round(np.exp(predicted_log)-1)

result=open('result1.txt','w')
result.write("id,units\n")

#Printing results for store-item combination where sales have been consistently zero for three years
for i in range(0,len(df_test3)):
    result.write("%d_%d_%s,%d\n" % (df_test3[['store_nbr']].iloc[i], df_test3[['item_nbr']].iloc[i], df_test3[['date']].iloc[i].values.ravel()[0],0))
result.close()

# result1 stores prediction for sales on test data using random forest model alone
#############################################################################################################

#printing results for store and items combinations where time-series is non-stationary 
non_stat=pd.read_csv('store_item_non_stationary.csv')
output=open('result_nonst.txt','w')
df_new=df_pre.copy()
nonst_store=zip(non_stat.store_nbr,non_stat.item_nbr)
for sno, ino in nonst_store:
       df=df_new[(df_new.store_nbr== sno) & (df_new.item_nbr == ino)].copy()
       if (len(df.index)==0):
            continue
       df_feature_vec = df[[ 'store_nbr' , 'item_nbr'  , 'station_nbr' ,
                                     'preciptotal_flag' , 'depart_flag',  'weekday',  'is_weekend',  'is_holiday',
                                    'is_holiday_weekday' , 'is_holiday_weekend',  'day',  'month',  'year','rmean']]
       predicted_log=lreg.predict(df_feature_vec)
       predicted_units=np.round(np.exp(predicted_log)-1)
       for i in range(0,len(predicted_units)):
           output.write("%d_%d_%s,%d\n" % (sno,ino, df[['date']].iloc[i].values.ravel()[0],predicted_units[i]))


#Predicting units for store and items combinations where time-series is stationary 
res=open('result_arma.txt','w')
stat=pd.read_csv('store_item_stationary.csv')
st_store=zip(stat.store_nbr,stat.item_nbr)
list=open('models/arma_store_item.pkl','rb')
i=0

for sno, ino in st_store:
       df=df_new[(df_new.store_nbr== sno) & (df_new.item_nbr == ino)].copy()
       if (len(df.index)==0):
            i=i+1
            continue
       df = df.set_index('date2', drop=False)
       df = df.sort_index()
       start = datetime.datetime.strptime("2012-01-01", "%Y-%m-%d")
       date_list = [(datetime.datetime.strptime(x, "%Y-%m-%d")-start).days for x in df.date]
       into=open('models/'+str(sno)+'_'+str(ino)+'.pkl','rb')
       results=pickle.load(into)
       for x in range(0,len(date_list)):
           if (date_list[x]<(list[i]-1)): 
               resultsarr = results.predict(start=date_list[x],end=date_list[x])
               predicted_units=np.round(np.exp(resultsarr)-1)
               res.write("%d_%d_%s,%d\n" % (sno,ino, df[['date']].iloc[x].values.ravel()[0],predicted_units))
           else:
               steps=date_list[-1]-list[i]
               resultsarr=results.forecast(steps)
                    
               for j in range(x,len(date_list)): 
                   predicted_units=np.round(np.exp(resultsarr[0][date_list[j]-list[i]-1])-1) 
                   res.write("%d_%d_%s,%d\n" % (sno,ino, df[['date']].iloc[j].values.ravel()[0],predicted_units))      
               break
       i=i+1               
res.close()          
            
#This part generates another csv that uses ARMA models for stationary store-item combinations and regressor for non stationary ones. 
##############################################################################################################################