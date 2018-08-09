import pandas as pd
import numpy as np
import pickle
import os.path
import datetime
from dateutil.relativedelta import relativedelta
import statsmodels.tsa.api as tsa
import math
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import adfuller
from sklearn.ensemble import RandomForestRegressor

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

def create_rollingmean(_df,valid_store_items):
    dfs = []
    for sno, ino in valid_store_items:
        df = _df[(_df.store_nbr == sno) & (_df.item_nbr == ino)].copy()
        df = df.set_index('date2', drop=False)
        df = df.sort_index()
        
        # calculate rolling mean
        window = 21
        df['rmean'] = pd.rolling_mean(df.log1p, window, center=True)
        df['rmean'] = df['rmean'].interpolate()
        df['rmean'] = df['rmean'].ffill()
        df['rmean'] = df['rmean'].bfill()

        # alldates
        alldates = pd.date_range('2012-01-01', '2014-10-31', freq='D')
        alldates.name = 'date2'
        df2 = pd.DataFrame(None, index = alldates)

        df2['store_nbr'] = sno
        df2['item_nbr'] = ino

        df2['log1p'] = df.log1p
        df2['rmean'] = df.rmean
        df2['rmean'] = df2['rmean'].interpolate()
        df2['rmean'] = df2['rmean'].ffill()
        df2['rmean'] = df2['rmean'].bfill()
        df2 = df2.reset_index()

        EPS = 0.000001
        df2['include1'] = (df2.rmean > EPS)
        dfs.append(df2)

    return pd.concat(dfs, ignore_index=True)

def test_stationarity(timeseries):
    dftest = adfuller(timeseries, autolag='AIC')
    if dftest[1] > 0.05:
        return True
    return False

# storing store & Item combination where time series is stationary & non-stationary
def create_stat_nonstat_combs(_df,dfs1,dfs2):
    for sno, ino in store_items:
        df = _df[(_df.store_nbr == sno) & (_df.item_nbr == ino)].copy()
        df = df.set_index('date2', drop=False)
        df = df.sort_index()
        flag=test_stationarity(df.log1p)
        if flag:
            dfs1.append([sno,ino])
        else:
            dfs2.append([sno,ino])
    


#creating ARMA model for each valid store,item combination (where the time series is stationary)
def create_arma(_df,store_items,filepath,exceptional):
    extra_arr=[]
    for sno, ino in store_items:   
        df = _df[(_df.store_nbr == sno) & (_df.item_nbr == ino)].copy()
        df = df.set_index('date2', drop=False)
        df = df.sort_index()
        start = datetime.datetime.strptime("2012-01-01", "%Y-%m-%d")
        date_list = [start + relativedelta(days=x) for x in range(0,len(df.index))]
        
        df['index'] =date_list
        df.set_index(['index'], inplace=True)
        df.index.name=None
        try:
            arma=tsa.ARMA(df.log1p,order=(1,1))
            extra_arr.append(len(date_list))
            results=arma.fit()
            print(sno,ino,len(date_list))
        except:
            exceptional.append([sno,ino])           
        else:
            out=open(filepath+str(sno)+'_'+str(ino)+'.pkl','wb')
            pickle.dump(results,out)
            out.close()
    
    # returns an array that contains number of dates it is trained on for each stationary item,store_no combo       
    return extra_arr

# read dataframes
key = pd.read_csv("data/key.csv")
wtr = pd.read_csv("data/weather.csv")
holidays = get_holidays("data/holiday.txt")
holiday_names = get_holiday_names("data/holidaynames.txt")

store_item_nbrs_path = 'store_item_nbrs.csv'
store_item_nbrs = pd.read_csv(store_item_nbrs_path)
valid_store_items = set(zip(store_item_nbrs.store_nbr, store_item_nbrs.item_nbr))

# preprocess 
df_train = pd.read_csv("data/train.csv")
mask_train = [(sno_ino in valid_store_items) for sno_ino in zip(df_train['store_nbr'], df_train['item_nbr']) ]
df_train = df_train[mask_train].copy()
df_preprocessed = preprocess(df_train, True)

#creating rolling mean
df_rollingmean = create_rollingmean(df_preprocessed)
df_rollingmean.to_pickle('df_rollingmean.pkl')
df_pre = df_preprocessed.merge(df_rollingmean[['item_nbr', 'store_nbr', 'date2', 'rmean', 'include1']],
                      how = 'left',
                      on = ['item_nbr', 'store_nbr', 'date2'])
df_pre = df_pre.reset_index(drop = True)

#Feature vector
df_feature_vector = df_pre[[ 'store_nbr' , 'item_nbr'  , 'station_nbr' ,
                                     'preciptotal_flag' , 'depart_flag',  'weekday',  'is_weekend',  'is_holiday',
                                    'is_holiday_weekday' , 'is_holiday_weekend',  'day',  'month',  'year','rmean']]

#response variable
df_train_target=df_pre[['log1p']]

#Training the randomForest model
rf = RandomForestRegressor()
rf.fit(df_feature_vector,df_train_target.values.ravel()) 
# Storing the random forest model
output=open('models/rf.pkl','wb')
pickle.dump(rf,output)
output.close()

importances = rf.feature_importances_
std = np.std([tree.feature_importances_ for tree in rf.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")
for f in range(df_feature_vector.shape[1]):
    print("%d. feature %d (%f) %s" % (f + 1, indices[f], importances[indices[f]], list(df_feature_vector)[indices[f]]))

############# Building ARMA model ###############



store_items = zip(store_item_nbrs.store_nbr, store_item_nbrs.item_nbr)
non_stat_combs=[]
stat_combs=[]
create_stat_nonstat_combs(df_new,non_stat_combs,stat_combs)

#creating ARMA model for stationary store and item combinations
df_new=df_preprocessed.copy()
filepath="models/"
exceptional=[]
r=create_arma(df_new,stat_combs,filepath,exceptional)

# storing store and item combinations for which arima model is built
x=open("models/arma_store_item.pkl","wb")
pickle.dump(r,x)
x.close()
stat_combs_safe=stat_combs

for i in exceptional:
    stats_combs_safe.remove(i)
    non_stat_coms.append(i)

# storing Store & Item combinations where time series is  stationary
with open('store_item_stationary.csv', 'w') as f: 
    f.write("store_nbr,item_nbr\n")
    for sno, ino in stat_combs_safe:
          f.write("{},{}\n".format(sno, ino))

# storing Store & Item combinations where time series is not stationary
with open('store_item_non_stationary.csv', 'w') as f: 
    f.write("store_nbr,item_nbr\n")
    for sno, ino in non_stat_combs:
          f.write("{},{}\n".format(sno, ino))
