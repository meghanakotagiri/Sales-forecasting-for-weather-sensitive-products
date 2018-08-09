1) The data files have been taken from the link https://www.kaggle.com/c/walmart-recruiting-sales-in-stormy-weather/data

(train.csv, test.csv, key.csv, weather.csv)

2) The holiday list has been taken from the site http://www.timeanddate.com/holidays/us/

(holiday.txt, holidaynames.txt)

3) For storing store-item combinations with non-zero sales and store-item combinations with zero sales:
command: python createValidStorenItemComb.py  
 
4)(i) for training:
 command: python Train.py
Running this file, would generate the 'rf.pkl' file, where the  randomforest model has been stored
It would also generate ARMA models for all store and item combinations where time series is stationary and store these models in 'models' directory. Ex 36_1.pkl in file that will be created in 'models' directory refers to ARMA model for 36th store no & 1st item no combination.

(ii) for testing
command: python Test.py 
It generates 2 csv files ( for 2 approaches used) with units sold corresponding to each date, store, item in test data. This csv file, we submitted to kaggle for evaluation and the final scores we got are mentioned in the report.


TEAM: G Neha, Meghana Kotagiri, Dakshayani Vadari