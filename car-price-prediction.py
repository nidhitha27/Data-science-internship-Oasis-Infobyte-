import warnings
import pandas as pd

warnings.filterwarnings('ignore')

data = pd.read_csv('car data.csv')

#Display Top 5 Rows of The Dataset
data.head()

#Check Last 5 Rows of The Dataset
data.tail()

#Find Shape of Our Dataset (Number of Rows And Number of Columns)
data.shape
print("Number of Rows",data.shape[0])
print("Number of Columns",data.shape[1])

#Get Information About Our Dataset Like the Total Number of Rows, Total Number of Columns, Datatypes of Each Column And Memory Requirement
data.info()

#Check Null Values In The Dataset
data.isnull().sum()

#Get Overall Statistics About The Dataset
data.describe()

#Data Preprocessing
data.head(1)

import datetime
date_time = datetime.datetime.now()
data['Age']=date_time.year - data['Year']
data.head()

data.drop('Year',axis=1,inplace=True)
data.head()

#Outlier Removal
import seaborn as sns
sns.boxplot(data['Selling_Price'])

sorted(data['Selling_Price'],reverse=True)

data = data[~(data['Selling_Price']>=33.0) & (data['Selling_Price']<=35.0)]
data.shape

#Encoding the Categorical Columns
data.head(1)

data['Fuel_Type'].unique()

data['Fuel_Type'] = data['Fuel_Type'].map({'Petrol':0,'Diesel':1,'CNG':2})
data['Fuel_Type'].unique()

data['Seller_Type'].unique()

data['Seller_Type'] = data['Seller_Type'].map({'Dealer':0,'Individual':1})

data['Seller_Type'].unique()

data['Transmission'].unique()

data['Transmission'] =data['Transmission'].map({'Manual':0,'Automatic':1})
data['Transmission'].unique()

data.head()

#Store Feature Matrix In X and Response(Target) In Vector y
X = data.drop(['Car_Name','Selling_Price'],axis=1)
y = data['Selling_Price']
y

#Splitting The Dataset Into The Training Set And Test Set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=42)

#Import the models
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor

#Model Training
lr = LinearRegression()
lr.fit(X_train,y_train)

rf = RandomForestRegressor()
rf.fit(X_train,y_train)

xgb = GradientBoostingRegressor()
xgb.fit(X_train,y_train)

xg = XGBRegressor()
xg.fit(X_train,y_train)

#Prediction on Test Data
y_pred1 = lr.predict(X_test)
y_pred2 = rf.predict(X_test)
y_pred3 = xgb.predict(X_test)
y_pred4 = xg.predict(X_test)

#Evaluating the Algorithm
from sklearn import metrics
score1 = metrics.r2_score(y_test,y_pred1)
score2 = metrics.r2_score(y_test,y_pred2)
score3 = metrics.r2_score(y_test,y_pred3)
score4 = metrics.r2_score(y_test,y_pred4)
print(score1,score2,score3,score4)

final_data = pd.DataFrame({'Models':['LR','RF','GBR','XG'],
             "R2_SCORE":[score1,score2,score3,score4]})
final_data

#Save The Model
xg = XGBRegressor()
xg_final = xg.fit(X,y)
import joblib
joblib.dump(xg_final,'car_price_predictor')
model = joblib.load('car_price_predictor')

#Prediction on New Data
import pandas as pd
data_new = pd.DataFrame({
    'Present_Price':5.59,
    'Kms_Driven':27000,
    'Fuel_Type':0,
    'Seller_Type':0,
    'Transmission':0,
    'Owner':0,
    'Age':8
},index=[0])
model.predict(data_new)




















