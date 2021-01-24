import numpy as np
import pandas as pd

#1. 데이터 
df = pd.read_csv('../../data/csv/삼성전자0115.csv', index_col=0, header=0, encoding='cp949', thousands=',') 
df2 = df.iloc[::-1].reset_index(drop=True)
df2 = df2.where(pd.notnull(df2), df2.mean(), axis='columns')     # 결측치(비어있는 데이터)에  평균으로 데이터값 집어넣기



# x, y 데이터 지정
x = df.iloc[1:80, [0,2,3,4,6,8]]
y = df.iloc[2:81, 1]
x_pred = df.iloc[1:80, [0,2,3,4,6,8]]


#train , val
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=311)
from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.8, shuffle=True, random_state=311)

x_pred = x_pred.values.reshape(1,-1)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_val = scaler.transform(x_val)
x_test = scaler.transform(x_test)
x_pred = scaler.transform(x_pred)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_val = x_val.reshape(x_val.shape[0], x_val.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)
x_pred = x_pred.reshape(x_pred.shape[0], x_pred.shape[1], 1)
print(x_train)
print(x_val)
print(x_test)
print(x_pred)
# 저장
np.save('../../data/npy/samsung_2_totaldata.npy',arr=([x_train, y_train, x_val, y_val, x_test, y_test,x_pred]))

