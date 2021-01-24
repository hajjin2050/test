import numpy as np
from numpy.core.fromnumeric import size
import pandas as pd
from tensorflow.keras import datasets
from tensorflow.keras.layers import Dropout,Dense,GRU,Input,LSTM
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import concatenate

#1

df = pd.read_csv('../../data/csv/samsung.csv',encoding='cp949',thousands = ',', index_col=0, header=0) 
df1 = pd.read_csv('../../data/csv/삼성전자0115.csv',encoding='cp949',thousands = ',', index_col=0, header=0) 

a = df[['종가','고가','저가','금액(백만)','거래량','시가']]
a1 = df1[['종가','고가','저가','금액(백만)','거래량','시가']]

a.columns = ['close','high','low','price(million)', 'volume','start'] # 열(columns)의 이름변경
a1.columns = ['close','high','low','price(million)', 'volume','start'] # 열(columns)의 이름변경


a = a.loc[::-1]
a1 = a1.loc[::-1]

print(a1)
print(a1.tail()) # 디폴트 5 = df[-5:]
print(a1.info())
print(a1.describe())

print(a)
print(a.info())
x1 = a1.dropna(axis=0)
print('=======')
print(x1)

# a = pd.to_datetime(a)
# a = pd.to_numeric(a)

# s1 = a.iloc[0 : 1738,:].astype(float)
s1 = a.iloc[1738:-1,:]
s2 = x1.iloc[-3:2400,:].astype('float')

print(s1)
print('=============')
print(s2.head())


c = pd.concat([s1,s2])
print(c)


print(c.info())
# print(c.head())
# print(c.tail())
# print(c.shape)
x = c.dropna(axis=0).values

y1 = x[6:663,5]
y2 = x[7:664,5]
y1 = y1[:, np.newaxis]
y2 = y2[:, np.newaxis]
print(y1)
print(y2)


y = np.hstack((y1,y2))

def split_x(seq,size,col):
    aaa=[]
    for i in range(len(seq) - size + 1):
        subset = seq[i:(i+size),0:col].astype('float32')
        aaa.append(np.array(subset))
    return np.array(aaa)

size = 6
col = 6

dataset = split_x(x,size, col)


x = dataset[:-2,:7,:]
x_pred = dataset[-2:,:,:]



print(x.shape)
print(y.shape)
print(x_pred.shape)


x_train, x_test, y_train, y_test = train_test_split(x,y, train_size = 0.8, random_state=104)
x_train, x_val, y_train, y_val = train_test_split(x_train,y_train,train_size = 0.8, random_state=104)

x = x.reshape(-1, 1)
x_train = x_train.reshape(-1, 1)
x_test = x_test.reshape(-1, 1)
x_val = x_val.reshape(-1,1)
x_pred = x_pred.reshape(1, -1)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)
x_pred = scaler.transform(x_pred)

print(x_train.shape) 
print(x_test.shape)

x = x.reshape(-1, 6, 6)
x_train = x_train.reshape(-1, 6, 6)
x_test = x_test.reshape(-1, 6, 6)
x_val = x_val.reshape(-1, 6, 6)
x_pred = x_pred.reshape(-1, 6, 6)


print(x_train.shape) 
print(x_test.shape)



#1 - 2

dfk = pd.read_csv('../../data/csv/KODEX 코스닥150 선물인버스.csv',encoding='cp949',thousands = ',', index_col=0, header=0) 

ak = dfk[['종가','고가','저가','금액(백만)','거래량','시가']]
ak.columns = ['close','high','low','price(million)', 'volume','start'] # 열(columns)의 이름변경

ak = ak.loc[::-1]

xk = ak.iloc[424 : 1089,:]

xk = xk.dropna(axis=0).values

def split_x(seq,size,col):
    aaa=[]
    for i in range(len(seq) - size + 1):
        subset = seq[i:(i+size),0:col].astype('float32')
        aaa.append(np.array(subset))
    return np.array(aaa)

size = 6
col = 6

datasetk = split_x(xk,size, col)


xk = datasetk[:-2,:7,:]
xk_pred = datasetk[-2:,:,:]



xk_train, xk_test = train_test_split(xk, train_size = 0.8, random_state=104)
xk_train, xk_val= train_test_split(xk_train,train_size = 0.8, random_state=104)

xk = xk.reshape(-1, 1)
xk_train = xk_train.reshape(-1, 1)
xk_test = xk_test.reshape(-1, 1)
xk_val = xk_val.reshape(-1,1)
xk_pred = xk_pred.reshape(1, -1)

scaler = MinMaxScaler()
scaler.fit(xk_train)
xk_train = scaler.transform(xk_train)
xk_test = scaler.transform(xk_test)
xk_val = scaler.transform(xk_val)
xk_pred = scaler.transform(xk_pred)


xk = xk.reshape(-1, 6, 6)
xk_train = xk_train.reshape(-1, 6, 6)
xk_test = xk_test.reshape(-1, 6, 6)
xk_val = xk_val.reshape(-1, 6 ,6)
xk_pred = xk_pred.reshape(-1, 6 ,6)

print(x.shape)
print(y.shape)
print(x_pred.shape)
print(x_train.shape) 
print(x_test.shape)

print('=======================')

print(xk.shape)
print(xk_pred.shape)
print(xk_train.shape) 
print(xk_test.shape)



input1 = Input(shape=(6,6))
dense1 = GRU(800, activation='relu')(input1)
dense1 = Dropout(0.4)(dense1)
dense1 = Dense(1600, activation='relu')(dense1)
dense1 = Dropout(0.4)(dense1)
dense1 = Dense(512, activation='relu')(dense1)
dense1 = Dropout(0.4)(dense1)

input2 = Input(shape=(6,6))
dense2 = GRU(800, activation='relu')(input2)
dense2 = Dropout(0.4)(dense2)
dense2 = Dense(1600, activation='relu')(dense2)
dense2 = Dropout(0.4)(dense2)
dense2 = Dense(512, activation='relu')(dense2)
dense2 = Dropout(0.4)(dense2)

merge1 = concatenate([dense1, dense2])
middlel1 = Dense(512,activation='relu')(merge1)
middlel1 = Dropout(0.2)(middlel1)
output1 = Dense(1)(middlel1)


# 모델 선언
model = Model(inputs=[input1, input2], outputs=output1)
model.summary()

#3
modelpath = '../../data/modelCheckPoint/samsung_dataset_{epoch:02d}-{val_loss:.8f}.hdf5'
mc = ModelCheckpoint(filepath=modelpath,monitor='val_loss',save_best_only=True,mode='auto')
early_stopping = EarlyStopping(monitor='val_loss',patience=70,mode='auto')
model.compile(loss='mse',optimizer='adam',metrics='mae')
model.fit([x_train,xk_train],y_train,epochs=1000,batch_size=8,validation_data=([x_val,xk_val],y_val),verbose=1,callbacks=[early_stopping,mc])


#4
loss = model.evaluate([x_test,xk_test],y_test,batch_size=16)
print(loss)


y_pred = model.predict([x_pred,xk_pred])
print(y_pred)