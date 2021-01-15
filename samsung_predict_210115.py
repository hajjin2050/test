import numpy as np
import pandas as pd

#1. 데이터 
df = pd.read_csv('../../data/csv/samsung.csv', index_col=0, header=0, encoding='cp949', thousands=',') 
df2 = df.iloc[::-1].reset_index(drop=True)
df2 = df2.where(pd.notnull(df2), df2.mean(), axis='columns')     # 결측치(비어있는 데이터)에  평균으로 데이터값 집어넣기

dfadd = pd.read_csv('../../data/csv/samsung_2.csv', index_col=0, header=0, encoding='cp949', thousands=',') 
dfadd = dfadd.iloc[::-1].reset_index(drop=True)
dfadd_data = dfadd.iloc[58:, [0,1,2,3,5,6,8,9]]
df20210114 = pd.concat([df2, dfadd_data]).reset_index(drop=True)



# x, y 데이터 지정
# 액면분할 시점 : 1740
x = df20210114.iloc[1800:2399, [0,1,2,3,5,6]]
y = df20210114.iloc[1801:2400, 3]
x_pred = df20210114.iloc[2399, [0,1,2,3,5,6]]


#train , val 데이터 쪼개기
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

# 저장
# np.save('../../data/npy/samsung_2_totaldata.npy',arr=([x_train, y_train, x_val, y_val, x_test, y_test,x_pred]))

#2. 모델
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Dropout, Conv1D, Flatten, MaxPooling1D, LSTM, GRU, LeakyReLU

model = Sequential()
model.add(Conv1D(filters = 1000, kernel_size = 7, strides=1, padding = 'same', input_shape = (x_train.shape[1], 1), activation='relu'))
model.add(MaxPooling1D(pool_size=1))
model.add(Conv1D(800, 1, activation='relu'))
model.add(Conv1D(500, 1, activation='relu'))
model.add(Conv1D(400, 1, activation='relu'))
model.add(Conv1D(300, 1, activation='relu'))
model.add(Conv1D(250, 1, activation='relu'))
model.add(Flatten())
model.add(Dense(200))
model.add(Dense(150))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(30))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(5))
model.add(Dense(3))
model.add(Dense(1))



#3. 컴파일
model.compile(loss='mae', optimizer='adam', metrics=['mae'])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
stop = EarlyStopping(monitor='val_loss', patience=80, mode='auto')

modelpath = '../../data/modelcheckpoint/samsung2_{epoch:02d}-{val_loss:08f}.hdf5'
check = ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, mode='auto')

hist = model.fit(x_train, y_train, epochs=1000, batch_size=16, validation_data=(x_val, y_val), verbose=1, callbacks=[stop, check])


#4. 평가
result = model.evaluate(x_test, y_test, batch_size=16)
# print('mse: ', result[0])
print('mae: ', result[0])

y_pred = model.predict(x_pred)
print('1/15일 종가: ', y_pred)

# mae:  875.7526245117188
# 1/15일 종가:  [[89487.]]