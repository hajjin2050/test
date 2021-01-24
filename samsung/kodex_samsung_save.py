import numpy as np
import pandas as pd

x1 = np.load('../../data/npy/samsung3.npy',allow_pickle=True)
y1 = np.load('../../data/npy/samsung3.npy',allow_pickle=True)
x1_pred = np.load('../../data/npy/samsung3.npy',allow_pickle=True)

x2 = np.load('../../data/npy/KODEX.npy',allow_pickle=True)
# print(x1.shape) (80, 7)
# print(x2.shape) (1088, 7)
# print(y1.shape) (80, 7)

x1 = np.transpose(x1)
x1_pred = np.transpose(x1_pred)
x2 = np.transpose(x2)
y1 = np.transpose(y1)


size = 6

def split_x1(seq, size):
    aaa = []
    for i in range(len(seq)-size+1):
        subset = seq[i : (i+size)]
        aaa.append([item for item in subset])
    print(type(aaa))
    return np.array(aaa)
x1_data = split_x1(x1, size)
print(x1.shape)

def split_x2(seq, size):
    aaa = []
    for i in range(len(seq)-size+1):
        subset = seq[i : (i+size)]
        aaa.append([item for item in subset])
    print(type(aaa))
    return np.array(aaa)
x2_data = split_x2(x1, size)
print(x2.shape)

from sklearn.model_selection import train_test_split
x1_train, x1_test,x2_train, x2_test, y1_train, y1_test = train_test_split(
    x1, x2, y1, shuffle=False, train_size = 0.8
)
from sklearn.model_selection import train_test_split
x1_train, x1_val,x2_train, x2_val, y1_train, y1_val = train_test_split(
    x1, x2, y1, shuffle=False, train_size = 0.8
)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x1_train)
x1_train = scaler.transform(x1_train)
x1_val = scaler.transform(x1_val)
x1_test = scaler.transform(x1_test)
x1_pred = scaler.transform(x1_pred)
print(x1_pred.shape)


x1_train = x1_train.reshape(x1_train.shape[0]* x1_train.shape[1], 1)
x1_val = x1_val.reshape(x1_val.shape[0]* x1_val.shape[1], 1)
x1_test = x1_test.reshape(x1_test.shape[0]* x1_test.shape[1], 1)
x1_pred = x1_pred.reshape(x1_pred.shape[0]* x1_pred.shape[1], 1)

scaler.fit(x2_train)
x2_train = scaler.transform(x2_train)
x2_val = scaler.transform(x2_val)
x2_test = scaler.transform(x2_test)


x2_train = x2_train.reshape(x2_train.shape[0]* x2_train.shape[1], 1)
x2_val = x2_val.reshape(x2_val.shape[0],x2_val.shape[1], 1)
x2_test = x2_test.reshape(x2_test.shape[0],x2_test.shape[1], 1)



#2.모델구성
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense, Input, Dropout, Conv1D, Flatten, MaxPooling1D, LSTM, GRU, LeakyReL

#모델1
input1 = Sequential()
input1.add = (Conv1D(filters = 1000, kernel_size = 7, strides=1, padding = 'same', input_shape = (x1_train.shape[1], 1), activation='relu'))
input1.add(MaxPooling1D(pool_size=1))
input1.add(Conv1D(400, 1, activation='relu'))
input1.add(Conv1D(300, 1, activation='relu'))
input1.add(Conv1D(250, 1, activation='relu'))
input1.add(Flatten())
dense1 = Dense(10, activation='relu')(input1)
dense1 = Dense(5, activation='relu')(dense1)
#output1 = Dense(3)(dense1)

#모델2
input2 = Input(shape=(x2_train.shape[0]*x2_train.shape[1],))
dense2 = Dense(10, activation='relu')(input2)
dense2 = Dense(5, activation='relu')(dense2)
dense2 = Dense(5, activation='relu')(dense2)
dense2 = Dense(5, activation='relu')(dense2)
#output2 = Dense(3)(dense2)

#모델변형 / concatenate
from tensorflow.keras.layers import concatenate, Concatenate
from keras.layers import concatenate, Concatenate
merge1 = concatenate([dense1, dense2])
middle1 = Dense(30)(merge1)
middle1 = Dense(10)(middle1)
middle1 = Dense(10)(middle1)

#모델 분기1
output1 = Dense(30)(middle1)
output1 = Dense(7)(output1)
output1 = Dense(1)(output1)

#모델 선언
model = Model(inputs=[input1, input2], 
              outputs=[output1])
model.summary()

#3.컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(([x1_train, x2_train],y1_train), epochs=100, batch_size=8,
          validation_split=0.2, verbose=1)

#4.평가 , 예측
result = model.evaluate([x1_test, x2_test], batch_size=8)
print(result.shape[0])
print(result.shape[1])
