import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential,Model, load_model
from tensorflow.keras.layers import Dense,LSTM,Conv1D,MaxPool1D,Conv2D,MaxPooling2D,Reshape, Flatten,Input,Dropout
from pandas.io.parsers import read_csv

df = np.load('../../data/dacon/npy/train.npy')
print(df.shape)

data = df.reshape(1095, 48, 9) #=> 하루데이터 48행 하루치로 변환(3차원)
print(df.shape)

def split_train(dataset,time_steps, y_column):
    x,y = list(), list() 
    for i in range(len(dataset)):
        x_end_number = i + time_steps
        y_end_number = x_end_number + y_column

        if y_end_number > len(dataset):
            break
        tmp_x = dataset[i:x_end_number] 
        tmp_y = dataset[x_end_number:y_end_number]
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)
time_steps=7
y_column=2
x,y = split_train(data , time_steps, y_column) #7일치데이터로 다음날 2일치 예측
print(x.shape) # (1, 7, 9)
print(y.shape)  #(1, 2, 9)

#사용할 컬럼지정
x = x[:,:,:,3:] 
y = y[:,:,:,3:]   #=> DHI(4번째컬럼)~ 끝까지만 사용
print(x.shape) #(1087, 7, 48, 6) 
print(y.shape) #(1087, 2, 48, 6)
# y데이터 처리
y = y.reshape(1087, 2*48*6)
#train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

#3. 컴파일, 훈련
model = load_model('../../data/dacon/h5/dacon_0118.h5')

# 4. 평가, 예측
loss, mae = model.evaluate(x_test, y_test, batch_size=64)

data_0118 = read_csv('../../data/dacon/test/0.csv', index_col=None, header=0)
data_0118 = df.values
print(data_0118.shape)

np.save('../../data/dacon/train/train.npy', arr=data_0118)
y_predict = model.predict(x_test)