import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, LSTM, Conv1D, Input, Flatten, MaxPooling1D, Dropout, Reshape, SimpleRNN, LSTM, LeakyReLU, GRU, Conv2D, MaxPool2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.backend import mean, maximum
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import os
import glob
import random
import tensorflow.keras.backend as K
#데이터 불러오기
dataset = pd.read_csv("C:/data/dacon/train/train.csv", index_col=None, header=0)
x_train = dataset.iloc[:,[1,3,4,5,6,7,8]]

def preprocess_data(data):
    temp = data.copy()
    return temp.iloc[-48:,[1,3,4,5,6,7,8]]

df_test = []

for i in range(81):
    file_path = 'C:/data/dacon/test/'+str(i)+'.csv'
    temp = pd.read_csv(file_path)
    temp = preprocess_data(temp)
    df_test.append(temp)

x_test = pd.concat(df_test)
# print(x_test.shape)(3888, 9)

def Add_features(data):
    data['cos'] = np.cos(np.pi/2 - np.abs(data['Hour'] % 12 - 6)/6 * np.pi/2)
    # pi = 원주율, abs = 절대값
    data.insert(1, 'GHI', data['DNI'] * data['cos'] + data['DHI'])
    # 데이터를 넣어줄건데 1열에(기존 열은 오른쪽으로 밀림), 'GHI'명으로, 마지막의 수식으로 나온 값을
    data.drop(['cos'], axis=1, inplace = True)
    #'cos' 열을 삭제를 할 것 이고. 이 삭제한 데이터프레임으로 기존 것을 대체하겠다.
    return data
#=======================================================================
x_train = Add_features(x_train)
df_test = Add_features(x_test).values
# print(x_train.shape)(52560, 7)
# print(x_test.shape)(3888, 7)
#===========타겟값설정("TARGET"열로 다음날 데이터와 다다음날데이터 지정)===========================================================
day_7 = x_train['TARGET'].shift(-48)
day_8 = dataset['TARGET'].shift(-48*2) # 왜 위에는 트레인에서 자르고 아래는 데이터셋에서? 같은거아닌가?

x_train = pd.concat([x_train, day_7, day_8], axis= 1)
x_train = x_train.iloc[:-96,:]

print(x_train.shape)#(52464, 9)
print(df_test.shape)#(3888, 7)
#======train자르기==========================================================

aaa = x_train.values
#values : x_train의 값을 나열한다고 생각하자

def split_xy(aaa,x_row,x_col,y1_row, y1_col, y2_col):
    x,y1,y2 = list(), list(),list()
    for i in range(len(aaa)):
        if i > len(aaa)- x_row: #i가aaa안에 있지만x_row를 더했을때 많으면 중단
            break
        tmp_x = aaa[i:i+x_row, :x_col]
        tmp_y1 = aaa[i+x_row-y1_row:i+x_row, x_col:x_col+y1_col]
        tmp_y2 = aaa[i+x_row-y1_row:i+x_row, x_col+y1_col:x_col+y1_col+y2_col]
        x.append(tmp_x)
        y1.append(tmp_y1)
        y2.append(tmp_y2)
    return np.array(x), np.array(y1), np.array(y2)

x_train, y1_train, y2_train = split_xy(aaa, 48,8,48,1,1)
print(x_train.shape)  
print(y1_train.shape)  
print(y2_train.shape)  

df_test = df_test.reshape(int(df_test.shape[0]/48), 48, df_test.shape[1])
df_test = df_test.reshape(df_test.shape[0], df_test.shape[1]*df_test.shape[2])
#=====트레인 분리를 위한 리솊===============================
x_train = x_train.reshape(x_train.shape[0],x_train.shape[1]*x_train.shape[2])
y1_train = y1_train.reshape(y1_train.shape[0],y1_train.shape[1]*y1_train.shape[2])
y2_train = y2_train.reshape(y2_train.shape[0],y2_train.shape[1]*y2_train.shape[2])
#=====트레인 분리=============Y는 Y끼리 X는 X끼리 묶어준다.==================
from sklearn.model_selection import train_test_split
y1_train, y1_test, y2_train, y2_test = train_test_split(y1_train, y2_train, train_size=0.8, shuffle=True, random_state=311)
from sklearn.model_selection import train_test_split
y1_train, y1_val, y2_train, y2_val = train_test_split(y1_train, y2_train, train_size=0.8, shuffle=True, random_state=311)
from sklearn.model_selection import train_test_split
x_train, x_test = train_test_split(x_train, train_size=0.8, shuffle=True, random_state=311)
from sklearn.model_selection import train_test_split
x_train, x_val = train_test_split(x_train, train_size=0.8, shuffle=True, random_state=311)
#StandardScaler
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_val = scaler.transform(x_val)
x_test = scaler.transform(x_test)
df_test = scaler.transform(df_test)


#=====mean= average/ K?==========================================================
#quntile_loss
def quantile_loss (q, y_true, y_pred) :
    err = (y_true - y_pred)
    return K.mean(K.maximum(q*err,(q-1)*err), axis=-1)
#===MODELING=======================================================
quantile = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
subfile = pd.read_csv('C:/data/dacon/sample_submission.csv')


def mymodel():#reshape을 안해줘서 안맞는거같다 수정해보자
    model = Sequential()
    model.add(Dense(128, activation = 'relu'))
    model.add(Dense(64, activation = 'relu'))
    model.add(Dense(32, activation = 'relu'))
    model.add(Dense(16, activation = 'relu'))
    model.add(Dense(8, activation = 'relu'))
    model.add(Dense(1))

loss_y1 = []
loss_y2 = []

from tensorflow.keras.optimizers import Adam, Adadelta, Adamax, Adagrad
from tensorflow.keras.optimizers import RMSprop, SGD, Nadam
# model.summary()
#AttributeError: 'function' object has no attribute 'summary' =>왜지?
#y1
for q in quantile :
    patience = 8
    print(str(q)+'번쨰 훈련중(y1)ว(˙∇˙)ว(ง˙∇˙)ว')
    model = mymodel()
    optimizer = Adam(lr=0.002)
    es = EarlyStopping(monitor = 'val_loss',patience = patience, mode='min')
    lr = ReduceLROnPlateau(monitor='val_loss', patience = patience/2, factor = 0.2)
    model.compile(loss = lambda y1_true, y1_pred : quantile_loss (q, y1_true, y1_pred), optimizer = optimizer ,metrics=['mae'])
    filepath =f'C:/data/dacon/modelcheckpoint/dacon_0126_2_{i:2d}_y1seq_{j:.1f}.hdf5'
    check = ModelCheckpoint(filepath = filepath,monitor = 'val_loss',save_best_only = True, mode = 'min')
    model.fit(x_train, y1_train, epochs=1000, batch_size=96, verbose=1, validation_split=0.2, callbacks=[es, lr])
#EVALUATE
result = model.evaluate(x_test, y1_test,batch_size = 16)
print('loss:',result[0])
print('mae:',result[1])
loss_lsit1.append(result)
y1_predict = model.predict(df_test)
print(y_predict.shape)
#제출파일에 넣기
y1_predict = pd.DataFrame(y_predict)
y1_predict2 = pd.concat([y_predict], axis= 1)
y1_predict2[y_predict2<0] = 0
y1_predict3 = y_predict2.to_numpy()

#y2
for q in quantile :
    patience = 8
    print(str(q)+'번쨰 훈련중(y2)ว(˙∇˙)ว(ง˙∇˙)ว')
    model = mymodel()
    optimizer = Adam(lr=0.002)
    es = EarlyStopping(monitor = 'val_loss',patience = patience, mode='min')
    lr = ReduceLROnPlateau(monitor='val_loss', patience = patience/2, factor = 0.2)
    model.compile(loss = lambda y2_true, y2_pred : quantile_loss (q, y2_true, y2_pred), optimizer = optimizer ,metrics=['mae'])
    filepath =f'C:/data/dacon/modelcheckpoint/dacon_0126_2_{i:2d}_y2seq_{j:.1f}.hdf5'
    check = ModelCheckpoint(filepath = filepath,monitor = 'val_loss',save_best_only = True, mode = 'min')
    hist = model.fit(x_train, y2_train, epochs=1000, batch_size=96, verbose=1, validation_split=0.2, callbacks=[es,lr])
#EVALUATE
result = model.evaluate(x_test, y2_test,batch_size = 16)
print('loss:',result[0])
print('mae:',result[1])
loss_lsit1.append(result)
y1_predict = model.predict(df_test)
print(y_predict.shape)
#제출파일에 넣기
y2_predict = pd.DataFrame(y_predict)
y2_predict2 = pd.concat([y_predict], axis= 1)
y2_predict2[y_predict2<0] = 0
y2_predict3 = y_predict2.to_numpy()

print(str(q)+'번째 지정')
subfile.loc[subfile.id.str.contains('Day7'),'q_'+str(q)] = y_predict3[:,0]
subfile.loc[subfile.id.str.contains('Day8'),'q_'+str(q)] = y_predict3[:,1]

loss_list1 = np.array(loss_list1)
loss_list1 = loss_list1.reshape(9,-1) #=>뒤집어서 9개의 로스값이 나오게.
loss_list2 = np.array(loss_list2)
loss_list2 = loss_list2.reshape(9,-1)
print('loss1 : \n', loss_list1)
print('loss2 : \n', loss_list2)

subfile.to_csv('C:/data/dacon/final_submit/sub_0125_3.csv', index =  False)