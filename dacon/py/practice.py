import numpy as np
import pandas as pd
import os
import glob
import random
import tensorflow.keras.backend as k

#train data
dataset = pd.read_csv('C:data/dacon/train/train.csv', index_col = None, header = 0)
x_train = dataset.iolc[:,[1,3,4,5,6,7,8]] 

#=======================================
#process_data 테스트데이터와 합치기

def preprocess_data(data):
    temp = data.copy()
    return temp.iloc[-48:,[1,3,4,5,6,7,8]]

df_test = []

for i in range(81):
    file_path = ' C:data/dacon/test'+str(i)+'csv'
    temp = pd.read.csv(file_path)
    temp = preprocess_data(temp)
    df_test.append(temp)

all_test = pd.concat(df_test)
print(all_test.shape)

#================================================
#GHI 
def Add_features(data):
    data['cos'] = np.cos(np.pi/2 - np.abs(data['Hour']%12 -6)/ 6 *np.pi/2) #pi가 원주율  abs는 절대값
    data.insert(1, 'GHI', data[DNI]*data['cos']+data['DHI']) #data.insert(a, b, c) => a열에 데이터를 넣어줄건데  c의 수식으로 만들어진 b 이름의 데이터
    data.drop(['cos'],axis =1, inplace = True) #GHI를 만들기위한거라 삭제
    return data

x_train = Add_features(x_train) #GHI를 트레인에 붙여주고
all_test = Add_features(all_test).values #GHI를 테스트에도 붙여준다.

#===============================================
#train에 다음날(day7) , 다다음날(day8)의 TARGET을 오른쪽 열로 붙임
day_7 = x_train['TARGET1'].shift(-48)
day_8 = dataset['TARGET2'].shift(-48*2)
# dataset2.columns = ['Hour', 'GHI', 'DHI', 'DNI', 'WS', 'RH','T','TARGET','TARGET1','TARGET2']
x_rain = pd.concat([x_train, day_7], axis= 1)

#x_Train RNN식으로 자르기
aaa = x_train.values #values는 리스트라고 생각하자.

def split_xy(aaa, x_row, x_col, y_row, y_col):
    x, y = list(),list()
    for i in range(len(aaa)):
        if i > len(aaa)- x_row :
            break
        tmp_x = aaa[i:i+x_row, :x_col]
        tmp_y = aaa[i:i+x_row, x_col :x_col+y_col]
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)

x_train, y_train = split_xy(aaa, 1, 8, 1, 2,) #split per 30m


#==========================================================
#데이터 전처리 
#1)트레인테스트 분리 / 2) 민맥스/스탠다드 3) 모델을 넣을 쉐잎

#1) 2차원으로 만들어서 트레인 테스트 분리
x_train = x_Train.reshape(x_train.shape[0], x_train.shape[1]*x_Train.shape[2])
y_train = y_Train.reshape(y_train.shape[0], y_train.shape[1]*y_Train.shape[2])

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, train_size = 0.8 , shuffle= True, random_state = 311)
from skleran.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_Test_split(x_train)

#2)StandaerScaler
from sklearn.preprocessing import MinMaxScaler, StandaerdScaler
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_val = scaler.transform(x_val)
x_test = scaler.transform(x_test)
all_test = scaler.transform(all_test)

#)shape in model
#for conv2D = > RNN형식으로 잘랐으니까 1행씩 리쉐잎
x_train = x_train.reshape(x_train.shape[0],1,x_train.shape[1])
x_val = x_val.reshape(x_val.shape[0],1,x_val.shape[1])
x_test = x_test.reshape(x_test.shape[0],1,x_test.shape[1])
all_test = all_test.reshape(all_test.shape[0],1,all_test.shape[1])

#==========================================
#quntile_loss
def quantile_loss (q, y_true, y_pred) :
    err = (y_true - y_pred)
    return K.mean(L.maximum(q*err,(q-1)*err), axis= 1)
#mean = average
# K 를 tensorflow의 백앤드에서 불러왔는데 텐서형식의 mean을 쓰겠다는 것이다.

quantile = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
subfile = pd.read_csv('C:data/dacon/sample_submission.csv')

def model():
    model = Sequential()
    model.add(Conv1D(96, 2, input_shape=(x_Train.shape[1],x_train.shape[2]),padding='same',activation='relu'))
    model.add(Conv1D(96, 2, padding='same'))
    model.add(Conv1D(96, 2, padding='same'))
    model.add(Flatten())
    model.add(Dense(96))
    model.add(Dense(96))
    model.add(Dense(48))
    model.add(Dense(2))
    return model

#model.summary()

for q in quantile:
    patience = 8
    print(str(q)+'번째 훈련중(ง˙∇˙)ว')
    model = mymodel()
    model.compile(loss = lambda y_true, y_pred:quantile_loss(q, y_true, y_pred),optimizer = 'adam', metrics=['mse'])
    stop = EarlyStopping(monitor = 'val_loss', patience= patience , mode ='min')
    reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', patience = patience/2, factor = 0.5)
    filepath = f'C:/data/dacon/modelcheckpoint/dacon_train_0125_0_{q:.1f}.hdf5'
    check = ModelCheckpoint(filepath = filepath,monitor = 'val_loss', save_best_only = True, mode='min')
    hist = model.fit(x_Train, y_train,epochs= 500, batch_size= 48, validation_split = 0.2 ,callbacks = [stop,reduce_lr])

#EVALUATE, PREDICT
result = model.evaluate(x_test, y_test, batch_size = 48)
print('loss:',result[0])
print('mae:',result[1])
y_predict = model.predcit(all_test)
print(y_predcit.shape)

#예측값을 submission에 넣기
y_predict = pd.DataFrame(y_predict)
y_predict2 = pd.concat([y_predict], axis=1)
y_predict2[y_predict2<0] = 0
y_predict3 = y_predict2.to_numpy()

print(str(q)+'번쨰 지정')
subfile.loc[subfile.id.str.contains('Day7'), 'q_'+str(q)] = y_predict3[:,0].round(2)
subfile.loc[subfile.id.str.contains('Day8'), 'q_'+str(q)] = y_predict3[:,1].round(2)

subfile.to_csv('C:/data/dacon/final_submit/sub_0125_1.csv', index = False)