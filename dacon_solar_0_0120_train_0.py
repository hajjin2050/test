import numpy as np
import pandas as pd

dataset = pd.read_csv('C:/data/dacon/practice/train/train.csv', index_col = None, header=0)
# print(dataset.shape) #(52560, 9)
dataset = dataset.iloc[:,[1,3,4,5,6,7,8]]
# print(dataset.shape)#(52560, 7)
# print(dataset.info())

#다음날, 다다음날의 TARGET을 오른쪽 옆으로 붙임
df1 = dataset['TARGET'].shift(-48)
df2 = dataset['TARGET'].shift(-48*2)

dataset2 = pd.concat([dataset, df1,df2], axis=1)
dataset2.columns = ['Hour','DHI','DNI','WS','RH','T','TARGET','TARGET+1','TARGET+2']
dataset2 = dataset2.iloc[:-96,:]

# print(dataset2.info())
print(dataset2.shape)

aaa = dataset2.values
# print(len(aaa)) 52560

def split_xy(aaa, x_row, x_col, y_row, y_col):
    x, y = list(),list()
    for i in range(len(aaa)):
        if i > len(aaa)-x_row:
            break
        tmp_x = aaa[i:i+x_row, :x_col]
        tmp_y = aaa[i:i+y_row, x_col : x_col + y_col]
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)
# print(x , '\n\n', y)
x, y = split_xy(aaa, 336, 7 ,336, 2)
# print(x.shape) (52225, 336, 7)
# print(y.shape) (52225, 336, 2)
# print(y[1,:,:].shape) (336, 2)

#전처리
# 2차원으로 만들어서 트레인분리
x = x.reshape(x.shape[0], x.shape[1]*x.shape[2])
y = y.reshape(y.shape[0], y.shape[1]*y.shape[2])

from sklearn.model_selection import train_test_split
x_train, x_test, y_train,y_test = train_test_split(x, y, train_size=0.8,shuffle=True, random_state=311)
x_train, x_val, y_train,y_val = train_test_split(x_train, y_train, train_size=0.8,shuffle=True, random_state=311)

# 2)MinMax
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_val = scaler.transform(x_val)
x_test = scaler.transform(x_test)

# print(x_train.shape)        #(33362, 2352)
# print(x_val.shape)          #(8341, 2352)
# print(x_test.shape)         #(10426, 2352)
# print(y_train.shape)        #(33362, 672)
# print(y_val.shape)          #(8341, 672)
# print(y_test.shape)         #(10426, 672)  

#3) reshape for 3
num1 = 7
num2 = 2
x_train = x_train.reshape(x_train.shape[0], int(x_train.shape[1]/num1), num1)
x_val = x_val.reshape(x_val.shape[0], int(x_val.shape[1]/num1), num1)
x_test = x_test.reshape(x_test.shape[0], int(x_test.shape[1]/num1), num1)


y_train = y_train.reshape(y_train.shape[0], int(y_train.shape[1]/num2), num2)
y_val = y_val.reshape(y_val.shape[0], int(y_val.shape[1]/num2), num2)
y_test = y_test.reshape(y_test.shape[0], int(y_test.shape[1]/num2), num2)

# print(x_train.shape)        #(33362, 336, 7)
# print(x_val.shape)          #(8341, 336, 7)
# print(x_test.shape)         #(10426, 336, 7)
# print(y_train.shape)        #(33362, 336, 2)
# print(y_val.shape)          #(8341, 336, 2)
# print(y_test.shape)         #(10426, 336, 2)

#model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM,Conv1D,MaxPooling1D, Dropout,Reshape

model = Sequential()
model.add(Conv1D(96, 7, input_shape=(x_train.shape[1],x_train.shape[2]),padding='same', activation='relu'))
model.add(Conv1D(48, 7, padding='same'))
model.add(Conv1D(48, 7, padding='same'))
model.add(Conv1D(28, 7, padding='same'))
model.add(Dense(28))
model.add(Dense(14))
model.add(Dense(7))
model.add(Dense(2))

#3.Compile
model.compile(loss='mse', optimizer='adam', metrics=['mae'])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
stop = EarlyStopping(monitor='val_loss', patience=20, mode='min')
filepath = 'C:/data/dacon/practice/mcp/dacon_train_1_{epoch:02d}-{val_loss:.4f},hdf5'
check = ModelCheckpoint(filepath = filepath, monitor = 'val_loss',svae_best_only = True, mode='min')
lr = ReduceLROnPlateau(monitor='val_loss', patience=10, factor=0.5, verbose=1)

hist = model.fit(x_train, y_train, epochs=20, batch_size=48, validation_data=(x_val, y_val), verbose=1, callbacks=[stop, lr])

#4.EVALUATAE
result = model.evaluate(x_test,y_test,batch_size=48)
print('mse:',result[0])
print('mae:',result[1])


y_predict = model.predict(x_test)
