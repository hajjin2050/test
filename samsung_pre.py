import numpy as np
import pandas as pd

#1. 데이터 불러옴
df1 = pd.read_csv('../../data/csv/samsung.csv', index_col=0, header=0, encoding='cp949', thousands=',') 

df2 = pd.read_csv('../../data/csv/samsung_2.csv', index_col=0, header=0, encoding='cp949', thousands=',')
df2 = df2.dropna()
df2 = df2.drop(['전일비','Unnamed: 6'], axis=1)
# 데이터 순서 역으로
df1 = df1.iloc[::-1].reset_index(drop=True)
print(df1)  #[2400 rows x 14 columns] 

# 데이터 병합
df = pd.concat([df1, df2], axis=0)
df = df.drop(['2021-01-13'])  #=> 중복데이터제거
print(df.shape)
print(df.tail())


'''
# 상관계수 확인

print(df1.corr())
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale=0.6, font='Malgun Gothic', rc={'axes.unicode_minus':False})
sns.heatmap(data=df.corr(), square=True, annot=True, cbar=True) 
plt.show()
#사용 할 칼럼 : 시가0 / 고가1 / 저가2 / 종가3 / 등락률4 / 금액6 / 신용비7 / 개인8 / 기관9
'''
# x, y 데이터 지정
x = df.iloc[1740:2401, [0,1,2,3,4,6,8]]
y = df.iloc[1741:2402, 3]
x_pred = df1.iloc[2401, [0,1,2,3,4,6,8]]
print(x.shape)       #(659, 7)
print(y.shape)       #(659,)
print(x_pred.shape)  #(7,)    

'''
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

print(x_train.shape[0])
print(x_train.shape[1])

np.save('../data/npy/samsung_x_train.npy', arr=x_train)
np.save('../data/npy/samsung_y_train.npy', arr=y_train)
np.save('../data/npy/samsung_x_val.npy', arr=x_val)
np.save('../data/npy/samsung_y_val.npy', arr=y_val)
np.save('../data/npy/samsung_x_test.npy', arr=x_test)
np.save('../data/npy/samsung_y_test.npy', arr=y_test)
np.save('../data/npy/samsung_x_pred.npy', arr=x_pred)

#2. 모델구성
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Dropout, Conv1D, Flatten, MaxPooling1D, LSTM


model = Sequential()
model.add(Conv1D(filters = 1000, kernel_size = 7, strides=1, padding = 'same', input_shape = (7,1), activation='relu'))
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

# model.summary()

#3. 컴파일, 핏
model.compile(loss='mse', optimizer='adam', metrics=['mae'])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
stop = EarlyStopping(monitor='val_loss', patience=16, mode='min')

modelpath = '../data/modelcheckpoint/samsung_{epoch:02d}-{val_loss:.4f}.hdf5'
mc = ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, mode='auto')

model.fit(x_train, y_train, epochs=800, batch_size=32, validation_data=(x_val, y_val), verbose=1, callbacks=[stop,mc])

#4. 평가, 예측
result = model.evaluate(x_test, y_test, batch_size=32)
print('mse: ', result[0])
print('mae: ', result[1])

y_pred = model.predict(x_pred)
print('1/14일 종가: ', y_pred)

# 1/14일 종가:  [[90845.27]]
'''