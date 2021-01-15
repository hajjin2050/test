import numpy as np
import pandas as pd

#1. 데이터 불러옴
df = pd.read_csv('../../data/csv/samsung.csv', index_col=0, header=0, encoding='cp949', thousands=',') 
print(df)

# 데이터 순서 역으로
df2 = df.iloc[::-1].reset_index(drop=True)
print(df2)  #[2400 rows x 14 columns]

# 상관계수 확인

print(df2.corr())
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale=0.6, font='Malgun Gothic', rc={'axes.unicode_minus':False})
sns.heatmap(data=df.corr(), square=True, annot=True, cbar=True) 
plt.show()
#사용 할 칼럼 : 시가0 / 고가1 / 저가2 / 종가3 / 등락률4 / 금액6 / 신용비7 / 개인8 / 기관9

# x, y 데이터 지정
x = df2.iloc[1740:2399, [0,1,2,3,4,6,8]]
y = df2.iloc[1741:2400, 3]
x_pred = df2.iloc[2399, [0,1,2,3,4,6,8]]

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
model.add(LSTM(1000, input_shape = (7,1), activation='relu'))
model.add(Dense(500, activation='relu'))
model.add(Dense(400, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(25, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1))

# model.summary()

#3. 컴파일, 핏
model.compile(loss='mse', optimizer='adam', metrics=['mae'])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
stop = EarlyStopping(monitor='val_loss', patience=16, mode='min')

modelpath = '../data/modelcheckpoint/samsung_{epoch:02d}-{val_loss:.4f}.hdf5'
mc = ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, mode='auto')

model.fit(x_train, y_train, epochs=1000, batch_size=32, validation_data=(x_val, y_val), verbose=1, callbacks=[stop,mc])

#4. 평가, 예측
result = model.evaluate(x_test, y_test, batch_size=32)
print('mse: ', result[0])
print('mae: ', result[1])

y_pred = model.predict(x_pred)
print('1/14일 종가: ', y_pred)

# 1/14일 종가:  [[90845.27]]