import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Dropout, MaxPool2D

#LOAD DATA
train = pd.read_csv('C:/data/dacon/mnist1/data/train.csv')
# print(train.shape)(2048, 787)
submission = pd.read_csv('C:/data/dacon/mnist1/data/submission.csv')
# print(submission.shape)(20480, 2)
pred = pd.read_csv('C:/data/dacon/mnist1/data/test.csv')
# print(test.shape)(20480, 786)

# #EDA (베이스라인 그대로 공부 더 필요)
# idx = 318 # 임의로 지정해준 인덱스 번호
# img = train.loc[idx, '0':].values.reshape(28, 28).astype(int)
# digit = train.loc[idx, 'digit']
# letter = train.loc[idx, 'letter']

# plt.title('Index: %i, Digit: %s, Letter: %s'%(idx, digit, letter))
# plt.imshow(img)
# plt.show()

#1.DATA
# X 
x = train.drop(['id', 'digit', 'letter'], axis=1).values # test데이터와 컬럼 맞춰주기
x = x.reshape(-1, 28, 28, 1)
x = x/255
# print(x.shape)(2048, 28, 28, 1)

# Y
y_tmp = train['digit']
y = np.zeros((len(y_tmp), len(y_tmp.unique()))) # np.zeros(shape, dtype, order)  
for i, digit in enumerate(y_tmp) :
    y[i, digit] = 1

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle = True,random_state=47)

# predict
x_pred = pred.drop(['id', 'letter'], axis=1).values
x_pred = x_pred.reshape(-1, 28, 28, 1)
x_pred = x_pred/255.

#2. Modeling
def modeling() : 
    model = Sequential()
    model.add(Conv2D(640, (2, 2), padding='same', activation='relu',\
        input_shape=(x_train.shape[1], x_train.shape[2], x_train.shape[3])))
    model.add(Conv2D(640, (2, 2), padding='same', activation='relu'))
    model.add(Dropout(0.2))
    model.add(Conv2D(320, (2, 2), padding='same', activation='relu'))
    model.add(Dropout(0.2))
    model.add(Conv2D(320, (2, 2), padding='same', activation='relu'))
    model.add(Dropout(0.2))
    model.add(Conv2D(128, (2, 2), padding='same', activation='relu'))
    model.add(Dropout(0.2))
    model.add(Conv2D(128, (2, 2), padding='same', activation='relu'))
    model.add(Flatten())
    model.add(Dense(320, activation='relu'))
    model.add(Dense(160, activation='relu'))
    model.add(Dense(80, activation='relu'))
    model.add(Dense(40, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    # model.summary()
    return model

model = modeling()

#3. Compile, Train
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['acc'])

model.fit(x_train, y_train, epochs=100, batch_size=32, validation_split=0.2)

#4. Evaluate, predict
loss, acc = model.evaluate(x_test, y_test, batch_size=32)
print("loss : ", loss)
print("acc : ", acc)

# submission
submission['digit'] = np.argmax(model.predict(x_pred), axis=1)
print(submission.head())

submission.loc[:,'digit'] = pred
submission.to_csv('C:/data/dacon/mnist1/submit/210202_cnn.csv',index= False)
# loss :  3.498981475830078
# acc :  0.6902438998222351