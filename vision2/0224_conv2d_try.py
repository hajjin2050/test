import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, BatchNormalization, Flatten
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
import string
import pandas as pd
import cv2

def model():
    model = Sequential()

    model.add(Conv2D(16,(3,3),activation='relu',input_shape=(128,128,1),padding='same'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    
    model.add(Conv2D(32,(3,3),activation='relu',padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(32,(5,5),activation='relu',padding='same')) 
    model.add(BatchNormalization())
    model.add(Conv2D(32,(5,5),activation='relu',padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(32,(5,5),activation='relu',padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((3,3)))
    model.add(Dropout(0.3))
    
    model.add(Conv2D(64,(3,3),activation='relu',padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(64,(5,5),activation='relu',padding='same')) 
    model.add(BatchNormalization())
    model.add(MaxPooling2D((3,3)))
    model.add(Dropout(0.3))
    
    model.add(Flatten())

    model.add(Dense(128,activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(64,activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(Dense(1,activation='sigmoid'))
    return model

#1. 데이터
x_train = np.load('C:/data/dacon/mnist2/copy/npy/train_data.npy')
x_train = x_train[:500]
x_test = np.load('../dacon/npy/x_test_128.npy')
answer = pd.read_csv('../dacon/dirty_mnist_answer.csv')
sub = pd.read_csv('../dacon/sample_submission.csv', header = 0)

# print(x_train.shape) (50000, 128, 128, 1)
# print(np.max(x_train[0])) # 1.0

alphabets = string.ascii_lowercase
alphabets = list(alphabets)

alphabet = 'a'
y_train = answer.loc[:,alphabet].to_numpy()
y_train = y_train[:500]

#2. 모델
model = model()

#3. 컴파일 훈련
es = EarlyStopping(patience = 20)
lr = ReduceLROnPlateau(patience = 10, factor = 0.25, verbose = 1)
cp = ModelCheckpoint(f'../dacon/mcp/{alphabet}.hdf5')
model.compile(loss = 'binary_crossentropy', optimizer= 'adam', metrics = ['acc'])
model.fit(x_train, y_train, epochs = 1, batch_size = 128, validation_split = 0.2, callbacks = [es,cp,lr])

#4. 평가 예측
y_pred = model.predict(x_test)
y_pred = np.where(y_pred<0.5,0,1)
sub.loc[:,alphabet] = y_pred

sub.to_csv('../dacon/submission_003.csv', index = 0)