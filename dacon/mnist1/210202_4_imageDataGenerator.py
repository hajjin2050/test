import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import string
from sklearn.model_selection import train_test_split, KFold
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def mymodel():
    model = Sequential()
    model.add(Conv2D(32, 3, padding = 'same', activation = 'relu', input_shape = (28,28,1)))
    model.add(MaxPooling2D(2))
    model.add(Conv2D(32, 3, padding = 'same', activation = 'relu'))
    model.add(Conv2D(32, 3, padding = 'same', activation = 'relu'))
    model.add(Conv2D(32, 3, padding = 'same', activation = 'relu'))
    model.add(Conv2D(32, 3, padding = 'same', activation = 'relu'))
    model.add(Conv2D(32, 3, padding = 'same', activation = 'relu'))
    model.add(Flatten())
    model.add(Dense(128, activation = 'relu'))
    model.add(Dense(64, activation = 'relu'))
    model.add(Dense(32, activation = 'relu'))
    model.add(Dense(16, activation = 'relu'))
    model.add(Dense(10, activation = 'softmax'))
    return model

train = pd.read_csv('C:/data/dacon/mnist1/data/train.csv')
test = pd.read_csv('C:/data/dacon/mnist1/data/test.csv')
submission = pd.read_csv('C:/data/dacon/mnist1/data/submission.csv', header = 0)
submission['tmp'] = test['letter']

alphabets = string.ascii_uppercase # 대문자가져오기 
alphabets = list(alphabets)

kfold = KFold(n_splits=7, random_state=33, shuffle=True)
datagen = ImageDataGenerator(rescale = 1./255, rotation_range= 20, width_shift_range= 0.1, height_shift_range= 0.1)
datagen2 = ImageDataGenerator()
es = EarlyStopping(monitor = 'val_loss', patience = 30)
lr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.25, patience = 10)


for alphabet in alphabets:
    print(f'{alphabet} 진행중...')
    x1_train = train.loc[train['letter'] == alphabet, '0':].to_numpy().reshape(-1,28,28,1)
    y1_train = train.loc[train['letter'] == alphabet, 'digit'].to_numpy()
    y1_train = to_categorical(y1_train)
    x_test = test.loc[test['letter'] == alphabet, '0':].to_numpy().reshape(-1,28,28,1)

    cp = ModelCheckpoint(monitor= 'val_loss', filepath=f'C:/data/dacon/mnist1/modelcheck/{alphabet}.h5', save_best_only= True)
'''
    result = []
    for train_index, valid_index in kfold.split(x1_train, y1_train) :
        x_train = x1_train[train_index]
        x_val = x1_train[valid_index]    
        y_train = y1_train[train_index]
        y_val = y1_train[valid_index]

        train_generator = datagen.flow(x_train,y_train,batch_size=8)
        val_generator = datagen2.flow(x_val,y_val)

        model = mymodel()
        model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['acc'])
        model.fit_generator(train_generator, epochs = 1000, validation_data= val_generator, callbacks = [es,cp,lr])

        model.load_weights(f'C:/data/dacon/mnist1/modelcheck/{alphabet}.h5')
        y_pred = np.argmax(model.predict(x_test), axis = 1)
        result.append(y_pred)
    result = np.array(result)
    mode = stats.mode(result).mode
    mode = np.transpose(mode)
    submission.loc[submission['tmp'] == alphabet, 'digit'] = mode

submission.drop('tmp', axis = 1,inplace=True)
submission.to_csv('C:/data/dacon/mnist1/data/submission.csv', index = 0)