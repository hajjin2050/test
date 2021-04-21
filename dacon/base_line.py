import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow.keras as keras
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
import seaborn as sns

from collections import Counter
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Conv2D, Flatten,MaxPooling2D,BatchNormalization,Lambda, AveragePooling2D, Dropout, SpatialDropout2D
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
train_x = pd.read_csv("C:/workspace/dacon/bitcoin/train_x_df.csv")
train_y = pd.read_csv("C:/workspace/dacon/bitcoin/train_y_df.csv")
test_x = pd.read_csv("C:/workspace/dacon/bitcoin/test_x_df.csv")
sample_submission = pd.read_csv("C:/workspace/dacon/bitcoin/sample_submission.csv")

#print(train_x.shape)(10572180, 12)
#print(train_y.shape)(919320, 12)
#print(test_x.shape)(738300, 12)

print(train_x.iloc[:,2:].shape)
print(test_x.iloc[:,2:].shape)

X_train = np.array(train_x.iloc[:,2:]).reshape(7661, 1380, len(train_x.columns)-2, 1) # > (#_sampleid, #_time, #_columns, 1)
X_test = np.array(test_x.iloc[:,2:]).reshape(535, 1380, len(test_x.columns)-2, 1) # > (#_sampleid, #_time, #_columns, 1)

price = (np.array(train_y['high']) + np.array(train_y['low']))/2 # high와 low의 중간값을 사용!
Y_train = price.reshape(7661, 120)
Y_train2 = Y_train.argmax(axis=1).astype(np.float32)
# Y_train2 = Y_train.argsort(axis=1)[:,-4:].astype(np.float32)
n=1

# X_train_slice_mean = np.zeros(7362*10).reshape(7362,1,10,1)
# for i in range(1380//n):
#     X_train_slice_mean = np.concatenate([X_train_slice_mean, X_train[:,i*n:(i+1)*n,:,:].mean(axis=1).reshape(7362,1,10,1)], axis=1)
# X_train_slice_mean = X_train_slice_mean[:,1:,:,:]

# X_test_slice_mean = np.zeros(529*10).reshape(529,1,10,1)
# for i in range(1380//n):
#     X_test_slice_mean = np.concatenate([X_test_slice_mean, X_test[:,i*n:(i+1)*n,:,:].mean(axis=1).reshape(529,1,10,1)], axis=1)
# X_test_slice_mean = X_test_slice_mean[:,1:,:,:]

# Y_train_slice_mean = np.zeros(7362).reshape(7362,1)
# for i in range(120//n):
#     Y_train_slice_mean = np.concatenate([Y_train_slice_mean, Y_train[:,i*n:(i+1)*n].mean(axis=1).reshape(7362,1)], axis=1)
# Y_train_slice_mean = Y_train_slice_mean[:,1:]

# Y_train2 = Y_train_slice_mean.argsort(axis=1)[:,-4:].astype(np.float32)

# print(X_train_slice_mean.shape)
# print(X_test_slice_mean.shape)
# print(Y_train_slice_mean.shape)
# print(Y_train2.shape)

# X_train = X_train_slice_mean.copy()
# X_test = X_test_slice_mean.copy()
def my_loss(y_true, y_pred):
    result = (y_pred-y_true)
    return K.mean(K.square(result))

def set_model(): 
    
    activation = 'relu'
    padding = 'global'
    model = Sequential()
    nf = 16
    fs = (3,1)
    model.add(Conv2D(nf,fs, padding=padding, activation=activation,input_shape=(X_train.shape[1],X_train.shape[2],X_train.shape[3]))) # 1300, 10, 1
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 1))) # 650, 10, 16

    model.add(Conv2D(nf*2,fs, padding=padding, activation=activation))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 1))) # 325, 10, 32

    model.add(Conv2D(nf*4,fs, padding=padding, activation=activation))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 1))) #  162, 10, 64

    model.add(Conv2D(nf*8,fs, padding=padding, activation=activation))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 1))) #  81, 10 ,128

    model.add(Conv2D(nf*16,fs, padding=padding, activation=activation))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 1))) #  40, 10 ,256

    model.add(Conv2D(nf*32,fs, padding=padding, activation=activation))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 1))) #  20, 10 ,512


    
    model.add(Flatten()) 
    model.add(Dense(512, activation ='elu')) 
    model.add(Dense(128, activation ='elu')) 
    model.add(Dense(32, activation ='elu')) 
    model.add(Dense(8, activation ='elu')) 
    model.add(Dense(4, activation ='elu')) 
    model.add(Dense(1))# output size 

    optimizer = keras.optimizers.Adam(learning_rate=1e-06)

    model.compile(loss=my_loss,
              optimizer=optimizer,)

    return model

def train(model, X, Y, is_val=False):
    MODEL_SAVE_FOLDER_PATH = './model/'
    if not os.path.exists(MODEL_SAVE_FOLDER_PATH):
        os.mkdir(MODEL_SAVE_FOLDER_PATH)

    model_path = MODEL_SAVE_FOLDER_PATH + '{epoch:02d}-{val_loss:.4f}.hdf5'
    best_save = ModelCheckpoint('best_m.hdf5', save_best_only=True, monitor='val_loss', mode='min')

    if is_val == False:
        history = model.fit(X, Y,
                      epochs=100,
                      batch_size=32,
                      shuffle=True,
                      validation_split=0.2,
                      verbose = 1,
                      callbacks=[best_save])

        fig, loss_ax = plt.subplots()
        acc_ax = loss_ax.twinx()

        loss_ax.plot(history.history['loss'], 'y', label='train loss')
        loss_ax.plot(history.history['val_loss'], 'r', label='val loss')
        loss_ax.set_xlabel('epoch')
        loss_ax.set_ylabel('loss')
        loss_ax.legend(loc='upper left')
        plt.show()    
        
#     else:
#         history = model.fit(X, Y,
#                       epochs=30,
#                       batch_size=256,
#                       shuffle=True,
#                       validation_split=0.2,
#                       verbose = 1,
#                       callbacks=[best_save])
    
    return model

def load_best_model():
    model = load_model('best_m.hdf5' ,custom_objects={'my_loss': my_loss, })
    score = model.evaluate(X_train, Y_train2, verbose=0) 
    print('loss:', score)
    return model

np.random.seed(42)
model = set_model()
train(model, X_train, Y_train2, is_val=False)    

best_model = load_best_model()
pred = best_model.predict(X_test)
pred[:5]

sns.displot(pred[:,0], bins=120)
# sns.displot(list(map(round, pred.argmax(axis=1))))

sns.displot(pred[:,0])

cutline1 = np.quantile(pred[:,0], 0.62)
cutline2 = np.quantile(pred[:,0], 0.67)
print(cutline1)
print(cutline2)

pred_var = np.var(pred, axis=1)
sample_submission['buy_quantity']=1
sample_submission['sell_time'] = list(map(lambda x: 0 if x <= cutline1 else (round(x) if x <= cutline2 else 119), pred[:,0]))
# sample_submission['buy_quantity'][sample_submission['sell_time']==1] = 0
sample_submission.to_csv("submission.csv", index=False)
Counter(sample_submission['sell_time'])

print(sample_submission['buy_quantity'].nunique())
print(sample_submission['sell_time'].nunique())