import numpy as np
import pandas as pd

df = pd.read_csv('../../data/dacon/train/train.csv')
df.drop(['Hour','Minute','Day'], axis=1, inplace = True)
print(df.shape) #(52560, 6)

data = df.to_numpy()
data = data.reshape(1095,48,6)

def split_xy(data, step, y_num):
    x,y = list(),list()
    for i in range(len(data)):
        x_end = i +step
        y_end = x_end + y_num
        if y_end > len(data):
            break
        x_tmp = data[i:x_end]
        y_tmp = data[x_end:y_end,:,-1]
        x.append(x_tmp)
        y.append(y_tmp)
    return(np.array(x),np.array(y))
x,y = split_xy(data,7,2)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, shuffle= True, random_state=30)

#2.MDOEL
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D,Flatten, Dropout, MaxPooling2D, LeakyReLU, Reshape
from tensorflow.python.keras.callbacks import ReduceLROnPlateau
model = Sequential()
a = model.add(LeakyReLU(alpha= 0.05))
b = model.add(MaxPooling2D(2))
c = model.add(Dropout(0.3))

model.add(Conv2D(512,2,padding='same',input_shape=(7,48,6)))
model.add(LeakyReLU(alpha= 0.05))
model.add(MaxPooling2D(2))
model.add(Dropout(0.3))
model.add(Conv2D(256, 2, padding='same'))
model.add(LeakyReLU(alpha= 0.05))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(256))
model.add(LeakyReLU(alpha= 0.05))
model.add(Dropout(0.3))
model.add(Dense(128))
model.add(LeakyReLU(alpha= 0.05))
model.add(Dropout(0.3))
model.add(Dense(96))
model.add(Dropout(0.3))
model.add(Reshape((2,48)))
# model.summary()

#3.compile

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
es = EarlyStopping(monitor = 'val_loss', patience=20)
lr = ReduceLROnPlateau(monitor= 'val_loss', factor= 0.3, patience=10, verbose=1)
model.compile(loss = 'mse', optimizer='adam', metrics=['mae'])

#
d = []
for l in range(9): #모델 9번 돌리기
    cp = ModelCheckpoint(filepath = '../../data/modelcheckpoint/dacon%d.hdf5'%l,save_best_only = True)
    model.fit(x,y,epochs=1000, validation_split=0.2, batch_size = 8, callbacks = [es,lr,cp])

    c = []
    for i in range(81): #81개의 테스트파일 돌리기
        testx = pd.read_csv('../../data/dacon/test/%d.csv'%i)
        testx.drop(['Hour','Minute','Day'], axis =1, inplace = True)
        testx = testx.to_numpy()
        testx = testx.reshape(7,48,6)
        testx,nully_y = split_xy(testx,7,0)
        y_pred = model.predict(testx)
        y_pred = y_pred.reshape(2, 48)
        a = []
        for j in range(2): #2일치
            b = []
            for k in range(48): #하루를 쪼갠 48개의 데이터
                b.append(y_pred[j,k])
            a.append(b)
        c.append(a)
    d.append(c)
d = np.array(d)


e = []
for i in range(81):
    f = []
    for j in range(2):
        g = []
        for k in range(48):
            h = []
            for l in range(9):
                h.append(d[l,i,j,k])
            g.append(h)
        f.append(g)
    e.append(f)

e = np.array(e)
df_sub = pd.read_csv('../../data/dacon/sample_submission.csv', index_col=0, header=0)

for i in range(81):
    for j in range(2):
        for k in range(48):
            df = pd.DataFrame(e[i,j,k])
            for l in range(9):
                df_sub.iloc[[i*96+j*48+k],[l]] = df.quantile(q = ((l+1)/10.),axis=0)[0]

df_sub.to_csv('../../data/dacon/submit_2_0120.csv')