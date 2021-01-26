import numpy as np
import pandas as pd
import tensorflow.keras.backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv1D,GRU

train = pd.read_csv('C:/data/dacon/practice/train/train.csv')
submission = pd.read_csv('C:/data/dacon/practice/sample_submission.csv')

def preprocess_data(data, is_train = True):
    data['cos'] = np.cos(np.pi/2 - np.abs(data['Hour']%12 - 6)/6*np.pi/2) # cos는 GHI라는 정확한 데이터를 나타내는 열을 만들기위해 임의수식
    data.insert(1,'GHI',data['DNI']*data['cos']+data['DHI'])
    temp = data.copy()
    temp = temp[['Hour','TARGET','GHI','DHI','DNI','WS','RH','T']]

    if is_train ==True:
        temp['TARGET1'] = temp['TARGET'].shift(-48).fillna(method = 'ffill')  # 새로운 TARGET1 , TARGET2 라는 열을 만들건데
        temp['TARGET2'] = temp['TARGET'].shift(-96).fillna(method = 'ffill')
        temp = temp.dropna()

        return temp.iloc[:-96]

    elif is_train == False:
        temp = temp[['Hour','TARGET','GHI','DHI','DNI','WS','RH','T']]
        return temp.iloc[-48:,:]

df_train = preprocess_data(train)
x_train = df_train.to_numpy()

print(df_train)

df_test = []
for i in range(81):
    file_path = 'C:/data/dacon/practice/test/%d.csv'%i
    temp = pd.read_csv(file_path)
    temp = preprocess_data(temp, is_train=False)
    df_test.append(temp)

x_test = pd.concat(df_test)
x_test = x_test.to_numpy()
print(x_test.shape) #(3888, 7)

def split_xy(data, timestep):
    x, y1,y2 = [],[],[]
    for i in range(len(data)): #for는 반복문 
        x_end = i + timestep #x의 끝을 정해주고 반복해주기
        if x_end>len(data): #근데 x가 정해진data보다 길면 break
            break
        tmp_x = data[i:x_end,:-2] #i부터x끝까지 행 , 뒤에서두번째 열까지 슬라이싱 
        tmp_y1 = data[x_end-1:x_end,-2] # x 뒤에값으로 첫째날 값을 정해줌 
        tmp_y2 = data[x_end-1:x_end,-1] #             둘쨰날 값을정해줌
        x.append(tmp_x)           
        y1.append(tmp_y1)
        y2.append(tmp_y2)
    return(np.array(x),np.array(y1),np.array(y2))

x,y1,y2 = split_xy(x_train,1)

def split_x(data, timestep):
    x=[]
    for i in range(len(data)):
        x_end = i + timestep
        if x_end>len(data):
            break
        tmp_x = data[i:x_end]
        x.append(tmp_x)
    return(np.array(x))

x_test = split_x(x_test,1)
print(x.shape, y1.shape, y2.shape) #(52464, 1, 8) (52464, 1) (52464, 1)

from sklearn.model_selection import train_test_split
x_train, x_val, y1_train, y1_val, y2_train, y2_val = train_test_split(x,y1,y2, train_size=0.7, shuffle=False,random_state =220)

def quantile_loss(q, y_true, y_pred):
    err = (y_true - y_pred)
    return K.mean(K.maximum(q*err, (q-1)*err), axis=-1)

quantiles = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

# 2.모델링
def mymodel():
    model = Sequential()
    model.add(Conv1D(256,2,padding = 'same', activation = 'relu',input_shape = (1,8)))
    model.add(Conv1D(128,2,padding = 'same', activation = 'relu'))
    model.add(Conv1D(64,2,padding = 'same', activation = 'relu'))
    model.add(Conv1D(32,2,padding = 'same', activation = 'relu'))
    model.add(Flatten())
    model.add(Dense(128, activation = 'relu'))
    model.add(Dense(64, activation = 'relu'))
    model.add(Dense(32, activation = 'relu'))
    model.add(Dense(16, activation = 'relu'))
    model.add(Dense(8, activation = 'relu'))
    model.add(Dense(1))
    return model
# def mymodel():  GRU모델에서는 loos,val값 모두 변동도없고 오류뜸 (nparray)
#     model = Sequential()
#     model.add(GRU(256,activation='relu', input_shape=(1,8)))
#     model.add(Dense(128, activation='relu'))
#     model.add(Dense(64, activation='relu'))
#     model.add(Dense(32,activation='relu'))
#     model.add(Dense(16,activation='relu'))
#     model.add(Dense(8,activation='relu'))
#     model.add(Dense(4,activation='relu'))
#     model.add(Dense(1,activation='relu'))
#     return model

#3.컴파일
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', patience=10)
lr = ReduceLROnPlateau(monitor = 'val_loss', patience=5, factor = 0.3, verbose= 1)
epochs = 100

#1일치
for i in quantiles:
    model = mymodel()
    filepath_cp = f'C:/data/dacon/modelcheckpoint/dacon_y1_quantile_{i:.1f}.hdf5'
    cp = ModelCheckpoint(filepath_cp,save_best_only=True,monitor = 'val_loss')
    model.compile(loss = lambda y_true,y_pred: quantile_loss(i,y_true,y_pred), optimizer = 'adam', metrics = [lambda y,y_pred: quantile_loss(i,y,y_pred)])
    model.fit(x_train,y1_train,epochs = epochs, batch_size = 36, validation_data = (x_val,y1_val),callbacks = [es,cp,lr])
    pred = pd.DataFrame(model.predict(x_test).round(2))
    x.append(pred) #AttributeError: 'numpy.ndarray' object has no attribute 'append'
df_temp1 = pd.concat(x, axis = 1)
df_temp1[df_temp1<0] = 0
num_temp1 = df_temp1.to_numpy()
submission.loc[submission.id.str.contains("Day7"), "q_0.1":] = num_temp1

x = []
# 모레!!
for i in quantiles:
    model = mymodel()
    filepath_cp = f'C:/data/dacon/modelcheckpoint/dacon_y2_quantile_{i:.1f}.hdf5'
    cp = ModelCheckpoint(filepath_cp,save_best_only=True,monitor = 'val_loss')
    model.compile(loss = lambda y_true,y_pred: quantile_loss(i,y_true,y_pred), optimizer = 'adam', metrics = [lambda y,y_pred: quantile_loss(i,y,y_pred)])
    model.fit(x_train,y2_train,epochs = epochs, batch_size = 36, validation_data = (x_val,y2_val),callbacks = [es,cp,lr])
    pred = pd.DataFrame(model.predict(x_test).round(2)) # 2째자리 뒤부터 반올림
    x.append(pred)
    
df_temp2 = pd.concat(x, axis = 1)
df_temp2[df_temp2<0] = 0
num_temp2 = df_temp2.to_numpy()
submission.loc[submission.id.str.contains("Day8"), "q_0.1":] = num_temp2
        
submission.to_csv('C:/data/dacon/data/0121_1.csv', index = False)