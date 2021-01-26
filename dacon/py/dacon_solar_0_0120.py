import pandas as pd
import numpy as np
import glob
import random
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K
import tensorflow.keras.layers as L
import tensorflow.keras.models as M


# base_line study

train = pd.read_csv('C:/data/dacon/practice/train/train.csv')
sub = pd.read_csv('C:/data/dacon/practice/sample_submission.csv')
#preprocess_data 함수정의
def preprocess_data(data):
    temp = data.copy()
    return temp.iloc[-48:,:]
    #마지막하루만리턴
df_test=[]
#81개의 테스트 파일들의 마지막 날짜하루치(48개의 데이터)를 붙임
for i in range(81):
    file_path = 'C:/data/dacon/practice/test/' + str(i) + '.csv'
    temp = pd.read_csv(file_path)
    temp = preprocess_data(temp)
    df_test.append(temp)
print(temp.shape) #(48, 9/)

X_test = pd.concat(df_test)
X_test = X_test.append(X_test[-96:])
X_test.shape
print(X_test.shape)#(3984, 9)

# 왜 c값과 b값이 243.12, 17.62일까  그리고  gamma에 대해서 알아보기
def Add_features(data):
    c = 243.12
    b = 17.62
    gamma = (b*(data['T'])/(c + (data['T']))) + np.log(data['RH']/100)
    dp = ( c* gamma)/(b - gamma)
    data.insert(1,'Td',dp)
    data.insert(1,'T-Td', data['T']-data['Td']) 
    data.insert(1,'GHI',data['DNI']+data['DHI'])
    return data #컬럼이 늘어남!

train = Add_features(train)
X_test = Add_features(X_test)

df_train = train.drop(['Day', 'Minute'], axis=1)
df_test = X_test.drop(['Day', 'Minute'], axis=1)

column_indices = {name: i for i , name in enumerate(df_train.columns)}
# i for i ,enumertae =>알아보기

#Train and aValidation split
n = len(train)
train_df = df_train[0:int(n*0.8)]
val_df = df_train[int(n*0.8):]
test_df = df_test

#Normalization
#데이터 전처리
num_features = train_df.shape[1]

train_mean = train_df.mean()
train_std = train_df.std()

train_df = (train_df - train_mean)/ train_std
val_df = (val_df - train_mean)/ train_std
test_df = (test_df - train_mean)/ train_std
''' # WindowGenrator 데이터 시각화
class WindowGenerator():
  def __init__(self, input_width, labe_width, shift,
    train_df = train_df, val_df=val_df, test_df = test_df,
    label_columns=None):
    #Store the raw data
    self.train_df = train_df
    self.val_df = val_df
    self.test_df = test_df
    #Work out the label column indices/
    self.label_columns = label_columns
    if label_columns is not None:
        self.label_columns_indices = {name: i for i, name in enumerate(label_columns)}
    self.column_indices = {name: i for i, name in enumerate(train_df.columns)}
    #Work out the woindow parameters
    self.input_width = input_width
    self.label_width = label_width
    self.shift = shift
    self.total_window_size = input_width + shift
    self.input_slice = slice(0, input_width)
    self.input_start = self.total_window_size - self.label_width
    self.labels_slice = slice(self.label_start, None)
    self.label_indices = np.arange(self.total_window_size)[self.labels_slice]
def __repr__(self):
    return '\n'.join([
        f'Total window size : {self.total_window_size}',
        f'Input indices: {self.input_indices}',
        f'Label indices: {self.label_indices}',
        f'Label column name(s): {self.label_columns}'])
'''

############Quatiole loss definition
# quantile함수=> 실제값이 더 클경우에는q, 예측값이 더 클경우에는1-q 곱한만큼의 절댓값
def quantile_loss(q,y_true,y_pred):
    err = (y_true, y_pred)
    return K.mean(K.maximum(q*err, (q-1)*err),axis=-1)

quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
OUT_STEPS = 96 #features output units 

early_sopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2,mode='min')

def DenseModel():
    model = tf.keras.Sequential()
    model.add(L.Lambda(lambda x: x[:,-1:, :]))
    model.add(L.Dense(512, activation='relu'))
    model.add(L.Dense(256, activation='relu'))
    model.add(L.Dense(OUT_STEPS*num_features, kernel_initializer=tf.initializers.zeros))
    model.add(L.Reshape([OUT_STEPS, num_features]))
    return model

# 빈 데이터 프레임 만들어주기
Dense_actual_pred = pd.DataFrame()
Dense_val_score = pd.DataFrame()




# for q in quantiles:
#     	model = DenseModel()
# 	model.compile(loss = lambda y_true, y_pred: quantile_loss(q, y_true, y_pred), optimizer='adam', metrics=[lambda y, pred: quantile_loss(q, y, pred)])
# 	history = model.fit(w1.train, validation_data=w1.val, epochs=20, callbacks=[early_stopping])
# 	pred = model.predict(w1.test, verbose=0)
# 	target_pred = pd.Series(pred[::48][:,:,9].reshape(7776)) #Save predicted value (striding=48 step, 9 = TARGET) 
# 	Dense_actual_pred = pd.concat([Dense_actual_pred,target_pred],axis=1)
# 	Dense_val_score[f'{q}'] = model.evaluate(w1.val)
# 	w1.quantile_plot(model, quantile=q)