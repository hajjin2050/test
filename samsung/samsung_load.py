import numpy as np

x_train = np.load('../../data/npy/samsung_2_totaldata.npy', allow_pickle=True)[0]
y_train = np.load('../../data/npy/samsung_2_totaldata.npy', allow_pickle=True)[1]
x_val = np.load('../../data/npy/samsung_2_totaldata.npy', allow_pickle=True)[2]
y_val = np.load('../../data/npy/samsung_2_totaldata.npy', allow_pickle=True)[3]
x_test = np.load('../../data/npy/samsung_2_totaldata.npy', allow_pickle=True)[4]
y_test = np.load('../../data/npy/samsung_2_totaldata.npy', allow_pickle=True)[5]
x_pred = np.load('../../data/npy/samsung_2_totaldata.npy', allow_pickle=True)[6]

from tensorflow.keras.models import load_model
model = load_model('../../data/modelcheckpoint/samsung2_160-751.326721.hdf5')

#4. 평가, 예측
result = model.evaluate(x_test, y_test, batch_size=16)
# print('mse: ', result[0])
print('mae: ', result[1])

y_pred = model.predict(x_pred)
print('1/15일 삼성주식 종가: ', y_pred)
# 1/15일  종가:  [[89487.]]