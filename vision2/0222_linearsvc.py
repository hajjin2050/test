
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from numpy import expand_dims
from sklearn.model_selection import StratifiedKFold
from keras import Sequential
from keras.layers import *
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
import string
from tensorflow.keras.models import load_model
from sklearn.svm import LinearSVC
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import reciprocal, uniform
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
tf.debugging.set_log_device_placement(True)

# train 데이터 
train = pd.read_csv('C:/data/dacon/mnist2/data/train.csv')
test = pd.read_csv('C:/data/dacon/mnist2/data/test.csv')


# *********************
# train 데이터 

tmp1 = pd.DataFrame()

train = train.drop(['id','digit'],1)
test = test.drop(['id'],1)

tmp1 = pd.concat([train,test])

tmp1.loc[tmp1['letter']!='A','letter'] = 0
tmp1.loc[tmp1['letter']=='A','letter'] = 1

x = tmp1.to_numpy().astype('int32')[:,1:] # (852, 784)
y = tmp1.to_numpy().astype('int32')[:,0] # (852,)




from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y,  train_size=0.7, random_state = 66 ) # shuffle False면 섞는다


lin_clf = LinearSVC(random_state=42)
lin_clf.fit(x_train, y_train)

y_pred = lin_clf.predict(x_train)
accuracy_score(y_train, y_pred)
print(accuracy_score)

#스탠다드 스케일링
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train.astype(np.float32))
x_test_scaled = scaler.fit_transform(x_test.astype(np.float32))

svm_clf = SVC(decision_function_shape="ovr", gamma="auto")
param_distributions = {"gamma" : reciprocal(0.001, 0.1), "C" : uniform(1, 10)}
rnd_search_cv = RandomizedSearchCV(svm_clf, param_distributions, n_iter=10, verbose=2, cv=3)
rnd_search_cv.fit(x_train_scaled[:], y_train[:])

print(x_train_scaled.shape)
print(x_test_scaled)
'''
#평가
rnd_search_cv.best_score_
rnd_search_cv.best_estimator_.fit(x_train_scaled, y_train)
y_pred = rnd_search_cv.best_estimator_.predict(x_train_scaled)

sub = pd.read_csv('C:/data/dacon/mnist2/sample_submission.csv')

sub['a'] = np.where(y_pred> 0.5, 1, 0)

sub.to_csv('C:/data/dacon/mnist2/0217_0.csv', index = False)
'''