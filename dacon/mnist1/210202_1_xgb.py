import numpy as np
import pandas as pd
import tensorflow.keras.backend as K

from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score,RandomizedSearchCV
from sklearn.metrics import accuracy_score

from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from keras.utils.np_utils import to_categorical
from sklearn.decomposition import PCA

train = pd.read_csv('C:/data/dacon/mnist1/data/train.csv')
submission = pd.read_csv('C:/data/dacon/mnist1/data/submission.csv')
test = pd.read_csv('C:/data/dacon/mnist1/data/test.csv')

temp = pd.DataFrame(train)
test_df = pd.DataFrame(test)

x = temp.iloc[:,3:]/255
y = temp.iloc[:,[1]]
x_test = test_df.iloc[:,2:]/255

x = x.to_numpy()
y = y.to_numpy()
x_pred = x_test.to_numpy()

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size =0.7, random_state = 66)
kfold = KFold(n_splits = 5, shuffle= True)

#2. MODEL
model = XGBClassifier(base_score=0.5, booster = 'gbtree', colsample_bylevel=1,colsample_bynode=1,
                      colsample_bytree=1, eval_metric = 'mlogloss',gamma=0, learning_rate=0.1
                      , max_delta_step=0, max_depth=6, nthread=None, objective = 'multi:softprob',
                      random_state=0,reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
                      silent = None, subsample=1, verbosity =1)

#3.TRAIN
model.fit(x_train, y_train)

#4. EVALUATE
acc = model.score(x_test, y_test)
print("acc:", acc)

y_pred = model.predict(x_pred)
pred = pd.DataFrame(y_pred)

submission.loc[:,'digit'] = pred
submission.to_csv('C:/data/dacon/mnist1/submit/210202_xgb.csv',index= False)

#acc: 0.4975609756097561