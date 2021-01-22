#참고 조 서동재님 깃허브 (dacon_load_model.py)

import numpy as np
import pandas as pd

df = pd.read_csv('../../data/dacon/train/train.csv', index_col = 0, header=0)

# df.drop(['Hour','Minute','Day'], axis =1 , inplace= True) #  inplace=True는 df = df.drop('A', axis=1)과 같다.
df.drop(['Hour','Minute'], axis=1, inplace = True)
data = df.to_numpy()
data = data.reshape(1095,48,6)

def split_xy(data,time_step, y_num):
    x,y = list() , list()
    for i in range(len(data)):
        x_end = i +time_step
        y_end = x_end + y_num
        if y_end > len(data):
            break                             
        x_tmp = data[i:x_end]
        y_tmp = data[x_end : y_end, :,-1]
        x.append(x_tmp) 
        y.append(y_tmp)
    return(np.array(x),np.array(y))
x,y = split_xy(data,7,2)
# print(x)
# print(y)


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, shuffle = True, random_state = 30)

#2. 모델구성
from tensorflow.keras.models import load_model

d = []
for l in range(9):
    model = load_model('../../data/modelcheckpoint/dacon%d.hdf5'%l)
    c = []
    for i in range(81):
        testx = pd.read_csv('../../data/dacon/test/%d.csv'%i)
        testx.drop(['Hour','Minute','Day'], axis = 1 , inplace= True)
        testx = testx.to_numpy()
        testx = testx.reshape(7,48,6)
        testx,nully_y = split_xy(testx,7,0)
        y_pred = model.predict(testx)
        y_pred = y_pred.reshape(2,48)
        a = []
        for j in range(2):
            b=[]
            for k in range(48):
                b.append(y_pred[j,k])
            a.append(b)
        c.append(a)
    d.append(c)
d=np.array(d)

e = []
for i in range(81):
    f=[]
    for j in range(2):
        g = []
        for k in range(48):
            h = []
            for l in range(9):
                h.append(d[l,i,j,k])
            g.append(h)
        f.append(g)
    e.append(f)

e=np.array(e)
df_sub = pd.read_csv('../../data/dacon/sample_submission.csv', index_col=0, header=0)

for i in range(81):
    for j in range(2):
        for k in range(48):
            df = pd.DataFrame(e[i,j,k])
            for l in range(9):
                df_sub.iloc[[i*96 + j*48 + k],[l]] = df.quantile(q = ((l+1)/10.), axis = 0)[0]

df_sub.to_csv('../../data/dacon/submit_2_0120.csv')