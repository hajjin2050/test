import numpy as np
import pandas as pd

#1.데이터
df = pd.read_csv('../../data/csv/KODEX 코스닥150 선물인버스.csv', index_col=0, header=0, encoding='cp949', thousands=',')
df = df.dropna()
df = df.drop(['전일비','Unnamed: 6'], axis=1)
df = df.drop(['금액(백만)','신용비','외인(수량)','외인비','기관','프로그램','외국계'], axis=1)

# [80 rows x 7 columns]

# 역방향읽기
df2 = df.iloc[::-1].reset_index(drop=True)

x = df.iloc[1010:1087, [0,2,3,4,6,8]]
y = df.iloc[1010:1090, 1]
x_pred = df.iloc[1010:1087, [0,2,3,4,6,8]]
print(x.shape)

#npy저장
data = df.values
np.save('../../data/npy/KODEX.npy',arr=data)