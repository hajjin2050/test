import numpy as np
import pandas as pd

# 데이터 로드
from pandas import read_csv
df = read_csv('../../data/dacon/train/train.csv', index_col=None, header=0)
print(df.tail())

data = df.values
print(data.shape)
np.save('../../data/dacon/npy/train.npy', arr=data)