import pandas as pd
import numpy as np


df = pd.read_csv('../../data/csv/samsung.csv', index_col =0, header=0)

print(df)

print(df.shape) # (150,5)
print(df.info())

# df = pd.DataFrame(my_array, columns = ['Column_A','Column_B','Column_C'])

aaa = df.to_numpy() # pandas --> numpy
print(aaa)
print(type(aaa))
# 위와 동일
bbb= df.value_counts
print(bbb)
print(type(bbb))

np.save("../../data/npy/samsung.npy")