import numpy as np 
import pandas as pd 
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno

import warnings
warnings.filterwarnings('ignore')

from skimage import io
from sklearn.cluster import KMeans

colors = []

img = io.imread('https://i.imgur.com/bGpKLYh.png')[:,:,:3]
img = img.reshape((img.shape[0] * img.shape[1], 3))

k = 10
clt = KMeans(n_clusters = k)
clt.fit(img)

for center in clt.cluster_centers_:
    color = [int(i) for i in list(center)]
    colors.append('#%02x%02x%02x' % (color[0], color[1], color[2]))

sns.palplot(colors)
plt.axis('off')
print(colors)

def custom_palette(custom_colors):
    customPalette = sns.set_palette(sns.color_palette(custom_colors))
    sns.palplot(sns.color_palette(custom_colors), size=0.8)
    plt.tick_params(axis='both', labelsize=0, length=0)

main_colors = ['#f03aa5', '#40c2f3', '#c489ce', '#bb3ca9']
custom_palette(main_colors)

#데이터셋 읽기
train = pd.read_csv('C:\workspace/tublar\data/train.csv')
test = pd.read_csv('C:\workspace/tublar\data/test.csv')
submission = pd.read_csv('C:\workspace/tublar\data/test.csv')

print('Shape of train dataset : ', train.shape) #(100000, 52)
print('Shape of test dataset : ', test.shape) #(50000, 51)

#EDA
#target value distribution
labels = list(train['target'].unique())
data = list(train['target'].value_counts())

plt.figure(figsize=(8,8))
plt.pie(data, autopct='%1.1f%%', labels=labels, textprops={'fontsize':15, 'color':'#505050'})

my_circle = plt.Circle((0,0), 0.8, color='white')
p = plt.gcf()
p.gca().add_artist(my_circle)

plt.legend(labels, loc='upper right', prop={'size':12})
plt.show()

#필요없는 칼럼 제거
train.drop('id', axis=1, inplace=True)
test.drop('id', axis=1, inplace=True)

#Making CMAP from main_colors
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

cmap = mpl.colors.LinearSegmentedColormap.from_list("", [main_colors[0], main_colors[1]])
cmap
