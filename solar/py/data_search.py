import pandas as pd
import numpy as np
from datetime import timedelta
from glob import glob
import plotly.express as px
import matplotlib.pyplot as plt
pd.options.plotting.backend = 'plotly'
import plotly.io as pio

pio.renderers.default = "notebook_connected"

path = 'C:\workspace\sokar\data/'
files = sorted(glob(path+'*.csv'))

site_info = pd.read_csv(files[4]) # 발전소 정보
energy = pd.read_csv(files[2]) # 발전소별 발전량

dangjin_fcst_data = pd.read_csv(files[0]) # 당진 예보 데이터
dangjin_obs_data = pd.read_csv(files[1]) # 당진 기상 관측 자료

ulsan_fcst_data = pd.read_csv(files[5]) # 울산 예보 데이터
ulsan_obs_data = pd.read_csv(files[6]) # 울산 기상 관측 자료

sample_submission = pd.read_csv(files[3]) # 제출 양식

#발전소 정보
print(site_info.head())

#발전소별 발전량
print(energy.info())
print(energy.head(10))
print(energy.describe())

energy['date'] = energy['time'].apply(lambda x: x.split()[0])
energy['time'] = energy['time'].apply(lambda x: x.split()[1])
energy['time'] = energy['time'].str.rjust(8,'0') # 한자릿수 시간 앞에 0 추가 ex) 3시 -> 03시

# 24시를 00시로 바꿔주기
energy.loc[energy['time']=='24:00:00','time'] = '00:00:00'
energy['time'] = energy['date'] + ' ' + energy['time']
energy['time'] = pd.to_datetime(energy['time'])
energy.loc[energy['time'].dt.hour==0,'time'] += timedelta(days=1)

fig = px.line(energy[:24*7], x='time', y=['dangjin_floating','dangjin_warehouse','dangjin','ulsan'])
fig.show()

energy['month'] = energy['time'].dt.month
energy['hour'] = energy['time'].dt.hour

mean_month = energy.groupby('month').mean()
fig = px.bar(mean_month, x=mean_month.index, y=['dangjin_floating','dangjin_warehouse','dangjin','ulsan'])
fig.show()
plt.show()