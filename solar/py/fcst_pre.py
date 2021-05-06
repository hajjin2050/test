
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#데이터 불러오기
dangjin_fcst = pd.read_csv('./data/dangjin_fcst_data.csv')
#print(dangjin_fcst) #162208 rows x 7 columns

#예보시간 컬럼의 데이터 타입을 datetime으로 변경합니다
dangjin_fcst['Forecast time'] = pd.to_datetime(dangjin_fcst['Forecast time'])
fcst_14 = dangjin_fcst[dangjin_fcst['Forecast time'].dt.hour==14]
fcst_14 = fcst_14[(fcst_14['forecast']>=10)&(fcst_14['forecast']<=33)]

def to_date(x):
    return pd.DateOffset(hours=x)
fcst_14['Forecast time'] = fcst_14['Forecast time'] + fcst_14['forecast'].map(to_date)
fcst_14 = fcst_14[['Forecast time', 'Temperature', 'Humidity', 'WindSpeed', 'WindDirection', 'Cloud']]

#print(fcst_14) [8768 rows x 6 columns]

fcst_14_ = pd.DataFrame()
fcst_14_['Forecast time'] = pd.date_range(start='2018-03-02 00:00:00', end='2021-03-01 23:00:00', freq='H')

#print(fcst_14_)[26304 rows x 1 columns]
#기존 예보 데이터프레임과 병합
fcst_14_ = pd.merge(fcst_14_, fcst_14, on='Forecast time', how='outer')
print(fcst_14_)#중간중간 결측치발견 

#pandas에서 제공하는 선형보간법으로 결측치를 채움
inter_fcst_14 = fcst_14_.interpolate()
#print(inter_fcst_14)[26304 rows x 6 columns]

#시각화
plt.figure(figsize=(20,5))
days = 5
plt.plot(inter_fcst_14.loc[:24*days, 'Forecast time'], inter_fcst_14.loc[:24*days, 'Temperature'], '.-')
plt.plot(fcst_14_.loc[:24*days, 'Forecast time'], fcst_14_.loc[:24*days, 'Temperature'], 'o')

inter_fcst_14.to_csv('./data/new_dangjin_fcst.csv', index=False)