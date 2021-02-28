#[step3] 파일로 저장된 데이터 불러오기 
import csv
import numpy as np
a = [[],[],[],[],[],[],[]] #7x24 크기의 리스트 선언

with open('C:/data/ebs/2-1/passby_data.CSV','r') as f :
    reader = csv.DictReader(f)
    i = j = 0    #i,j 변수 초기화 선언
    for row in reader :  #csv파일에 저장된 데이터 수만큼  반복해줌 
        a[i].append(row) 
        j = j + 1# 24개의 행을 추가한 후 , 다음요일의 리스트로 이동
        if(j % 24 == 0):
            i = i + 1

################2-1의 내용 복붙#########################
day_title = ['MON','TUE','WED','THR','FRI','SAT','SUN']
hour_title =['01', '02', '03', '04', '05', '06', \
             '07','08','09','10','11','12', \
             '13','14','15','16','17','18', \
             '19','20','21','22','23','24',]
#시간대별로 주간 평균 구하기
avgh = []
for j in range(0,24):
    day_sum = 0
    #j번째 시간대 주간 총합
    for i in range(0,7):
        day_sum = day_sum + int(a[i][j]['num'])

    avgh.append(day_sum/7)

#시간대별로 평균 유동 인구 출력하기
for j in range(0, 24):
    print("[~{0}:00]:{1:4}". format(hour_title[j],int(avgh[j])))