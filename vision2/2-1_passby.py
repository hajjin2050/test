#[step3] 파일로 저장된 데이터 불러오기 
import csv
import numpy as np
a = [[],[],[],[],[],[],[]] #7x24 크기의 리스트 선언

with open('C:/data/ebs/2-1/passby_data.CSV','r') as f :
    reader = csv.DictReader(f)
    i = j = 0    #i,j 변수 초기화 선언
    for row in reader :  #csv파일에 저장된 데이터 수만큼  반복해줌 
        a[i].append(row) 
        j = j+1# 24개의 행을 추가한 후 , 다음요일의 리스트로 이동
        if(j%24 == 0):
            i = i + 1

x_title = ['MON', 'TUE','WED','THR','FRI','SAT','SUN']

for i in range(0,7):
    for j in range(0, len(a[i])):
        print(x_title[i],'[',j,']=',a[i][j])