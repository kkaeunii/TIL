# 모듈
## 내부 모듈     
* [다양한 내부 모듈](https://docs.python.org/ko/3/library/index.html)     
* 모듈 전체 가져오기 -> import 모듈 이름       
* 모듈 내 필요한 것만 가져오기 -> from 모듈 이름 import 가져오고 싶은 것        
* 모듈 사용 -> 모듈 뒤에 . 찍고 원하는 함수 입력     
       
* 자주 쓰이는 모듈과 함수         
### 숫자와 수학 모듈         
                 
| math |     
| --- |       
| import math <br><br> math.sqrt() : 제곱근 <br> math.ceil() : 올림 <br> math.floor() : 내림 <br> math.pow() : 제곱 <br> math.factorial() : n! |              
            
| random |        
| --- |            
| import random <br><br> random.random() : 0 ~ 1사이 실수<br> random.randint(n1, n2) : n1 ~ n2 사이 정수 <br>random.choice(['a','b','c']) : 리스트에서 랜덤 선택 <br> random.shuffle(['a','b','c']) : 리스트 섞기 |      
                  
| statistics |            
| --- |               
| import statistics <br><br> statistics.mean([리스트]) : 평균 <br> statistics.median([리스트]) : 중앙값<br> statistics.variance([리스트]) : 분산 |                                    
                                
|decimal|        
| --- |           
|from decimal import Decimal <br><br> Decimal('소수') 연산자 Decial('소수') : 고정 소수점 연산 |      
            
### 일반 운영 체제 서비스       
              
| os |          
| --- |           
|import os <br><br> os.getcwd() : 현재 디렉토리 <br> os.listdir() : 현재 폴더의 파일 목록 <br> os.mkdir('폴더명') : 새 폴더 생성 <br> os.remove('파일명') : 파일 삭제 |               
                           
| time |           
| --- |          
|import time <br><br> time.time() : 현재 시간(timestamp) <br> time.sleep(초) : 초만큼 멈추기 <br> time.ctime() : 현재 시간 문자열 |           
              
### 데이터형 - 날짜와 시간       
        
| datetime |        
| --- |         
|import datetime <br><br> 변수 = datetime.datetime.now() : 현재 시간 <br> 변수 = datetime.date.today() : 오늘 날짜 <br> datetime.datetime(연, 월, 일) : 특정 날짜 생성 |              
                         
## 외부 모듈       
### 수치 계산       
         
| NumPy |      
| --- |      
| import numpy as np <br><br> np.array([1,2,3]) : 배열 생성 <br> np.zeros((행,열)) : 0으로 이뤄진 배열 <br>np.ones() : 1로 채워진 배열 <br> np.arange(n1,n2,n3) : n1부터 n2까지 n3간격 <br> np.mean() : 평균 <br> np.std() : 표준편차 <br> np.dot(행,열) : 행렬 내적 |           
        
### 데이터 처리     
      
| Pandas |      
| --- |         
| import pandas as pd <br><br> pd.read_csv('파일 경로') <br> df.head() / df.tail() : 상위/하위 행 보기 <br> df.info() : 데이터 요약 정보 <br> df.describe() : 데이터 통계 요약 <br> df['column'] / df.iloc[] / df.loc[] : 열/행 접근 <br> df.drop(column, axis = ?) : 열 삭제 <br> df.groupby() : 그룹화|           
                
### 데이터 시각화     
       
| Matplotlib |          
| --- |             
|import matplotlib.pyplot as plt <br><br> plt.plot() : 선 그래프 <br> plt.bar() : 막대 그래프 <br> plt.hist() : 히스토그램 <br> plt.title(), plt.xlabel(), plt.ylabel() : 제목 <br> plt.show() : 그래프 출력 |        
          
### 고급 시각화      
       
| Seaborn |           
| --- |          
|import seaborn as sns <br><br> sns.lineplot(x='x', y='y', data=데이터) : 선 그래프<br> sns.barplot(x='x', y='y', data=데이터) : 막대 그래프 <br> sns.heatmap(데이터.corr(), annot=True) : 상관관계 히트맵 <br> sns.boxplot(x=column, data=데이터) : 박스플롯|             
                   
### 웹 데이터 수집    
            
| BeautifulSoup |         
| --- |           
|from bs4 import BeautifulSoup <br> import requests <br><br> html = requests.get('url 주소').text <br> soup = BeautifulSoup(html, 'html.parser') <br><br> soup.title.text : 제목 추출 <br> soup.find('태그') : 첫번째 '태그' 추출 <br> soup.find_all('태그') : 모든 '태그' 리스트 <br> soup.select('css 선택자') : 선택자 기반 추출 |         
                  
### 머신러닝       
            
| Scikit-Learn |        
| --- |          
| from sklearn.model_selection import train_test_split <br> from sklearn.linear_model import LinearRegression <br> from sklearn.metrics import accuracy_score <br><br> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3) : 데이터 분할<br> model = LinearRegression() <br>model.fit(X_train, y_train) : 모델 생성 및 학습 <br> y_pred = model.predict(X_test) <br> accuracy_score(y_test, y_pred) : 예측 및 평가 |
