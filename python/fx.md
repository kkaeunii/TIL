# EDA 실습     

* 데이터 불러오기    
* 데이터 살펴보기     
* 데이터 분포 살펴보기    
* 데이터 전처리
====================================================
## 환경 설정    
**패키지 설치**        
```
import numpy as np      
import pandas as pd       
import sklearn.datasets as D
```     
## 데이터 불러오기    
**데이터셋 준비**    
```
data = pd.read_csv('파일명', encoding = '인코딩 형식')
```          
**데이터 확인**
```
# 테이블 형태로 표 전체 정보 나옴    
# 상단 5행과 하단 5행을 보여주고, 열과 행 개수도 나옴     
data
```       

## 데이터 살펴보기  
* EDA 과정에서 데이터 정보 확인은 기본적인 정보 확인을 시작으로 분포나 상관관계같은 고수준 정보 확인하는 순서로 진행     
* 분석과정에 방해되는 요소 확인 및 전처리 진행       
      
**데이터 크기 확인**      
```
# 행과 열 정보 나옴    
print(data.shape)
```    
**데이터 컬럼 이름 확인**   
```
# 데이터에 columns을 지정하지 않았거나 잘못 파싱해서 첫번째 행이 column으로 나올 때   
# read_csv의 header 지정    
# pd.read_csv('파일명','header = ?')   
# header의 기본값은 0(첫 번째 행을 열 이름으로 사용)     
# None일 경우 column 없이 자동으로 0, 1, 2... 지정    
# n일 경우 n번째 행을 열 이름으로, [0,1]일 경우 0~1행을 계층적 열 이름으로 사용      
# 'infer'은 자동 추정으로 기본값과 동일       
       
data.columns
```    
**데이터 정보 확인**    
```
# column 이름과 행 개수, 데이터타입 등 확인 가능    
# 결측치 확인하기     
data.info()
```        
**랜덤으로 feature 하나 보기**    
```
# 여러 번 실행해도 같은 결과가 나오도록 초기값 고정    
# 새로운 방식 : numpy.random.default_rng(숫자)    
np. random.seed(숫자넣기)      

# 인덱스 랜덤 선택   
# 기본 문법 : np.random.choice(a, size=None, replace=True, p=None)      
# a는 선택 대상, size는 뽑을 개수, replace는 복원추출여부, p는 각 항목이 뽑힐 확률 리스트     
# seed를 먼저 설정하면 choice 결과도 고정됨     
feature_index = np.random.choice(len[data.columns))      

# 중복값 제거    
values = data[분석할 column].unique()    
# 중복값이 몇 개 존재하는지 확인    
data[분석할 column].value_counts()
```        
**데이터 결측치 확인**      
```
# 데이터 결측치 확인     
data.isnull()        
# 데이터 결측치 합    
data.isnull().sum()
```   
* 결측치 확인을 통해, 결측값 패턴을 생각해볼 수 있음.
* 랜덤으로 비어있는 게 아니라 동시에 비어있는 건 아닌지 등    
```
# 결측치가 하나라도 있는 행 확인    
sample_nan = data.isnull().sum(axis=1)>0   
# 결측치에 따라 가설을 세우고 확인한 후 전처리 진행
```      
* 이 외에도 별도의 값 확인 및 처리    
```
# 음수값 확인    
(data['음수값 의심 컬럼명'] < 0).any()       

# 최소/최대값 확인    
# print로 확인    
data[].min()     
datap[].max()
```      
**데이터 요약 통계량 확인**    
```
data.describe()
```    

## 데이터 분포 살펴보기   
**데이터 시각화 패키지 임포트**    
```
import matplotlib.pyplot as plt    
import seaborn as sns
```   
**boxplot**   
```
target_feature = '확인할 column'   
sns.boxplot(data=data, y=target_feature)    
plt.show()
```   
* boxplot 모양 확인 후 이상치 제거   
```
Q3 = data['확일할 column'].quantile(q=0.75)   
Q1 = data['확인할 column'].quantile(q=0.25) 
```   
**변수 분포 확인**    
```
# 히스토그램   
# kde 값을 설정하면 라인도 표
sns.histplot(data=data, x=target_feature)  
plt.show()     

# 꼬리가 긴 분포를 가진 변수의 히스토그램을 로그스케일로 확인    
sns.histplot(data=data, x=target_feature, log_scale=(False, True))    
plt.show()
```      
**막대그래프**     
```
# 막대로 범주 확인할 변수     
category_feature = '확인할 column'     
# 높이로 값을 확인할 변수    
target_feature = '확인할 column'     
    
# 막대그래프    
barplot = sns.barplot(data=data, x=category_feature, y=target_feature, color = '지정', errorbar=None)   
# x축 레이블 위치와 방향 설정 변경    
loc, labels = plt.xticks()     
barplot.set_xticklabels(labels, rotation=90)      
      
plt.title('타이틀 명')      
plt.show()
```     
**변수 간 관계 확인**     
* 상관관계는 두 변수가 함께 움직이는 경향성(상관관계 강도)을 -1~1 사이의 수치로 나타낸 지표       
```
# 상관계수 행렬 생성    
# numeric_only는 숫자형 변수만 계산하도록 함      
corr = data.corr(numeric_only=True)        
# figure에서 생략될 부분 지정하는 mask 행렬 생성, 실제로는 mask 없어도 괜찮음     
# mask = np.ones_like(corr, dtype=bool)    
# mask = np.triu(mask)         
         
# 히트맵 형태로 시각화       
sns.heatmap(data=corr, annot=True, fmt='.2f', mask=mask, linewidths=.5, cmap='색상')     
plt.title('타이틀 명')     
plt.show()
```
## 데이터 전처리     
**결측치 처리**    
```
# 처리할 데이터 확인   
# 비교를 위해 원복 데이터 복사   
# 결측치 중 가장 많은 비중을 차지하는 column을 0으로 대체   
data['바꿀 column'].fillna(0, inplace=True)   
# 나머지 결측치를 포함하는 모든 행 버리기(drop)   
data.dropna(axis=0, inplace=True)   
# 결측치 제거 확인   
data.isnull().sun()
```     
**데이터 분포 변환**   
* 분포 변환을 위한 함수   
```
from sklearn.preprocessing import scale
```   
* log scale로 변환   
```
# 로그 변환할 변수 선택 
target_feature = '변환할 column'   
   
# 데이터 로그 변환 
# log에 0값이 들어가는 것을 피하기 위해 1 더하기   
data[f'log_{target_feature}'] = scale(np.log(data[target_feature]+1))        
# 로그 변환한 변수 요약 통계량    
data[f'log_{target_feature}'].describe()
```            
**데이터 단위 변환**    
* 정규화와 표준화      
* 표준화 : 평균이 0이고 분산이 1이 되도록 변환        
* 정규화 : 최소값이 0이고 최대값이 1인 0과 1사이 값을 갖도록 변환     
* 데이터 분포 변환을 위한 함수 불러오기     
```
from sklearn.preprocessing import StandarScaler, MinMaxScaler
```     
* 표준화
```
# 스케일링 진행할 변수 선택    
target_feature = '진행할 column'    
# 표준화     
standard_scaler = StandardScaler()           
# .fit(x) : 입력 데이터 x의 통계적 정보를 계산해 저장 = 학습   
# .transform(x) : 저장된 정보를 바탕으로 x를 변환 (e.g. 평균 0, 표준편차 1로 스케일링)   
# .fit_transform(x) : 두 과정 한번에 처리     
data[f'standarized_{target_scaler}'] = standard_scaler.fit_transform(data[[target_feature]])
```    
* 정규화    
```
# normalized_scaler = MinMaxScaler()     
data[f'normalized_{target_feature}'] = normalized_scaler.fit_transform(data[[target_feature]])
```     
* 최대/최소값 비교    
* 시각화로 비교   
