# 선형회귀 모델링 방법론 실습
* 실습 데이터셋 준비     
* 선형회귀모델 학습      
* 상관계수 분석           

## 환경설정       
* 당뇨병 진행도 예측 데이터셋 활용   
```
import pandas as pd     
import numpy as np      
import matplotlib.pyplot as plt       
import seaborn as sns          
from sklearn.datasets import load_diabetes    
from sklearn.model_selection import train_test_split     
from sklearn.linear_model import LinearRegression         
from sklearn.metrics import mean_squared_error
```      

## 실습 데이터셋 준비     
**데이터셋 로드 및 설명 확인**
```
# 데이터셋이 기본적으로 column마다 샘플을 다 더했을 때 표준편차가 1이 되도록 정규화되는 scaled 옵션이 설정되어 있음     
# scaled=False 로 스케일 조정 없이 원본 데이터 받음      
diabetes = load_diabetes(scaled=False)       
          
# 데이터셋 설명(description) 확인   
# DESCR : 데이터셋 설명(문자열 타입), sklearn.datasets에서 제공     
print(diabetes['DESCR'])
```
**데이터셋 전처리**    
```
# pandas 데이터프레임으로 변환      
data = diabetes['data']        
data = pd.DataFrame(data, column=diabetes['feature_names'])      
         
# feature 별 평균값과 표준편차 계산        
fts_mean = data.mean(axis=0)     
fts_mean = data.std(axis=0)             
        
# 평균 0, 표준편차 1이 되도록 표준화        
data = (data - fts_mean) / fts_std              
        
# 결과 확인         
data.describe()
```    
**target 확인**        
```
# target은 DESCR로 확인 가능하며, 훈련용 데이터셋에서는 정답이 이미 포함되어있음     
# 데이터프레임으로 만들었을 때 안 보이는 이유는, 데이터프레임은 data로 만들어서 target이 들어가지 않았기 때문       
target = diabets['target']        
       
# target 변수 모양 확인        
print(target.shape)             
        
# target의 평균값과 표준편차 계산      
tgt_mean = target.mean()       
tgt_std = target.std()        
        
# 표준화 적용       
target = (target - tgt_mean) / tgt_std
```      
**데이터셋 분할**
```
# 무작위 동작을 고정하기 위한 시드값      
# 무작위가 달라지면 모델 훈련 결과, 성능 평가 결과 등이 달라짐     
# 숫자는 아무 정수나 지정 ; 숫자를 고정하는 게 중요     
random_state = 1234      
       
# train과 test를 7:3으로 분할     
# train_test_split이 함수
train_data, test_data, train_target, test_target = train_test_split(data, target, test_size=0.3, random_state=random_state)       
    
# 분할 확인    
print(train_data.shape)    
print(train_target.shape)    
print(test_data.shape)    
print(test_target.shape)
```      
## 선형회귀 모델 학습
**선형회귀 모델 학습**
```
# (다중) 선형회귀모델 초기화    
multi_regressor = LinearRegression()     
        
# 학습용 데이터로 학습 진행    
# train_data 는 입력 데이터, train_target 은 정답값(예측하고 싶은 값)
multi_regressor.fit(train_data, train_target)      
      
# 회귀식 intercept(절편) 확인    
print(multi_regressor.intercept_)      
# 회귀식 계수 확인     
print(multi_regressor.coef_)
```
**학습 결과 확인**
```
# 회귀식 예측값 계산    
# 왜 train_data도 예측하지? 학습이 잘 이루어졌는지 보려고    
multi_train_pred = multi_regressor.predict(train_data)      
multi_test_pred = multi_regressor.predict(test_data)
```   
**MSE 손실함수 값 계산**  
```
multi_train_mse  = mean_squared_error(multi_train_pred, train_target)     
multi_test_mse = mean_squared_error(multi_test_pred, test_target)       
      
# 소수점 5자리로 고정 포맷팅     
# : 포맷팅 시작 신호, .5 소수점 아래 5자리, f : 고정 소수점 형식    
# 주의! 반드시 문자열 안에 {} 변수로 포맷팅       
print(f'Train MSE : {multi_train_mse: .5f}')     
print(f'Test MSE : {multi_test_mse: .5f}')
```   
**결과 시각화**
* 라인에 가까울 수록 잘 예측된 것   
```
plt.figure(figsize=(4, 4))    
    
plt.xlabel('target')    
plt.ylabel('prediction')       
        
# 산점도로 도식화     
y_pred = multi_regressor.predict(test_data)     
plt.plot(test_target, y_pred, '.')            
         
# y = x 직선     
x = np.linspace(-2.5, 2.5, 10)     
y = x     
plt.plot(x, y)     
      
plt.show()
```    
**해석적 해법으로 파라미터 계산**
```
# linear equation(선형방정식)인 Ax = b 형태로 변형     
# @ 는 행렬 곱 연산자 
A = train_data.T @ train_data     
b = train_data.T @ train_target     
     
# linear equation 풀기     
# np.linalog.solve()는 역행렬을 직접 계산하지 않고 더 빠르게 정확하게 구함      
coef = np.linalg.solve(A, b)      
       
# 학습된 parameter를 이용해 예측값을 내놓는 함수 정의     
def predict(data, coef) :
    return data @ coef         
            
# 학습, 평가 데이터셋에서 회귀식의 예측값 계산     
train_pred = predict(train_data, coef)     
test_pred = predict(test_data, coef)       
         
# MSE 손실함수 값 계산       
train_mse = mean_squared_error(train_pred, train_target)      
test_mse = mean_squared_error(test_pred, test_target)      
    
print(f'Multi Regression Train MSE is {train_mse: .5f}')       
print(f'Multi Regression Test MSE is {test_mse: .5f}')
```
**학습 결과**    
```
# scikit-learn 패키지 활용 학습 결과     
print(multi_regressor.coef_)     
# 해석적 해법 활용 학습 결과     
prin(coef)        
        
# 잘못한 게 아니라면 둘의 결과가 같음
```    
## 단순선형회귀와 상관관계 분석
**상관행렬 구하고 시각화**  
```
# 상관계수 행렬 생성    
corr = data.corr(numeric_only = True)           
       
# 생략될 부분 지정하는 mask 행렬 생성     
mask = np.ones_like(corr, dtype=bool)     
mask = np.triu(mask)        
      
# 시각화      
plt.figure(figsize = (13, 10)      
sns.heatmap(data=corr, annot=True, fmt=' .2f', mask=mask, linewidth=.5, camp='색상')     
plt.title('타이틀 명')      
plt.show()
```   
**상관계수와 회귀계수(결정계수)**
* 상관행렬에서 두 변수를  선택해 단순선형회귀분석 모델 학습     
```
# 두 변수 선정    
x_feature = 's3'      
y_feature = 's2'         
      
# 모델 초기화 및 학습    
simple_regressor = LinearRegression()      
simple_regressor.git(data[[x_feature]], data[[y_feature]])       
        
# 결과 회귀 계수 확인      
coef = simple_regressor.coef_     
print(coef)
```     
```
# 원본 데이터변수 산점도로 시각화    
plt.figure(figsize=(4, 4))     
plt.xlabel(x_feature)      
plt.ylabel(y_feature)     
plt.plot(data[[x_feature]], data[[y_feature]], '.')        
      
# 도식화할 x값 범위 설정    
x_min, x_max = -5, 5     
plt.xlim(x_min, x_max)         
      
# 회귀 직선식 플롯     
x = np.linspace(x_min, x_max, 10)      
y = coef.item()  # item()은 원소가 1개인 array를 단순 scalar로 변환할 때 사용        
plt.plot(x, y)     
      
plt.show()
```
