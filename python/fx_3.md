# 분류문제 모델링 실습      
* 실습 데이터셋 준비      
* 로지스틱 회귀      
* 결정트리      
* 서포트 벡터 머신         

## 환경설정
```
import pandas as pd    
import numpy as np    
import matplotlib.pyplot as plt       
      
# 데이터 관련     
from sklearn.model_selection import train_test_split      
from sklearn.datasets import load_breast_cancer # 사용할 데이터셋     
      
# SVM 실습 관련     
from sklearn.datasets import make_blobs      
from sklearn.datasets import make_circles     
        
# 로지스틱 회귀모델      
from sklearn.linear_model import LogisticRegression     
     
# 결정트리 모델과 시각화 관련       
import graphviz     
from sklearn.tree import export_graphviz      
from sklearn.tree import DecisionTreeClassifier       
       
# SVM 모델 
from sklearn.svm import SVC       
from sklearn.gaussian_process.kernels import RBF
```
## 실습 데이터셋 준비
**데이터셋 불러오고 설명 확인**
```
cancer = load-breast_cancer()      
print(cancer['DESCR'])
``` 
**데이터셋 기본 정보 확인**
* feature의 전체 수량과 종류 확인     
* target의 전체 수량과 종류 확인    
* target에서 각 범주별 빈도 확인   
```
# 데이터셋 샘플과 변수 개수 확인     
print(cancer['data'].shape)        
      
# 데이터셋 feature 이름 확인
print(cancer['feature_names'])
```
```
# 데이터셋 target 확인      
print(cancer['target'].shape)        
    
# target 클래스 확인       
print(cancer['target_names']
```
```
# 데이터셋 target 보기     
cancer['target']       
     
# target 내 샘플 수 확인      
count_malignant, count_benign = np.bincount(cancer['target'])
print(f'악성 샘플 : {count_malignant})
print(f'양성 샘플 : {count_benign})
```    
**데이터 시각화 및 특성 파악**         
* 코드 단순화 위한 변수 작업
```
# alias(별명) 설정         
data = cancer['data']      
target = cancer['target']       
            
# 악성과 양성에 해당하는 샘플만 저장       
data_malignant = data[target == 0]        
data_benign = data[target == 1]          
            
# 데이터셋 분리 결과 확인        
print(data_malignant.shape)      
print(data_benign.shape)
```    
```
# 시각화에 사용할 변수의 인덱스 지정    
feature_idx = 0       
        
# 히스토그램 형태로 시각화     
plt.his(data_malignant[:, feature_idx], bins=20, alpha=0.3)       
plt.his(data_benign[:, feature_idx], bins=20, alpha=0.3)         
        
# 데이터셋 정보를 그래프에 추가       
plt.title(cancer['feature_names'][feature_idx])            
plt.legend(cancer['target_names'])
```
```
# 전체 변수 시각화        
plt. figure(figsize=[20,15])      
for feature_idx in range(30):       
    plt.subplot(6, 5, feature_idx + 1)             
                  
    # 히스토그램으로 악성, 양성 샘플 분포 시각화        
    plt.his(data_malignant[:, feature_idx], bins=20, alpha=0.3)            
    plt.his(data_benign[:, feature_idx], bins=20, alpha=0.3)              
                
    # 데이터셋 정보를 그래프에 추가         
    plt.title(cancer['타이틀 명'][feature_idx])         
    # 범례는 첫번째 히스토그램만 표시       
    if feature_idx == 0:        
        plt.legend(cancer['target_names'])          
    plt.xticks([])
```   

## 로지스틱 회귀     
**로지스틱 회귀모델 학습**   
```
# 재현성을 위한 random seed 설정       
random_state = 1234        
          
# 데이터셋 8:2 분할         
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, test_size=0.2, random_state=random_state)            
       
# 모델 초기화 및 최적화      
model = LogisticRegression(max_iter=5000)         
model.fit(X_train, y_train)          
         
# 테스트셋 정확도 계산        
score = model.score(X_test, y_test)         
        
print(score)
```
**출력값 조정**        
* 1종오류 방지를 위함        
```
# 학습된 모델에서 결과 확률값 가져오기      
probs = model.predict_proba(X_test)[:, 1]              
      
# 기본값인 0.5를 기준으로 판단한 결과는 원래 모델 예측 함수와 동일    
print('원래 예측값 : \n', model.predict(X_test))          
prediction = (probs > 0.5).astype(int)        
print(f'한계값 0.5로 판단한 예측값 : \n', prediction)        
            
# y_test == 0(악성, positive)지만 prediction == 1(양성, negative)인 False Negative 계산      
false_neg = ~y_test & prediction      
print(f'위음성(False Negative) 개수 : {false_neg.sum()}')          
          
# 한계값 조절해 위음성 빈도 줄이기    
threshold = 0.7        
           
prediction = (probs > threshold).astype(int)       
print(f'한계값 {threshold}로 판단한 예측값 : \n', prediction)        
      
flase_neg = ~y_test & prediction        
print(f'위음성 개수 : {flase_neg.sum()})
```    

## 결정트리 
**결정트리 학습과 출력**
```
# 학습     
dec_tree = DecisionTreeClassifier(max_depth=10, random_state=1234)      
dec_tree.fit(X_train, y_train)          
       
# 결과 출력       
print(f'학습 데이터셋 분류 정확도 : {dec_tree.score(X_train, y_train): .3f}')        
print(f'평가 데이터셋 분류 정확도 : {dec_tree.score(X_test, y_test): .3f}')
```
**결정트리 시각화**
* export_graphviz 함수 사용       
```
# 학습한 그래프 파일 저장      
export_graphviz(dec_tree, out_file='tree.dot', class_names=cancer['target_names'], feature_names=cancer['feature_names'], impurity=True, filled=True)             
              
# graphviz로 시각화        
with open('tree.dot') as f:         
    dot_graph = f.read()       
display(graphviz.Source(dot_graph))
```
**feature importance 확인**       
```
print(dec_tree.feature_importances_)
```
```
n_features = data.shape[1]          
plt.barh(np.arange(n_features), dec_tree.feature_importanes_, align='center')        
plt.yticks(np.arange(n_features), cancer['feature_names'])          
plt.xlabel('x라벨 명')        
plt.ylabel('y라벨 명')            
plt.ylim(-1, n_features)
```
* 시각화    
```
# 시각화 변수 지정     
feature_name = 'worst concave points'  # feature importance 높은 node       
feature_threshold = 0.142  # 임의로 지정 가능     
# feature_name = 'worst fractal dimension'  # feature importance 낮은 node     
# feature_threshold = 0.065     
        
# 변수 이름으로 index 찾고 feature importance 값 출력      
list_feature_names = cancer['feature_names'].tolist()      
feature_idx = list_feature_names.index(feature_name)     
print(dec_tree.feature_importances_[feature_idx])       
          
# 히스토그램으로 샘플 분포 시각화      
plt.hist(data_malignant[:, feature_idx], bins=20, alpha=0.3)      
plt.his(data_benign[:, feature_idx], bins=20, alpha=0.3)          
             
# 데이터셋 정보 그래프에 추가      
plt.axvline(feature_threshold)      
plt.title(cancer['feature_names'][feature_idx])        
plt.legend(['threshold'] + list(cancer['target_names']))             
plt.show()
```    
## SVM    
**합성 데이터셋 생성**     
```
# 재현성을 위한 랜덤 시드 고정     
random_state = 20         
          
# 합성 데이터 생성      
# n_sample은 샘플 수, center은 클러스터 수인데 이진분류라서 2, cluster_std는 샘플 표준편차        
X, y = make_blobs(n_samples=100, centers=2, cluster_std=1.2, random_state=random_state)                
         
plt.scatter(X[:,0], X[:,1], c=y, s=30)           
plt.title('타이틀 명')         
plt.show()
```
**SVM 분류모델 학습**         
```
# 보조 함수 정의         
def make_xy_grid(xlim, ylim, n_points):        
    # x, y 각각 일정 간격으로 변화하는 grid 생성           
    xx = np.linspace(*xlim, n_points)           
    yy = np.linspace(*ylim, n_points)        
    YY, XX = np.meshgrid(yy,xx)            
                        
    # gird 위의 900개 점 좌표를 순서로 나타낸 array           
    xy = np.stack([XX.reshape(-1), YY.reshape(-1)], axis=1)        
    return XX, YY, xy
```   
* 학습 및 시각화      
```
# 학습      
clif = SVC(kernel = 'linear', C=1.0)         
clif.fit(X,y)               
           
# 데이터셋 시각화      
plt.scatter(X[:,0], X[:,1], c=y, s=30, cmap=plt.cm.Paired)          
      
# x와 y 값 범위 확인       
ax = plt.gca()       
xlim = ax.get_xlim()          
ylim = ax.get_ylim()         
          
# 결정경계          
XX, YY, xy = make_xy_grid(xlim, ylim, 30)            
Z = clf.decision_function(xy).reshpae(XX.shape)           
           
# 결정 경계와 마진 시각화           
ax.contour(XX, YY, Z, colors='s', levels=[[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])         
         
# 서포트 벡터 표시  
ax.scatter(clif.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100, linewidth=1, facecolors='none', edgecolors='k')        
plt.title('타이틀 명')           
plt.show()
```    
**선형분리 불가능한 데이터셋과 커널함수**      
* 데이터 생성             
```
# 데이터 생성     
X,y = make_circles(factor=0.1, noise=0.1)  # factor : 생성할 원의 반지름 비율    
      
# 데이터 시각화     
plt.figure(figsize=(4,4))     
plt.scatter(X[:,0], X[:,1], c=y, s=30, cmap=plt.cm.Paired)        
plt.title('타이틀 명')         
plt.show()
```     
* 커널함수 적용        
```
# RBF 커널함수 적용      
z = RBF(1.0).__call__(X)[0]      
        
# 3D 공간에 커널함수 적용 및 시각화       
fig = plt.figure()         
ax = fig.add_subplot(111, projection='3d')          
        
ax.scatter(X[:,0], X[:,1], z, c=y, s=30, cmap=plt.cm.Paired)        
         
ax.set_xlabel('x라벨 명')        
ax.set_ylabel('y라벨 명')       
ax.set_zlabel('z라벨 명')           
             
plt.title('타이틀 명')        
plt.show()             
plt.clf()
```      
* 학습       
```
def plot_svc_decision_function(model, ax=None):          
    if ax is None:              
        ax = plt.gca()             
    xlim = ax.get_xlim()           
    ylim = ax.get_ylim()           
                     
    x = np.linspace(xlim[0], xlim[1], 30)       
    y = np.linspace(ylim[0], xlim[1], 30)         
    Y,X = np.meshgrid(y,x)        
    xy = np.vstack([X.ravel(), Y.ravel()]).T       
    P = model.decision_function(xy).reshape(X.shape)              
                   
    # 결정경계 시각화         
    ax.contour(X, Y, P, colors='k', levels=[-1,0,1], alpha=0.5, linestyles=['--','-','--'])           
    ax.set_xlim(xlim)          
    ax.set_ylim(ylim)           
                  
# 데이터셋 호출         
X,y = make_circles(factor=0.2, noise=.1)  # factor = R2/R1, nosie = std                    
# 커널트릭으로 SVC 학습         
clf = SVC(kernel='rbf').fit(X,y)          
                 
plt.scatter(X[:,0], X[:,1], c=y, s=30, cmap=plt.cm.Paired)           
plot_svc_decision_function(clf)          
plt.title('타이틀 명')           
plt.show() 
```
