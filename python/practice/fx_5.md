# Clustering    
* 데이터셋 준비        
* k-means 군집화 방법론       

## 환경설정      
```
import numpy as np          
import pandas as pd       
         
# k-means clustering         
from sklearn import datasets          
from sklearn.preprocessing import MinMaxScaler         
from sklearn.metrics import pairwise_distances         
from sklearn.cluster import KMeans          
             
import matplotlib.pyplot as plt          
import plotly.express as px
```
## 데이터 준비          
* iris 데이터셋 활용     
```
# 데이터 불러오기     
iris = datasets.load_iris()        
         
# input, ouput을 X, y 변수로 할당        
X = iris.data         
y = iris.target         
            
print(X.shape)        
print(y.shape)
```    
* 데이터 전처리       
```
# 정규화          
scaler = MinMaxScaler()        
X = scaler.fit_transform(X)
```  
## k-means clustering       
**초기 군집 설정**     
```
# 랜덤 시드 고정      
np.random.seed(1234)          
           
# 군집화할 군집 수       
n_cluster = 3        
             
# 랜덤으로 초기 군집 중심 설정          
init_idx = np.random.randint(X.shape[0], size=n_cluster)         
init_data = X[init_idx, :]
```
**시각화**      
```
# 정규화된 데이터 시각화           
plt.figure(figsize=(5,4))           
plt.scatter(X[:, 0], X[:, 1], c='CO', label='라벨명', edgecolor='k')            
                
# 초기 군집 중심 시각화           
plt.scatter(init_data[:, 0], init_data[:, 1], edgecolor='k', c=['C1', 'C2', 'C3'], label='라벨명', marker='*', s=100)              
plt.legend()                 
plt.show()
```     
**군집 중심으로 초기 데이터 할당**            
```
# 군집 초기화         
iter = 0              
                 
# 군집 중심을 설정한 초기 데이터로 정의          
centroid = init_data             
                      
# step 1: 각 데이터 포인트에서 각 군집 중심까지의 거리를 계산 후, 가장 가까운 군집중심을 가진 군집으로 할당
diff = X.reshape(-1, 1, 2) - centroid
distances = np.linalg.norm(diff, 2, axis=-1)
clusters = np.argmin(distances, axis=-1)
```
```
# 군집화 결과 시각화 도움 함수
def plot_cluster(X, clusters, centroid, iter=0):
    plt.figure(figsize=(5, 4))
    # 군집화된 데이터
    plt.scatter(X[:, 0], X[:, 1], c=clusters, cmap=plt.cm.Set1, edgecolor="k", label='data')
    # 군집의 중심 시각화
    plt.scatter(centroid[:, 0], centroid[:, 1], marker='*', s=100, edgecolor="k", c=['C1','C2','C3'], label='centroid')
    plt.title(f'Clustering results (iter={iter})')
    plt.legend()
    plt.show()
```
```
# 군집화 결과 시각화           
plot_cluster(X, clusters, centroid, iter)
```
**군집화 과정 반복**
```
iter += 1  # 다음 iteration

# step2: 군집의 중심을 각 군집에 할당된 데이터들의 평균으로 정의 (mu_k)
clusters = np.array(clusters)
centroid = np.array([
    X[clusters == 0, :].mean(axis=0),
    X[clusters == 1, :].mean(axis=0),
    X[clusters == 2, :].mean(axis=0),
])

# step 1: 각 데이터 포인트에서 각 군집 중심까지의 거리를 계산 후, 가장 가까운 군집중심을 가진 군집으로 할당 (r_ik)
diff = X.reshape(-1, 1, 2) - centroid
distances = np.linalg.norm(diff, 2, axis=-1)
clusters = np.argmin(distances, axis=-1)

# 결과를 시각화
plot_cluster(X, clusters, centroid, iter)
```
```
# step3: step2 반복: iteration 3~11
fig = plt.figure(figsize=(15, 10))
for iter in range(6):
    # 군집의 중심을 각 군집에 할당된 데이터들의 평균으로 정의 (mu_k)
    clusters = np.array(clusters)
    centroid = np.array([
        X[clusters == 0, :].mean(axis=0),
        X[clusters == 1, :].mean(axis=0),
        X[clusters == 2, :].mean(axis=0),
    ])

    # step 1 반복: 각 데이터 포인트와 군집의 중심과의 거리를 계산하여 가장 가까운 군집으로 할당 (r_ik)
    diff = X.reshape(-1, 1, 2) - centroid
    distances = np.linalg.norm(diff, 2, axis=-1)
    clusters = np.argmin(distances, axis=-1)

    # 결과 시각화
    ax = fig.add_subplot(2, 3, iter + 1)
    # 군집화된 데이터 시각화
    ax.scatter(X[:, 0], X[:, 1], c=clusters, cmap=plt.cm.Set1, edgecolor="k")
    # 군집의 중심 시각화
    plt.scatter(centroid[:, 0], centroid[:, 1], marker='*', s=100, edgecolor="k", c=['C1','C2','C3'], label='centroid')

    ax.set_title(f'Clustering results (iter={iter + 3})')

plt.show()
```
* plotly로 시각화
```
def k_means_clustering(data, n_cluster, num_iter):
    '''
    k-means clustering 구현 및 결과값 저장
    '''
    result = {}
    # 반복횟수만큼 반복
    for iter in range(0, num_iter + 1):
        # 군집의 중심 정의
        if iter == 0:
            # 초기화: 첫번째 iteration에서는 군집의 중심을 위에서 설정한 초기 데이터로 정의 (mu_k)
            np.random.seed(0)
            init_idx = np.random.randint(data.shape[0], size=n_cluster)
            centroid = data[init_idx, :]
        else:
            # step2: 군집의 중심을 각 군집에 할당된 데이터들의 평균으로 정의 (mu_k)
            clusters = np.array(clusters)
            centroid = np.array([
                data[clusters == 0, :].mean(axis=0),
                data[clusters == 1, :].mean(axis=0),
                data[clusters == 2, :].mean(axis=0),
            ])

        # step 1: 각 데이터 포인트와 군집의 중심과의 거리를 계산하여 가장 가까운 군집으로 할당 (r_ik)
        diff = X.reshape(-1, 1, 2) - centroid
        distances = np.linalg.norm(diff, 2, axis=-1)
        clusters = np.argmin(distances, axis=-1)

        # clustering 결과를 dictionary에 iteration index 별로 저장
        result[iter] = {
            'clusters': clusters,
            'centroid': centroid,
        }
    return result
```
```
# iris 데이터 X에 대해 k-means clustering 수행
result = k_means_clustering(X, n_cluster=3, num_iter=10)
```
```
# plotly 버전
def plotly_results(data, result):

    # 데이터 전처리
    df = pd.DataFrame()
    for idx, res in result.items():

        # 각 데이터의 군집 인덱스
        cluster = pd.DataFrame(data, columns=['X','Y'])
        cluster['cluster_idx'] = res['clusters']
        cluster['iter'] = idx
        cluster['label'] = 'data'
        cluster['label_size'] = 0.1
        df = pd.concat([df, cluster], axis=0)

        # 군집 중심 데이터
        centroid = pd.DataFrame(res['centroid'], columns=['X','Y'])
        centroid.reset_index(inplace=True)
        centroid.columns = ['cluster_idx']+list(centroid.columns[1:])
        centroid['iter'] = idx
        centroid['label'] = 'cluster_centroid'
        centroid['label_size'] = 1
        df = pd.concat([df, centroid], axis=0)

    # 애니메이션 시각화
    fig = px.scatter(df, x="X", y="Y", animation_frame="iter", color="cluster_idx", size='label_size', symbol='label', width=1000, height=800, symbol_sequence= ['circle', 'star'])
    fig.update_coloraxes(showscale=False)
    fig.show()

plotly_results(X, result)
```
### 사이킷런 활용 k-means 군집화 
```
# k-means clustering 수행하기
kmeans_clustering = KMeans(n_clusters=n_cluster, n_init="auto")
kmeans_clustering.fit(X)

# k-means clustering 에 의해 정의된 군집(cluster)과 군집의 중심(centroid)
clusters = kmeans_clustering.labels_
centroid = kmeans_clustering.cluster_centers_
```
```
# 시각화
fig = plt.figure(figsize=(10, 4))

# ground truth 시각화
ax = fig.add_subplot(1, 2, 1)
for name, label in [("Setosa", 0), ("Versicolour", 1), ("Virginica", 2)]:
    ax.text(
        X[y == label, 0].mean(),
        X[y == label, 1].mean() + 0.15,
        name,
        horizontalalignment="center",
        bbox=dict(alpha=0.2, edgecolor="w", facecolor="w"),
    )
# Reorder the labels to have colors matching the cluster results
y = np.choose(y, [2, 0, 1]).astype(float)
ax.scatter(X[:, 0], X[:, 1], c=y, edgecolor="k", cmap=plt.cm.Set1)

ax.set_xlabel("Sepal length")
ax.set_ylabel("Sepal width")
ax.set_title("Ground Truth")

# clustering 결과 시각화
ax = fig.add_subplot(1, 2, 2)
# 군집화된 데이터 시각화
ax.scatter(X[:, 0], X[:, 1], c=np.array(clusters).astype(float), edgecolor="k")
# 군집의 중심 시각화
plt.scatter(centroid[:, 0], centroid[:, 1], marker='*', s=200, edgecolor="k", c=['C1','C2','C3'], label='centroid')
ax.set_xlabel("Sepal length")
ax.set_ylabel("Sepal width")
ax.set_title("Clustering result")

plt.subplots_adjust(wspace=0.25, hspace=0.25)
plt.show()
```      
**랜덤 합성데이터 군집화**         
```
# 재현성을 위한 랜덤시드 설정
np.random.seed(123)

# 실제 데이터의 군집 수와 k-means에서 지정할 군집의 수
n_centers = 5
n_clusters = 3

# 데이터 생성
X, Y = datasets.make_blobs(n_features=2, centers=n_centers)
```
```
# 인풋 데이터와 라벨 시각화
plt.figure(figsize=(10, 4))
plt.subplot(121)
plt.title("Three blobs", fontsize="small")
plt.scatter(X[:, 0], X[:, 1], marker="o", c=Y, s=25, edgecolor="k")

# k-means 군집화 수행
kmeans_clustering = KMeans(n_clusters=n_clusters, n_init="auto")
kmeans_clustering.fit(X)
clusters = kmeans_clustering.labels_

# 군집화 결과 시각화
plt.subplot(122)
plt.title('Clustering result', fontsize='small')
plt.scatter(X[:, 0], X[:, 1], marker="o", c=clusters, s=25, edgecolor="k")

plt.show()
```
```
# 군집수(n_cluster)를 변화시켜가며 k-means clustering 수행
results = []
for n in range(2, 10):
    kmeans_clustering = KMeans(n_clusters=n, n_init="auto")
    kmeans_clustering.fit(X)
    results.append(kmeans_clustering.inertia_)  # inertia_: Sum of squared distances of samples to their closest cluster center, weighted by the sample weights if provided.
```
```
# 군집수(n_cluster)를 변화시켜가며 k-means clustering 을 수행한 결과의 거리 시각화
plt.plot([*range(2, 10)], results, '-o')
plt.xlabel('number of clusters')
plt.ylabel('Within Cluster Sum of Square')
plt.title('Elbow method')
plt.show()
```
