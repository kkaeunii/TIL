# PCA   
* SVD 행렬분해 특성과 활용      
* Truncated SVD 활용한 행렬 rank 축소         
* PCA 구현       
* t-SNE 시각화     

## 환경설정       
```
import numpy as np       
from sklearn import datasets       
from sklearn.preprocessing import MinMaxScaler         
from sklearn.manifold import TSNE            
from sklearn.decomposition import PCA, TruncatedSVD       
       
import seaborn as sns       
import matplotlib.pyplot as plt          
from matplotlib import offsetbox
```    
## SVD 행렬분해       
**보조함수 정의**     
```
def plot_matrix(matrix, numbers=True, size_scale=0.7):         
    n_rows, n_cols = matrix.shape           
    figure_size = (size_scale*n_cols, size_scale*n_rows)           
    fig, ax = plt.subplots(figsize=figure_size)             
            
    # 불필요한 부분 비활성화        
    viz_args = dict(cmap='Purples', cbar=False, xticklabels=False, yticklabels=False)           
    sns.heatmap(data=matrix, annot=numbers, fmt='.2f', linewidths=.5, **viz_args)
```
**행렬 생성**   
```
# 랜덤 시드 고정    
np.random.seed(1234)     
       
# 배열 생성       
# 6x9 배열       
M = np.random.randn(6, 9)      
           
print(M)
```
```
# 보조함수 활용 시각화        
plot_matrix(M)
```
**SVD 구현**        
* 보조함수 정의    
```
def full_svd(matrix):                    
    U, singular_values, V = np.linalg.svd(matrix)            
              
    # svd 결과로 나온 sigma의 diagonal 성분으로 diagonal matrix 복원         
    m, n = matrix.shape  # matrix 행렬 차원        
    sigma = np.zeros([m,n])  # matrix 행렬과 같은 차원의 영행렬              
                    
    rank = len(singular_values)      
    sigma[:rank, :rank] = np.diag(singular_values)              
    return U, sigma, V.T
```
```
# SVD 수행          
U, Sigma, V = full_svd(M)
```    
* 원본 행렬 복원 
```
restored = U @ Sigma @ V.T                      
          
# 행렬 시각화     
plot_matrix(restored)       
print(np.abs(M- restored).max())
```    
**행렬분해 결과**       
```
print(U.shape, Sigma.shape, V.shapae)
```   
**U, V의 정규직교행렬로서 특성**          
```
# 행렬 U 시각화                 
plot_matrix(U)         
            
# 행렬의 서로 다른 row를 골라 내적하면 0이 나옴          
print(U[1] @ U[3])          
           
# 자기 자신과 내적은 항상 1        
print(U[1] @ U[1])     
            
# U와 U.T 시각화
plot_matrix(U @ U.T)
```       
**Sigma 특성**
```
# 특성 시각화로 확인    
plot_matrix(Sigma)      
          
# 행렬의 (k,k)번째 원소만 남기고 나머지를 0으로 만드는 함수 정의          
def select_diag(sigma, k):
    result = np.zeros_like(sigma)  # 영행렬 만들기         
    result[k,k] = sigma[k,k]  # sigma 행렬의 (k,k) 원소만 남김      
    return result                   
         
# 나머지 0으로 만들기            
sigma_k = select_diag(Sigma, 3)              
           
plot_matrix(sigma_k)
```    
* 원본 행렬과 비교    
```
# 원래 행렬 rank     
r = np.linalg.matrix_rank(M)                
           
# 결과 행렬을 미리 초기화        
result = np.zeros_like(M)          
for k in range(r):                   
    # Sigma_k를 계산 후 결과행렬에 더하기             
    sigma_k = select_diag(Sigma, k)          
    result += U @ sigma_k @ V.T            
                       
    print(np.abs(M - result).max())                
    plot_matrix(result)
```    
* 열벡터 계산     
```
# k번째 열벡터를 가져오는 함수 정의         
def col_vec(matrix, k):            
    return matrix[:, [k]]          
                    
    r = np.linalg.martrix_rank(M)            
                     
    result = np.zeros_like(M)
    for k in range(r):        
        # k번째 singular value ; 스칼라값          
        sig_k = Sigma[k,k]                   
        # U, V에서 한 개의 column vector만 가져와 사용해도 동일한 결과           
        result += sig_k * col_vec(U, k) @ col_vec(V, k).T                
                               
print(np.abs(M - result).max())              
plot_matrix(result)
```
* rank-1 matrix
```
# 임의의 k 선택           
k = 2       
# 곱해지는 U, Y의 열벡터를 표시              
print(col_vec(U, k).shape)         
print(col_vec(V, k).T.shape)                 
               
# rank-1 matrix          
rank1_mat = col_Vec(U, k) @ col_vec(V, k).T           
          
print(rank1_mat.shape)          
        
print(np.linalg.matrix_rank(rank1_mat))  # rank는 1로 나옴               
           
plot_matrix(rank1_mat)
```
## rank 축소 
```
def reduce_dim(M, n_components=None):        
    U, Sigma, V = full_svd(M)           
                   
    r = np.linalg.matrix_rank(M)           
    if n_components is None:                      
        n_components = r          
                      
    assert n_components <= r, \
        f'남길 components의 개수({n_components})는 전체 랭크{r}보다 클 수 없습니다.'       
                       
    result = np.zeros_like(M, dtype=np.float64)          
    # 첫 n_components개까지만 rank-1 matrix 더하기        
    for k in range(n_components):          
        sig_k = Sigma[k, k]               
        result += sig_k * col_vec(U, k) @ col_vec(V, k).T           
    return result
```
* 원래 행렬과 비교    
```
# 남길 성분의 수 ; 0~6까지 직접 조절        
n_components = 5       
size_scale = 0.6                    
          
# 원래 행렬 rank          
print(np.linalg.matrix_rank(M))           
              
# TruncatedSVD로 n_component개 남기기           
result = reduce_dim(M, n_components)              
                      
print(np.linalg.matrix_rank(result))              
print(np.abs(M - result).max())            
          
# 원래 행렬과 비교 시각화
plot_matrix(np.abs(M - result), size_scale=size_scale)       
plt.show()
``` 
## PCA 구현 
**데이터 불러오기**      
```
faces, _ = datasets.fetch_olivetti_faces(return_X_y=True, shuffle=True, random_state=1234)       
             
n_samples, n_features = faces.shape           
print(n_samples)         
print(n_features)
```     
* 이미지가 미리 가공된 상태로 제공됨        
* 이미지 각각을 flatten 하여 vector로 만들어 행렬로 만든 것      
```
# 데이터 범위 확인         
print(faces.mean())      
print(faces.max())       
print(faces.min())
```
**이미지 가져와 변환**       
```
# 가져올 샘플 번호를 랜덤으로 고르기         
index = np.random.choice(len(faces))            
print(index)              
             
# 원본 이미지 크기     
img_h, img_w = (64, 64)                
           
# 데이터에서 샘플 선택해 가져오기        
face_vector = faces[index]                
            
# 이미지를 원래 크기로 변환 후 보여주기          
face_image = face_vector.reshape(img_h, img_w)         
plt.imshow(face_image, cmap='색상')  # 보통 cmap = 'gray'
```
**전처리**      
```
# 전체 샘플 단위 평균을 구하여 원본 데이터에서 빼 평균 0으로 맞추기        
samplewise_mean = faces.mean(axis=0)         
faces_centered = faces - samplewise_mean          
              
# 각 이미지마다 픽셀값 평균 구하여 언본 이미지에서 빼 평균 0으로 맞추기          
pixelwise_mean = faces_centered.mean(axis=1).reshape(n_samples, -1)        
faces_centered -= pixelwise_mean
```
```
def plot_faces(title, images, n_cols=3, n_rows=2, shuffle=False, cmap='gray', size_scale=2.0, random_seed=0, image_shape=(64,64)):            
    if shuffle:       
        np.random.seed(random_seed)            
        indices = np.random.choice(len(images), n_cols * n_rows)         
    else:             
        indices = np.arange(n_cols * n_rows)                        
                     
    fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(n_cols * size_scale, n_rows * size_scale), facecolor='white', constrained_layout=True,)                 
    fig.set_constrained_layout_pads(w_pad=0.01, h_pad=0.02, hspace=0, wspace=0)        
    fig.set_edgecolor('black')             
    fig.suptitle(title, size=16)                  
                        
    for ax, idx in zip(axs.flat, indices):             
        face_vec = images[idx]           
        vmax = max(face_vec.max(), - face_vec.min())              
        im = ax.imshow(face_vec.reshape(image_shape), cmap=cmap, interpolation='nearest', vmin=-vmax, vmax=vmax,)              
        ax.axis('off')           
    fig.colobar(im, ax=axs, orientation='horizontal', shrink=0.99, aspect=40, pad=0.01)               
    plt.show()
```
```
# 이미지 6개 시각화      
plot_face(faces_centered, shuffle=True, random_seed=1234)
```
* PCA 수행
```
# 줄일 차원의 수 지정하기
n_components = 20

# PCA 수행하기
pca_estimator = PCA(n_components=n_components, svd_solver="full", whiten=True)
pca_estimator.fit(faces_centered)

# PCA 결과 (Eigenface) 시각화
plot_faces("Components of PCA", pca_estimator.components_, n_rows=2, n_cols=4)
```   
* 차원축소된 벡터 계산     
```
reduced_vec = pca_estimator.transform(faces_centered[index].reshape(1, -1))
print(reduced_vec)
print('차원 축소된 벡터의 크기:', reduced_vec.shape)
```   
* PCA된 component들을 위 reduced_vec 원소를 계수로 선형결합   
```
# 결과 행렬 미리 initialize
canvas = np.zeros([64, 64], dtype=np.float64)
for value, comp in zip(reduced_vec[0], pca_estimator.components_):
    # 각 component 벡터를 이미지 크기로 resize한 뒤, 이를 차원축소된 벡터의 각 값과 선형결합
    canvas += comp.reshape(64, 64) * value
```   
* 이미지 비교   
```
# 원본 이미지와 차원축소된 이미지들 비교하기
def compare_reduced_faces(title, images, index=123, n_components_list=[5, 20, 100], n_cols=4, n_rows=1, shuffle=False, cmap="gray", size_scale=2.5, random_seed=0, image_shape=(64, 64)):
    # 그림 관련 설정
    fig, axs = plt.subplots(
        nrows=n_rows, ncols=n_cols,
        figsize=(n_cols * size_scale, n_rows * size_scale),
        facecolor="white",
        constrained_layout=True,
    )
    fig.set_constrained_layout_pads(w_pad=0.01, h_pad=0.02, hspace=0, wspace=0)
    fig.set_edgecolor("black")

    # 보여줄 이미지 선정
    face_vec = faces[index]

    # 첫 이미지로 원본 이미지를 보여줍니다.
    axs[0].set_title("Original Face Image", y=-0.2)
    axs[0].imshow(face_vec.reshape(image_shape), cmap="gray")
    axs[0].axis("off")

    # 다음 이미지부터는 PCA를 이용해 차원축소된 이미지를 보여줍니다.
    # 각 차원마다 보여주므로 줄일 차원의 수 리스트 중 하나씩 지정하여 PCA를 수행합니다.
    for img_index, n_components in enumerate(n_components_list):

        # PCA 수행하기
        pca_estimator = PCA(n_components=n_components, svd_solver="full", whiten=True)
        pca_estimator.fit(images)

        # 차원축소된 벡터 계산하기
        reduced_vec = pca_estimator.transform(face_vec.reshape(1, -1))
        # 결과 행렬 미리 initialize
        canvas = np.zeros([64, 64], dtype=np.float64)
        for value, comp in zip(reduced_vec[0], pca_estimator.components_):
            # 각 component 벡터를 이미지 크기로 resize한 뒤, 이를 차원축소된 벡터의 각 값과 선형결합
            canvas += comp.reshape(64, 64) * value

        # PCA 결과 (Eigenface) 시각화
        vmax = max(canvas.max(), - canvas.min())
        im = axs[img_index+1].imshow(
            canvas.reshape(image_shape),
            cmap=cmap,
            interpolation="nearest",
            vmin=-vmax,
            vmax=vmax,
        )
        axs[img_index+1].axis("off")
        axs[img_index+1].set_title(f'Dimension={n_components}', y=-0.2)

    # 최종 이미지 보여주기
    plt.suptitle(title + f': images at index {index}', fontsize=20)
    plt.show()                          
                     
compare_reduced_faces('Comparisons of different dimensions', faces_centered, n_components_list=[5, 20, 100])
```      
## t-SNE 
* 데이터 불러오기       
* 데이터 시각화     
**데이터 불러오기**    
```
# 데이터 불러오기     
digits = datasets.load_digits(n_class=6)
X, y = digits.data, digits.target
n_samples, n_features = X.shape
print(n_samples) # 데이터 수 확인
print(n_features) # 차원수 확인
```
* 숫자 이미지 시각화     
```
fig, axs = plt.subplots(nrows=10, ncols=10, figsize=(5, 5))
for idx, ax in enumerate(axs.ravel()):
    ax.imshow(X[idx].reshape((8, 8)), cmap=plt.cm.binary)
    ax.axis("off")
_ = fig.suptitle("A selection from the 64-dimensional digits dataset", fontsize=16)
```     
* plot helper 함수 정의
```
def plot_embedding(X, title):
    _, ax = plt.subplots()
    # 정규화
    X = MinMaxScaler().fit_transform(X)
    # 색깔로 숫자로 scatter 표시
    for digit in digits.target_names:
        ax.scatter(
            *X[y == digit].T,
            marker=f"${digit}$",
            s=60,
            color=plt.cm.Dark2(digit),
            alpha=0.425,
            zorder=2,
        )
    # 이미지 그림 표시
    shown_images = np.array([[1.0, 1.0]])
    for i in range(X.shape[0]):
        # 모든 숫자 임베딩을 scatter하고, 숫자 그룹에 annotation box를 보기
        dist = np.sum((X[i] - shown_images) ** 2, 1)
        # 보기 쉽게 하기 위해 너무 가까운 데이터는 보여주지 않기
        if np.min(dist) < 4e-3:
            continue
        # 이미지 합치기
        shown_images = np.concatenate([shown_images, [X[i]]], axis=0)
        imagebox = offsetbox.AnnotationBbox(
            offsetbox.OffsetImage(digits.images[i], cmap=plt.cm.gray_r), X[i]
        )
        imagebox.set(zorder=1)
        ax.add_artist(imagebox)

    ax.set_title(title)
    ax.axis("off")
```     
**t-SNE 이용해 2차원 시각화**   
```
# t-SNE 적용
transformer = TSNE(n_components=2, random_state=0)
projection = transformer.fit_transform(X, y)

# t-SNE 결과 시각화
plot_embedding(projection, 't-SNE embedding')
plt.show()
```   
**TruncatedSVD 적용 시각화**
```
# Truncated SVD 적용
transformer = TruncatedSVD(n_components=2)
projection = transformer.fit_transform(X, y)

# TruncatedSVD 결과 시각화
plot_embedding(projection, 'TruncatedSVD embedding')
plt.show()
```
