import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
# from adjustText import adjust_text
from 小楠楠论文绘图.k均值聚类k值确定 import k_SSE, k_silhouette, confidence_ellipse

N_CLUSTERS = 4

data = np.genfromtxt('参考数据.csv', delimiter=',', names=True)
data_pd = pd.read_csv("参考数据.csv")
data1 = np.array([[j for j in list(i)[1:]] for i in data])

print(data1)
row, col = data1.shape
for r in range(row):
    for c in range(col):
        if np.isnan(data1[r][c]):
            data1[r][c] = 0
            data1[r][c] = np.nanmean(data1[:, c])

# 查看簇内离差和随k的关系图
k_SSE(pd.DataFrame(data1), 15, ax=plt.gca(), ok_k=3)
k_silhouette(pd.DataFrame(data1), 15, ax=plt.gca(), ok_k=3)

# 使用K均值算法聚类
kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=0)
# K均值模糊聚类
# kmeans = AgglomerativeClustering(n_clusters=N_CLUSTERS, linkage='complete', affinity='euclidean')
#kmeans.fit(data1)
#y_kmeans = kmeans.fit_predict(data1)

# Scale the data
scaler = MinMaxScaler()
scaler.fit(data1)
scaled = scaler.transform(data1)
print(scaled)
# Obtain principal components
pca = PCA(n_components=2)
pc = pca.fit_transform(scaled)
y_kmeans = kmeans.fit_predict(scaled)
# y_kmeans = kmeans.fit_predict(pc)
print(pc.shape, y_kmeans.shape)
print(y_kmeans)
colors = ["red", "blue", "gray", "green", "purple", "black"]
# 可视化结果
for i in range(1, N_CLUSTERS + 1):
    color_n = colors[i - 1]
    plt.scatter(pc[y_kmeans == (i - 1), 0], pc[y_kmeans == (i - 1), 1], s=50, c=color_n, label=f"source {i}")
    confidence_ellipse(pc[y_kmeans == (i - 1), 0], pc[y_kmeans == (i - 1), 1], plt.gca(),
                        n_std=2, edgecolor=color_n, linewidth=2)
texts = []
for i, label in enumerate(data_pd['name']):
    # print(label, data1[i])
    text = plt.annotate(
        label, (pc[i, 0], pc[i, 1]),
        xytext=(2, 2),
        textcoords='offset points',
        fontsize=8,
        fontweight='normal',
        color='black',
        # bbox=dict(boxstyle='round,pad=0.5', fc='yellow', ec='k', alpha=0.7)
    )
    texts.append(text)
# adjust_text(texts, expand=(1.2, 2), arrowprops=dict(arrowstyle='->', color='red'))

# centers = kmeans.cluster_centers_
# print("聚类中心：", centers)
# plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75, marker='X')
plt.xlabel(f'PC1 {round(pca.explained_variance_ratio_[0] * 100, 2)} %')
plt.ylabel(f'PC2 {round(pca.explained_variance_ratio_[1] * 100, 2)} %')
plt.legend()
plt.title('K-Means Clustering')
plt.show()
