import pandas as pd
from sklearn.cluster import KMeans, OPTICS
import matplotlib.pyplot as plt
import numpy as np
from kmodes.kmodes import KModes
from kmodes.kprototypes import KPrototypes


from 小楠楠论文绘图.k均值聚类k值确定 import k_SSE


data = np.genfromtxt('OPEs.csv', delimiter=',', names=True)
print(data)
data1=np.array([np.array([j for j in i]) for i in data])
for r in data1.shape[0]:
    for c in data1[r]:
        if np.isnan(data1[r][c]):
            data1[r][c] = np.average(data1[r])
# data1=np.array([np.nansum(np.array([j for j in i])) for i in data])
# data1 = data1.reshape(-1, 1)
print(data1)

# 查看簇内离差和随k的关系图
k_SSE(pd.DataFrame(data1), 15)
# 使用K均值算法聚类
optics = KModes()
optics.fit(data1)
y_kmeans = optics.fit_predict(data1)

# 可视化结果
plt.scatter(data1[:, 0], data1[:, 1], c=y_kmeans, s=50, cmap='viridis')
# centers = spec.cluster_centers_
# plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75, marker='X')
plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.title('OPTICS Clustering')
plt.legend()
plt.show()
