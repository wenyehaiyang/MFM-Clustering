import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralCoclustering
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from adjustText import adjust_text
from 小楠楠论文绘图.k均值聚类k值确定 import k_SSE, k_silhouette, confidence_ellipse

# 中文和负号的正常显示
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


data_pd = pd.read_csv('采样点m.csv')
data = np.genfromtxt('采样点m.csv', delimiter=',', names=True)
data1 = np.array([[j for j in list(i)[1:]] for i in data])
row, col = data1.shape
for r in range(row):
    for c in range(col):
        if np.isnan(data1[r][c]):
            data1[r][c] = 0
            data1[r][c] = np.nanmean(data1[:, c])

ax = plt.subplot(1, 2, 1)
ax.matshow(data1, cmap=plt.cm.Blues)
plt.title("聚类前")

# Scale the data
scaler = MinMaxScaler()
scaler.fit(data1)
scaled = scaler.transform(data1)

# Obtain principal components
pca = PCA(n_components=2)
pc = pca.fit_transform(scaled)


model = SpectralCoclustering(n_clusters=3, random_state=0)
model.fit(data1)
print(model.rows_[0])
print("*******************************************")
print(model.rows_[1])

fit_data = data1[np.argsort(model.row_labels_)]
print(fit_data)
fit_data = fit_data[:, np.argsort(model.column_labels_)]

ax = plt.subplot(1, 2, 2)
ax.matshow(fit_data, cmap=plt.cm.Blues)
plt.title("After biclustering; rearranged to show biclusters")
plt.xlabel(f'PC1 {round(pca.explained_variance_ratio_[0] * 100, 2)} %')
plt.ylabel(f'PC2 {round(pca.explained_variance_ratio_[1] * 100, 2)} %')
plt.show()

