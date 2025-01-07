from sklearn.cluster import OPTICS, SpectralClustering
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from 小楠楠论文绘图.k均值聚类k值确定 import confidence_ellipse

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

data = np.genfromtxt('采样点m.csv', delimiter=',', names=True)
data_pd = pd.read_csv("采样点m.csv")
data1 = np.array([[j for j in list(i)[1:]] for i in data])

print(data1)
row, col = data1.shape
for r in range(row):
    for c in range(col):
        if np.isnan(data1[r][c]):
            data1[r][c] = 0
            data1[r][c] = np.nanmean(data1[:, c])

# 使用SpectralClustering聚类
spec = SpectralClustering(n_clusters=3, random_state=1)
y_spec = spec.fit_predict(data1)

# Scale the data
scaler = StandardScaler()
scaler.fit(data1)
scaled = scaler.transform(data1)

# Obtain principal components
pca = PCA(n_components=2)
pc = pca.fit_transform(scaled)
pca.score_samples(scaled)
# y_spec = spec.fit_predict(pc)
print(pc.shape, y_spec)
# 可视化结果
colors = ["red", "blue", "gray", "green", "purple", "black", "wheat"]
# 可视化结果
for i in range(0, np.unique(y_spec).size):
    color_n = colors[i]
    plt.scatter(pc[y_spec == i, 0], pc[y_spec == i, 1], s=50, c=color_n, label=f"source {i + 1}")
    confidence_ellipse(pc[y_spec == i, 0], pc[y_spec == i, 1], plt.gca(), n_std=2, edgecolor=color_n, linewidth=2)
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
plt.xlabel(f'PC1 {round(pca.explained_variance_ratio_[0] * 100, 2)} %')
plt.ylabel(f'PC2 {round(pca.explained_variance_ratio_[1] * 100, 2)} %')
plt.title('SpectralClustering Clustering')
plt.legend()
plt.show()
