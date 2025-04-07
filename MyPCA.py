import pandas as pd
from matplotlib import gridspec
from sklearn.cluster import KMeans, AgglomerativeClustering
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from adjustText import adjust_text
from k均值聚类k值确定 import k_SSE, k_silhouette, confidence_ellipse

# 中文和负号的正常显示
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def mfm_pca(ax, data, k, title="(a)"):
    data_pd = pd.read_csv(data)
    data = np.genfromtxt(data, delimiter=',', names=True)
    data1 = np.array([[j for j in list(i)[1:]] for i in data])
    print(data1)
    row, col = data1.shape
    for r in range(row):
        for c in range(col):
            if np.isnan(data1[r][c]):
                data1[r][c] = 0
                data1[r][c] = np.nanmean(data1[:, c])

    # 使用K均值算法聚类
    kmeans = KMeans(n_clusters=k, random_state=0)
    # K均值模糊聚类
    # kmeans = AgglomerativeClustering(n_clusters=k, linkage='complete', affinity='euclidean')
    # kmeans.fit(data1)
    # y_kmeans = kmeans.fit_predict(data1)

    # Scale the data
    scaler = MinMaxScaler()
    scaler.fit(data1)
    scaled = scaler.transform(data1)
    print(scaled)
    # Obtain principal components
    pca = PCA(n_components=3)
    pc = pca.fit_transform(scaled)
    y_kmeans = kmeans.fit_predict(scaled)
    loadings = pca.components_.T * 4  # 计算标准化载荷
    # y_kmeans = kmeans.fit_predict(pc)
    print(pc.shape, y_kmeans.shape)
    print(y_kmeans)
    colors = ["red", "blue", "gray", "green", "purple", "black"]
    # 获取载荷（主成分的方向）
    # loadings = pca.components_.T  # 转置是为了方便绘图，因为我们通常需要按特征绘制，而不是按样本
    print(loadings.shape[0])

    # 可视化结果
    ax.scatter(pc[:, 0], pc[:, 1], s=50, c=colors[1], label="Scores")
    texts = []
    for i in range(len(data_pd["name"])):
        text = ax.text(pc[i, 0], pc[i, 1], data_pd["name"][i], color=colors[1], fontsize=9)
        texts.append(text)
    # 绘制载荷箭头
    for i in range(loadings.shape[0]-2):  # 对于每个主成分
        ax.arrow(0, 0, loadings[i, 0], loadings[i, 1], color=colors[0], head_width=0.03, head_length=0.05, label="Explanatory Plot")
        text = ax.text(loadings[i, 0], loadings[i, 1], data_pd.columns.tolist()[1:][i], color=colors[0], fontsize=9)
        texts.append(text)
    '''for i in range(loadings.shape[0]-2, loadings.shape[0]):
        ax.arrow(0, 0, loadings[i, 0], loadings[i, 1], label="Explanatory Plot")
        text = ax.text(loadings[i, 0], loadings[i, 1], data_pd.columns.tolist()[1:][i], color=colors[3])
        texts.append(text)'''
    plt.ylim(-1, 1.4)
    plt.xlabel(f'PC1 {round(pca.explained_variance_ratio_[0] * 100, 2)} %')
    plt.ylabel(f'PC2 {round(pca.explained_variance_ratio_[1] * 100, 2)} %')
    # ax.legend()
    plt.title(title, loc='left', ha='left', fontdict={'weight': 'bold', 'size': 16})

    # confidence_ellipse(pc[y_kmeans == (i - 1), 0], pc[y_kmeans == (i - 1), 1], plt.gca(), n_std=2, edgecolor=color_n, linewidth=2)
    adjust_text(texts, expand=(1.1, 1.6), arrowprops=dict(arrowstyle='->', color='red'))

    # centers = kmeans.cluster_centers_
    # print("聚类中心：", centers)
    # plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75, marker='X')
    plt.xlabel(f'PC1 {round(pca.explained_variance_ratio_[0] * 100, 2)} %')
    plt.ylabel(f'PC2 {round(pca.explained_variance_ratio_[1] * 100, 2)} %')
    # ax.legend()
    plt.title(title, loc='left', ha='left', fontdict={'weight': 'bold', 'size': 16})


if __name__ == '__main__':
    plt.figure(figsize=(10, 6))
    gs = gridspec.GridSpec(1, 1)
    ax3 = plt.subplot(gs[0, 0])
    mfm_pca(ax3, '采样点m.csv', 3, title="(a)")
    plt.show()