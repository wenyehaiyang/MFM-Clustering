import pandas as pd
from adjustText import adjust_text
from sklearn.cluster import OPTICS
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import davies_bouldin_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from k均值聚类k值确定 import confidence_ellipse

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def mfm_clustering_optics(ax, data_path, title, x_lim, y_lim, min_samples=2, legend=False):
    data = np.genfromtxt(data_path, delimiter=',', names=True)
    data_pd = pd.read_csv(data_path)
    data1 = np.array([[j for j in list(i)[1:]] for i in data])
    print(data1)
    row, col = data1.shape
    for r in range(row):
        for c in range(col):
            if np.isnan(data1[r][c]):
                data1[r][c] = 0
                data1[r][c] = np.nanmean(data1[:, c])

    # 使用OPTICS聚类
    optics = OPTICS(min_samples=min_samples)
    optics.fit(data1)
    y_optics = optics.fit_predict(data1)

    # Scale the data
    scaler = StandardScaler()
    scaled = scaler.fit_transform(data1)

    # Obtain principal components
    pca = PCA(n_components=2)

    pc = pca.fit_transform(scaled)
    pca.score_samples(scaled)
    # y_optics = optics.fit_predict(pc)
    print(pc.shape, y_optics)

    # 可视化结果
    colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
    # 可视化结果
    for i in range(0, np.unique(y_optics).size):
        color_n = colors[i]
        ax.scatter(pc[y_optics == (i - 1), 0], pc[y_optics == (i - 1), 1], s=50, c=color_n, label=f"source {i + 1}")
        confidence_ellipse(pc[y_optics == (i - 1), 0], pc[y_optics == (i - 1), 1], plt.gca(), n_std=2, edgecolor=color_n,
                           linewidth=2)
    texts = []
    for i, label in enumerate(data_pd['name']):
        # print(label, data1[i])
        if x_lim[1] > 1:
            if label.strip() in ['TCEP', 'T3CPP', 'TCIPP']:
                text = ax.text(
                    pc[i, 0], pc[i, 1], label,
                    # xytext=(2, 2),
                    # textcoords='offset points',
                    fontsize=8,
                    fontweight='normal',
                    color='black',
                    # bbox=dict(boxstyle='round,pad=0.5', fc='yellow', ec='k', alpha=0.7)
                )
                texts.append(text)
        else:
            if label.strip() not in ['TCEP', 'T3CPP', 'TCIPP']:
                text = ax.text(
                    pc[i, 0], pc[i, 1], label,
                    # xytext=(2, 2),
                    # textcoords='offset points',
                    fontsize=8,
                    fontweight='normal',
                    color='black',
                    # bbox=dict(boxstyle='round,pad=0.5', fc='yellow', ec='k', alpha=0.7)
                )
                texts.append(text)
        if data_path != "OPEs1":
            text = ax.text(
                pc[i, 0], pc[i, 1], label,
                # xytext=(2, 2),
                # textcoords='offset points',
                fontsize=8,
                fontweight='normal',
                color='black',
                # bbox=dict(boxstyle='round,pad=0.5', fc='yellow', ec='k', alpha=0.7)
            )
            texts.append(text)
    plt.xlim(x_lim)
    plt.ylim(y_lim)
    adjust_text(texts, arrowprops=dict(arrowstyle='->', color='red'))
    print('Davies-Bouldin score: ', davies_bouldin_score(data1, optics.labels_))
    plt.xlabel(f'PC1 {round(pca.explained_variance_ratio_[0] * 100, 2)} %')
    plt.ylabel(f'PC2 {round(pca.explained_variance_ratio_[1] * 100, 2)} %')
    plt.title(title, loc='left', ha='left', fontdict={'weight': 'bold', 'size': 16})
    if legend:
        ax.legend()
