import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, OPTICS
from sklearn import metrics


def test_modal_weight(X, ax, test_limit=(1e-4, 1e2), method='kmeans', cluster_k=4, min_samples=2, plot_title=''):
    """
    :return best_weight
    """
    weight_n = test_limit[0]
    w = []
    S = []
    X_ = X.copy()
    while weight_n < test_limit[1]:
        X_[:, -2:] = X[:, -2:] * weight_n
        if method == 'kmeans':
            kmeans = KMeans(n_clusters=cluster_k)
            kmeans.fit(X_)
            labels = kmeans.labels_
            S.append(metrics.silhouette_score(X_, labels, metric='euclidean'))
        else:
            kmeans = OPTICS(min_samples=min_samples)
            kmeans.fit(X_)
            labels = kmeans.labels_
            S.append(metrics.silhouette_score(X_, labels, metric='euclidean'))
        weight_n = weight_n * 2
        w.append(np.log10(weight_n))
    ax.plot(w, S, 'b*-')
    max_index = max(enumerate(S), key=lambda x: x[1])[0]
    ax.plot([w[max_index], w[max_index]], [min(S), max(S)], color='red')
    ax.text(w[max_index], (min(S) + max(S)) / 2, f'{round(10**w[max_index], 4)}',
            fontsize=12,fontweight='bold',color='purple')
    plt.xlabel('modal weight (scale by log10)')
    plt.title(plot_title, loc='left', ha='left', fontdict={'weight': 'bold', 'size': 16})
    plt.ylabel('Silhouette score')
    return 10**w[max_index]

def test_modal_weight_sse(X, ax, test_limit=(1e-4, 1e2), cluster_k=4, plot_title='', ok_sse = 0.6):
    """
    :return best_weight
    """
    weight_n = test_limit[0]
    w = []
    S = []
    while weight_n < test_limit[1]:
        X_ = X.copy()
        X_[:, -2:] = X[:, -2:] * weight_n
        X_ = pd.DataFrame(X_)
        SSE = []
        kmeans = KMeans(n_clusters=cluster_k)
        kmeans.fit(X_)
        # 返回簇标签
        labels = kmeans.labels_
        # 返回簇中心
        centers = kmeans.cluster_centers_
        # 计算各簇样本的离差平方和，并保存到列表中
        for label in set(labels):
            SSE.append(np.sum((X_.loc[labels == label,] - centers[label, :]) ** 2))
        S.append(np.sum(SSE))
        weight_n = weight_n * 2
        w.append(np.log10(weight_n))

    ax.plot(w, S, 'b*-')
    max_index = max(enumerate(S), key=lambda x: x[1])[0]
    ax.plot([ok_sse, ok_sse], [min(S), max(S)], color='red')
    ax.text(ok_sse, (min(S) + max(S)) / 2, f'{round(10**ok_sse, 4)}',
            fontsize=12,fontweight='bold',color='purple')
    plt.xlabel('modal weight (scale by log10)')
    plt.title(plot_title, loc='left', ha='left', fontdict={'weight': 'bold', 'size': 16})
    plt.ylabel('SSE within clusters')
    return 10**w[max_index]


if __name__ == '__main__':
    # 中文和负号的正常显示
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.rcParams['font.size'] = 15
    plt.rcParams['axes.unicode_minus'] = False
    # 设置边框粗细
    plt.rcParams['axes.linewidth'] = 1.5
    # 设置线条粗细
    plt.rcParams['lines.linewidth'] = 2
    ax = plt.gca()

    # 查看指标量
    data = np.genfromtxt('OPEs1.csv', delimiter=',', names=True)
    data1 = np.array([[j for j in list(i)[1:]] for i in data])
    row, col = data1.shape
    for r in range(row):
        for c in range(col):
            if np.isnan(data1[r][c]):
                data1[r][c] = 0
                data1[r][c] = np.nanmean(data1[:, c])
    test_modal_weight(data1, ax, test_limit=(1e-4, 1e2), method='optics', min_samples=2, plot_title='test')
    plt.show()