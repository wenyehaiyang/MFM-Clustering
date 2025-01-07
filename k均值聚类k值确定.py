# 导入第三方包
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from sklearn.cluster import KMeans
from sklearn import metrics
import matplotlib.transforms as transforms


# 拐点法【肘部图确定】（计算簇内离差平方和）
def k_SSE(X, clusters, ax, ok_k=3, title="K-SSE"):
    # 选择连续的K种不同的值
    K = range(1, clusters + 1)
    # 构建空列表用于存储总的簇内离差平方和
    TSSE = []
    for k in K:
        # 用于存储各个簇内离差平方和
        SSE = []
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(X)
        # 返回簇标签
        labels = kmeans.labels_
        # 返回簇中心
        centers = kmeans.cluster_centers_
        # 计算各簇样本的离差平方和，并保存到列表中
        for label in set(labels):
            SSE.append(np.sum((X.loc[labels == label,] - centers[label, :]) ** 2))
        # 计算总的簇内离差平方和
        TSSE.append(np.sum(SSE))

    # 设置绘图风格
    # plt.style.use('ggplot')
    # 绘制K的个数与GSSE的关系
    ax.plot(K, TSSE, 'b*-')
    ax.plot([ok_k, ok_k], [0, max(TSSE)], color='red')
    plt.xlabel('clusters')
    plt.title(title, loc='left', ha='left', fontdict={'weight': 'bold', 'size': 16})
    plt.ylabel('SSE within clusters')
    # 显示图形
    # ax.show()


# 轮廓系数确定
def k_silhouette(X, clusters, ax, ok_k=3, title="k_silhouette"):
    K = range(2, clusters + 1)
    # 构建空列表，用于存储个中簇数下的轮廓系数
    S = []
    for k in K:
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(X)
        labels = kmeans.labels_
        # 调用字模块metrics中的silhouette_score函数，计算轮廓系数
        S.append(metrics.silhouette_score(X, labels, metric='euclidean'))
    # X为所需要计算的数据集、labels为计算标签、metric为度量距离，这里选择的是欧氏距离

    # 设置绘图风格
    # ax.style.use('ggplot')
    # 绘制K的个数与轮廓系数的关系
    ax.plot(K, S, 'b*-')
    ax.plot([ok_k, ok_k], [0, max(S)], color='red')
    plt.xlabel('clusters')
    plt.title(title, loc='left', ha='left', fontdict={'weight': 'bold', 'size': 16})
    plt.ylabel('Contour coefficient')
    # 显示图形
    # ax.show()


def confidence_ellipse(x, y, ax, n_std=2.0, facecolor='none', **kwargs):
    """
    绘制一个置信椭圆，表示数据的分布范围。
    x, y: 数据点的坐标。
    ax: matplotlib的轴对象。
    n_std: 置信范围倍数，2.0对应95%的置信度。
    facecolor: 椭圆的填充颜色。
    kwargs: 其他可选参数，用于定制椭圆。
    """
    # 计算数据的均值和协方差矩阵
    if x.size < 2:
        return
    if x.size != y.size:
        raise ValueError("x 和 y 必须具有相同的长度")
    cov = np.cov(x, y)
    mean_x = np.mean(x)
    mean_y = np.mean(y)

    # 特征值和特征向量
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]

    # 计算椭圆的宽度和高度
    width, height = 2 * n_std * np.sqrt(np.abs(eigvals))
    angle = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))

    # 创建椭圆对象
    ellipse = Ellipse(xy=(mean_x, mean_y), width=width, height=height, angle=angle, facecolor=facecolor, **kwargs)

    # 添加到ax对象
    ax.add_patch(ellipse)
    return ellipse


def confidence_ellipse1(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])

    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


if __name__ == '__main__':
    # 随机生成三组二元正态分布随机数
    np.random.seed(1234)
    mean1 = [0.5, 0.5]
    cov1 = [[0.3, 0], [0, 0.3]]
    x1, y1 = np.random.multivariate_normal(mean1, cov1, 1000).T

    mean2 = [0, 8]
    cov2 = [[1.5, 0], [0, 1]]
    x2, y2 = np.random.multivariate_normal(mean2, cov2, 1000).T

    mean3 = [8, 4]
    cov3 = [[1.5, 0], [0, 1]]
    x3, y3 = np.random.multivariate_normal(mean3, cov3, 1000).T

    # 绘制三组数据的散点图
    plt.scatter(x1, y1)
    plt.scatter(x2, y2)
    plt.scatter(x3, y3)
    # 显示图形
    plt.show()
    # 将三组数据集汇总到数据框中
    X = pd.DataFrame(np.concatenate([np.array([x1, y1]), np.array([x2, y2]), np.array([x3, y3])], axis=1).T)
    # 自定义函数的调用
    k_SSE(X, 15)
    # 自定义函数的调用
    k_silhouette(X, 15)
