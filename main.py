import pandas as pd
from matplotlib import gridspec
import matplotlib.pyplot as plt
import numpy as np

from K均值聚类 import mfm_clustering_mean
from k均值聚类k值确定 import k_SSE, k_silhouette
from optics均值聚类 import mfm_clustering_optics

# 中文和负号的正常显示
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['font.size'] = 15
plt.rcParams['axes.unicode_minus'] = False
# 设置边框粗细
plt.rcParams['axes.linewidth'] = 1.5
# 设置线条粗细
plt.rcParams['lines.linewidth'] = 2


def mfm_xnn():
    # subplot分格显示
    plt.figure(figsize=(13, 11))

    # -----------------------------
    # MFM-clustering-mean
    # -----------------------------
    gs = gridspec.GridSpec(3, 2, height_ratios=[1, 0.5, 0.5])
    ax1 = plt.subplot(gs[0, 0])
    mfm_clustering_mean(ax1, '采样点m.csv', 4)

    # -----------------------------
    # MFM-clustering-optics
    # -----------------------------
    ax2 = plt.subplot(gs[0, 1])
    mfm_clustering_optics(ax2, 'OPEs1.csv', '(b)', [-8, 32], [-12, 10], legend=False)

    # 查看指标量
    data = np.genfromtxt('OPEs1.csv', delimiter=',', names=True)
    data1 = np.array([[j for j in list(i)[1:]] for i in data])
    row, col = data1.shape
    for r in range(row):
        for c in range(col):
            if np.isnan(data1[r][c]):
                data1[r][c] = 0
                data1[r][c] = np.nanmean(data1[:, c])
    ax3 = plt.subplot(gs[1, 0])
    k_SSE(pd.DataFrame(data1), 15, ax3, title='(c)', ok_k=4)
    ax4 = plt.subplot(gs[2, 0])
    k_silhouette(pd.DataFrame(data1), 15, ax4, title='(d)', ok_k=4)

    # -----------------------------
    # MFM-clustering-optics放大图
    # -----------------------------
    ax5 = plt.subplot(gs[1:, 1])
    # mfm_clustering_optics(ax5, 'OPEs1.csv', '', [-6, 1], [-2, 2.5], legend=False)
    # mfm_clustering_optics(ax5, 'OPEs1.csv', '', [-6, 1], [-1.5, 1.5], legend=False)
    mfm_clustering_optics(ax5, 'OPEs1.csv', '', [-6, 1], [-4.5, 4.5], legend=False)
    plt.subplots_adjust(left=0.08, right=0.98, bottom=0.06, top=0.97, wspace=0.18, hspace=0.33)
    plt.savefig("output.png", dpi=300)
    plt.show()


def mfm_ref():
    # -----------------------------
    # MFM-clustering-optics参考数据绘图
    # -----------------------------
    plt.figure(figsize=(13, 7))
    gs = gridspec.GridSpec(1, 2, height_ratios=[1])
    ax1 = plt.subplot(gs[0, 0])
    mfm_clustering_mean(ax1, '参考数据.csv', 3, title="(a)")
    ax2 = plt.subplot(gs[0, 1])
    mfm_clustering_mean(ax2, '参考数据OPE.csv', 2, title="(b)")

    plt.subplots_adjust(left=0.08, right=0.95, bottom=0.1, top=0.95, wspace=0.18, hspace=0.33)
    plt.savefig("output_ref.png", dpi=300)
    plt.show()

    # 查看指标量
    plt.figure(figsize=(10, 5))
    gs = gridspec.GridSpec(1, 2, height_ratios=[1])
    data = np.genfromtxt('参考数据.csv', delimiter=',', names=True)
    data1 = np.array([[j for j in list(i)[1:]] for i in data])
    row, col = data1.shape
    for r in range(row):
        for c in range(col):
            if np.isnan(data1[r][c]):
                data1[r][c] = 0
                data1[r][c] = np.nanmean(data1[:, c])
    ax3 = plt.subplot(gs[0, 0])
    k_SSE(pd.DataFrame(data1), 11, ax3, title='(c)', ok_k=3)
    ax4 = plt.subplot(gs[0, 1])
    k_silhouette(pd.DataFrame(data1), 11, ax4, title='(d)', ok_k=3)
    plt.show()


mfm_xnn()


# mfm_ref()
def user_confirm(tips="是否继续?[Y/N] ", _next: callable = lambda: ()) -> bool:
    cmd = input(tips)
    if cmd.lower() == 'y':
        _next()
        return True
    print("已取消")
    return False

