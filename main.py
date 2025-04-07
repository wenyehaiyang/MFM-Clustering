import pandas as pd
from matplotlib import gridspec
import matplotlib.pyplot as plt
import numpy as np

from K均值聚类 import mfm_clustering_mean
from MyPCA import mfm_pca
from k均值聚类k值确定 import k_SSE, k_silhouette
from optics均值聚类 import mfm_clustering_optics
from 权重矩阵 import test_modal_weight, test_modal_weight_sse

# 中文和负号的正常显示
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['font.size'] = 15
plt.rcParams['axes.unicode_minus'] = False
# 设置边框粗细
plt.rcParams['axes.linewidth'] = 2
# 设置线条粗细
plt.rcParams['lines.linewidth'] = 2.5


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
    data = np.genfromtxt('采样点m.csv', delimiter=',', names=True)
    data1 = np.array([[j for j in list(i)[1:]] for i in data])
    row, col = data1.shape
    for r in range(row):
        for c in range(col):
            if np.isnan(data1[r][c]):
                data1[r][c] = 0
                data1[r][c] = np.nanmean(data1[:, c])
    ax3 = plt.subplot(gs[1, 0])
    k_SSE(pd.DataFrame(data1), 15, ax3, title='(c)', ok_k=4)
    data = np.genfromtxt('OPEs.csv', delimiter=',', names=True)
    data1 = np.array([[j for j in list(i)[1:]] for i in data])
    row, col = data1.shape
    for r in range(row):
        for c in range(col):
            if np.isnan(data1[r][c]):
                data1[r][c] = 0
                data1[r][c] = np.nanmean(data1[:, c])
    ax4 = plt.subplot(gs[2, 0])
    test_modal_weight(data1, ax4, test_limit=(1e-4, 1e2), method='optics', min_samples=2, plot_title='(d)')
    # k_silhouette(pd.DataFrame(data1), 15, ax4, title='(d)', ok_k=4)

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

def mfm_xnn2():
    # subplot分格显示
    plt.figure(figsize=(13, 11))

    # -----------------------------
    # MFM-clustering-mean
    # -----------------------------
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1])
    ax1 = plt.subplot(gs[0, 0])
    mfm_clustering_mean(ax1, '采样点m.csv', 4)

    # -----------------------------
    # MFM-clustering-optics
    # -----------------------------
    ax2 = plt.subplot(gs[0, 1])
    mfm_clustering_optics(ax2, 'OPEs1.csv', '(b)', [-8, 32], [-12, 10], legend=False)

    # PCA
    ax3 = plt.subplot(gs[1, 0])
    mfm_pca(ax3, '采样点m.csv', 3, title="(c)")

    # -----------------------------
    # MFM-clustering-optics放大图
    # -----------------------------
    ax4 = plt.subplot(gs[1, 1])
    # mfm_clustering_optics(ax5, 'OPEs1.csv', '', [-6, 1], [-2, 2.5], legend=False)
    # mfm_clustering_optics(ax5, 'OPEs1.csv', '', [-6, 1], [-1.5, 1.5], legend=False)
    mfm_clustering_optics(ax4, 'OPEs1.csv', '', [-6, 1], [-4.5, 4.5], legend=False)
    plt.subplots_adjust(left=0.08, right=0.98, bottom=0.06, top=0.97, wspace=0.19, hspace=0.22)
    plt.savefig("output_pca.png", dpi=300)
    plt.show()


def mfm_xnn_test():
    # 查看指标量
    plt.figure(figsize=(10, 6))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1])
    data = np.genfromtxt('OPEs.csv', delimiter=',', names=True)
    data1 = np.array([[j for j in list(i)[1:]] for i in data])
    row, col = data1.shape
    for r in range(row):
        for c in range(col):
            if np.isnan(data1[r][c]):
                data1[r][c] = 0
                data1[r][c] = np.nanmean(data1[:, c])
    ax3 = plt.subplot(gs[0, 0])
    k_silhouette(pd.DataFrame(data1), 11, ax3, title='(a)', ok_k=5)

    # 查看指标量
    data = np.genfromtxt('OPEs.csv', delimiter=',', names=True)
    data1 = np.array([[j for j in list(i)[1:]] for i in data])
    row, col = data1.shape
    for r in range(row):
        for c in range(col):
            if np.isnan(data1[r][c]):
                data1[r][c] = 0
                data1[r][c] = np.nanmean(data1[:, c])
    ax4 = plt.subplot(gs[0, 1])
    # test_modal_weight_sse(data1, ax4, test_limit=(1e-4, 1e1), cluster_k=2, plot_title='(d)')
    test_modal_weight(data1, ax4, test_limit=(1e-4, 1e4), method='optics', min_samples=2, cluster_k=2, plot_title='(b)')
    # k_silhouette(pd.DataFrame(data1), 11, ax4, title='(b)', ok_k=3)

    ## 采样点
    data = np.genfromtxt('采样点m.csv', delimiter=',', names=True)
    data1 = np.array([[j for j in list(i)[1:]] for i in data])
    row, col = data1.shape
    for r in range(row):
        for c in range(col):
            if np.isnan(data1[r][c]):
                data1[r][c] = 0
                data1[r][c] = np.nanmean(data1[:, c])
    ax3 = plt.subplot(gs[1, 0])
    k_SSE(pd.DataFrame(data1), 11, ax3, title='(c)', ok_k=4)

    # 查看指标量
    data = np.genfromtxt('采样点m.csv', delimiter=',', names=True)
    data1 = np.array([[j for j in list(i)[1:]] for i in data])
    row, col = data1.shape
    for r in range(row):
        for c in range(col):
            if np.isnan(data1[r][c]):
                data1[r][c] = 0
                data1[r][c] = np.nanmean(data1[:, c])
    ax4 = plt.subplot(gs[1, 1])
    k_silhouette(pd.DataFrame(data1), 11, ax4, title='(d)', ok_k=4)

    plt.subplots_adjust(left=0.09, right=0.95, bottom=0.10, top=0.95, wspace=0.25, hspace=0.43)
    plt.savefig("output_指标.png", dpi=300)
    plt.show()

def mfm_ref():
    # -----------------------------
    # MFM-clustering-optics参考数据绘图
    # -----------------------------
    plt.figure(figsize=(13, 6))
    gs = gridspec.GridSpec(1, 2, height_ratios=[1])
    ax1 = plt.subplot(gs[0, 0])
    mfm_clustering_mean(ax1, '参考数据.csv', 3, title="(a)")
    ax2 = plt.subplot(gs[0, 1])
    mfm_clustering_mean(ax2, '参考数据OPE2_1.csv', 2, title="(b)")
    # mfm_clustering_optics(ax2, '参考数据OPE2_1.csv', "(b)", [-3, 8], [-4.5, 4.5], min_samples=3)

    plt.subplots_adjust(left=0.08, right=0.95, bottom=0.1, top=0.95, wspace=0.18, hspace=0.33)
    plt.savefig("output_ref.png", dpi=300)
    plt.show()

    # 查看指标量
    plt.figure(figsize=(10, 6))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1,1])
    data = np.genfromtxt('参考数据OPE2.csv', delimiter=',', names=True)
    data1 = np.array([[j for j in list(i)[1:]] for i in data])
    row, col = data1.shape
    for r in range(row):
        for c in range(col):
            if np.isnan(data1[r][c]):
                data1[r][c] = 0
                data1[r][c] = np.nanmean(data1[:, c])
    ax3 = plt.subplot(gs[0, 0])
    k_SSE(pd.DataFrame(data1), 11, ax3, title='(a)', ok_k=2)

    # 查看指标量
    data = np.genfromtxt('参考数据OPE2.csv', delimiter=',', names=True)
    data1 = np.array([[j for j in list(i)[1:]] for i in data])
    row, col = data1.shape
    for r in range(row):
        for c in range(col):
            if np.isnan(data1[r][c]):
                data1[r][c] = 0
                data1[r][c] = np.nanmean(data1[:, c])
    ax4 = plt.subplot(gs[0, 1])
    # test_modal_weight_sse(data1, ax4, test_limit=(1e-4, 1e1), cluster_k=2, plot_title='(d)')
    test_modal_weight(data1, ax4, test_limit=(1e-3, 1e4), method='optics', min_samples=2, cluster_k=2, plot_title='(b)')
    # k_silhouette(pd.DataFrame(data1), 11, ax4, title='(b)', ok_k=3)

    ## 采样点
    data = np.genfromtxt('参考数据.csv', delimiter=',', names=True)
    data1 = np.array([[j for j in list(i)[1:]] for i in data])
    row, col = data1.shape
    for r in range(row):
        for c in range(col):
            if np.isnan(data1[r][c]):
                data1[r][c] = 0
                data1[r][c] = np.nanmean(data1[:, c])
    ax3 = plt.subplot(gs[1, 0])
    k_SSE(pd.DataFrame(data1), 11, ax3, title='(c)', ok_k=3)

    # 查看指标量
    data = np.genfromtxt('参考数据.csv', delimiter=',', names=True)
    data1 = np.array([[j for j in list(i)[1:]] for i in data])
    row, col = data1.shape
    for r in range(row):
        for c in range(col):
            if np.isnan(data1[r][c]):
                data1[r][c] = 0
                data1[r][c] = np.nanmean(data1[:, c])
    ax4 = plt.subplot(gs[1, 1])
    k_silhouette(pd.DataFrame(data1), 11, ax4, title='(d)', ok_k=3)

    plt.subplots_adjust(left=0.09, right=0.95, bottom=0.10, top=0.95, wspace=0.25, hspace=0.43)
    plt.savefig("output_ref1.png", dpi=300)
    plt.show()

def mfm_ref_gulf():
    # -----------------------------
    # MFM-clustering-optics在地中海西北部数据分析中得应用(Gulf of Lion (NW Mediterranean Sea))
    # -----------------------------
    # 第一步，查看基于采样点聚类和基于OPE种类聚类
    plt.figure(figsize=(13, 6))
    gs = gridspec.GridSpec(1, 2, height_ratios=[1])
    ax1 = plt.subplot(gs[0, 0])
    mfm_clustering_mean(ax1, '地中海西北部.csv', 3, title="(a)")
    # mfm_clustering_optics(ax1, '地中海西北部.csv', "(a)", [-5, 5.2], [-5, 5], min_samples=2)
    ax2 = plt.subplot(gs[0, 1])
    mfm_clustering_mean(ax2, '地中海西北部1_乘矩阵系数后.csv', 3, title="(b)")
    # mfm_clustering_optics(ax2, '参考数据OPE2_1.csv', "(b)", [-3, 8], [-4.5, 4.5], min_samples=3)

    plt.subplots_adjust(left=0.08, right=0.95, bottom=0.1, top=0.95, wspace=0.18, hspace=0.33)
    plt.savefig("地中海1.png", dpi=300)
    plt.show()

    # 查看指标量(基于OPE聚类SSE)
    plt.figure(figsize=(10, 6))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1,1])
    data = np.genfromtxt('地中海西北部1.csv', delimiter=',', names=True)
    data1 = np.array([[j for j in list(i)[1:]] for i in data])
    row, col = data1.shape
    for r in range(row):
        for c in range(col):
            if np.isnan(data1[r][c]):
                data1[r][c] = 0
                data1[r][c] = np.nanmean(data1[:, c])
    ax3 = plt.subplot(gs[0, 0])
    k_silhouette(pd.DataFrame(data1), 11, ax3, title='(a)', ok_k=3)

    # 查看指标量(测试以找到最佳权重矩阵的系数)
    data = np.genfromtxt('地中海西北部1.csv', delimiter=',', names=True)
    data1 = np.array([[j for j in list(i)[1:]] for i in data])
    row, col = data1.shape
    for r in range(row):
        for c in range(col):
            if np.isnan(data1[r][c]):
                data1[r][c] = 0
                data1[r][c] = np.nanmean(data1[:, c])
    ax4 = plt.subplot(gs[0, 1])
    # test_modal_weight_sse(data1, ax4, test_limit=(1e-4, 1e1), cluster_k=2, plot_title='(d)')
    test_modal_weight(data1, ax4, test_limit=(1e-3, 10**3.8), method='optics', min_samples=2, cluster_k=2, plot_title='(b)')
    # k_silhouette(pd.DataFrame(data1), 11, ax4, title='(b)', ok_k=3)

    # 指标量(基于采样点聚类SSE)
    data = np.genfromtxt('地中海西北部.csv', delimiter=',', names=True)
    data1 = np.array([[j for j in list(i)[1:]] for i in data])
    row, col = data1.shape
    for r in range(row):
        for c in range(col):
            if np.isnan(data1[r][c]):
                data1[r][c] = 0
                data1[r][c] = np.nanmean(data1[:, c])
    ax3 = plt.subplot(gs[1, 0])
    k_SSE(pd.DataFrame(data1), 11, ax3, title='(c)', ok_k=3)

    # 查看指标量(基于采样点聚类k_silhouette)
    data = np.genfromtxt('地中海西北部.csv', delimiter=',', names=True)
    data1 = np.array([[j for j in list(i)[1:]] for i in data])
    row, col = data1.shape
    for r in range(row):
        for c in range(col):
            if np.isnan(data1[r][c]):
                data1[r][c] = 0
                data1[r][c] = np.nanmean(data1[:, c])
    ax4 = plt.subplot(gs[1, 1])
    k_silhouette(pd.DataFrame(data1), 11, ax4, title='(d)', ok_k=3)

    plt.subplots_adjust(left=0.12, right=0.97, bottom=0.10, top=0.95, wspace=0.25, hspace=0.43)
    plt.savefig("地中海指标.png", dpi=300)
    plt.show()


# mfm_xnn()
# mfm_xnn2()
# mfm_xnn_test()

mfm_ref()

# mfm_ref_gulf()

# mfm_ref_taiwan()

