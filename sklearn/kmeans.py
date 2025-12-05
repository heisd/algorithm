"""
K-Means 聚类算法 Demo
包含：手写实现 + sklearn 对比 + 可视化

Author: Claude
适用于学习理解 K-Means 算法原理
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans as SklearnKMeans
from sklearn.datasets import make_blobs

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class KMeans:
    """
    手写 K-Means 聚类算法
    
    参数:
        k: 簇的数量
        max_iters: 最大迭代次数
        tol: 收敛阈值（质心移动距离小于此值则停止）
    """
    
    def __init__(self, k=3, max_iters=100, tol=1e-4):
        self.k = k
        self.max_iters = max_iters
        self.tol = tol
        self.centroids = None
        self.labels = None
        self.history = []  # 记录每次迭代的质心位置
        
    def fit(self, X):
        """
        训练 K-Means 模型
        
        参数:
            X: 数据矩阵，shape (n_samples, n_features)
        """
        n_samples, n_features = X.shape
        
        # 步骤1: 随机初始化质心（从数据点中随机选择k个）
        random_indices = np.random.choice(n_samples, self.k, replace=False)
        self.centroids = X[random_indices].copy()
        self.history.append(self.centroids.copy())
        
        for iteration in range(self.max_iters):
            # 步骤2: 分配 - 将每个点分配给最近的质心
            self.labels = self._assign_clusters(X)
            
            # 步骤3: 更新 - 重新计算每个簇的质心
            new_centroids = self._update_centroids(X)
            
            # 记录历史
            self.history.append(new_centroids.copy())
            
            # 检查收敛：质心是否不再移动
            centroid_shift = np.sqrt(np.sum((new_centroids - self.centroids) ** 2))
            
            print(f"迭代 {iteration + 1}: 质心移动距离 = {centroid_shift:.6f}")
            
            if centroid_shift < self.tol:
                print(f"✓ 算法在第 {iteration + 1} 次迭代后收敛")
                break
                
            self.centroids = new_centroids
        else:
            print(f"达到最大迭代次数 {self.max_iters}")
            
        return self
    
    def _assign_clusters(self, X):
        """
        将每个数据点分配给最近的质心
        
        返回: 每个点的簇标签
        """
        # 计算每个点到所有质心的距离
        distances = self._compute_distances(X)
        # 选择距离最小的质心
        return np.argmin(distances, axis=1)
    
    def _compute_distances(self, X):
        """
        计算所有点到所有质心的欧几里得距离
        
        返回: 距离矩阵 shape (n_samples, k)
        """
        n_samples = X.shape[0]
        distances = np.zeros((n_samples, self.k))
        
        for i, centroid in enumerate(self.centroids):
            # 欧几里得距离: sqrt(sum((x - c)^2))
            distances[:, i] = np.sqrt(np.sum((X - centroid) ** 2, axis=1))
            
        return distances
    
    def _update_centroids(self, X):
        """
        更新质心位置：计算每个簇内所有点的均值
        
        返回: 新的质心位置
        """
        new_centroids = np.zeros((self.k, X.shape[1]))
        
        for i in range(self.k):
            # 获取属于簇 i 的所有点
            cluster_points = X[self.labels == i]
            
            if len(cluster_points) > 0:
                # 计算均值作为新质心
                new_centroids[i] = cluster_points.mean(axis=0)
            else:
                # 如果簇为空，保持原质心
                new_centroids[i] = self.centroids[i]
                
        return new_centroids
    
    def predict(self, X):
        """
        预测新数据点的簇标签
        """
        return self._assign_clusters(X)
    
    def inertia(self, X):
        """
        计算簇内平方和 (SSE/惯性)
        """
        total = 0
        for i in range(self.k):
            cluster_points = X[self.labels == i]
            if len(cluster_points) > 0:
                total += np.sum((cluster_points - self.centroids[i]) ** 2)
        return total


def visualize_kmeans(X, kmeans, title="K-Means 聚类结果"):
    """
    可视化聚类结果和质心移动轨迹
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    colors = ['#ef4444', '#3b82f6', '#22c55e', '#f59e0b', '#8b5cf6']
    
    # 左图：最终聚类结果
    ax1 = axes[0]
    for i in range(kmeans.k):
        mask = kmeans.labels == i
        ax1.scatter(X[mask, 0], X[mask, 1], 
                   c=colors[i], label=f'Cluster {i+1}', 
                   alpha=0.6, s=50)
    
    # 绘制质心
    ax1.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1],
               c='black', marker='X', s=200, edgecolors='white',
               linewidths=2, label='Centroids')
    
    ax1.set_title(f'{title}\nSSE = {kmeans.inertia(X):.2f}')
    ax1.set_xlabel('Feature 1')
    ax1.set_ylabel('Feature 2')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 右图：质心移动轨迹
    ax2 = axes[1]
    ax2.scatter(X[:, 0], X[:, 1], c='gray', alpha=0.3, s=30)
    
    # 绘制每个质心的移动轨迹
    history = np.array(kmeans.history)
    for i in range(kmeans.k):
        trajectory = history[:, i, :]
        ax2.plot(trajectory[:, 0], trajectory[:, 1], 
                'o-', color=colors[i], markersize=8, 
                linewidth=2, label=f'Centroid {i+1}')
        # 标记起点和终点
        ax2.scatter(trajectory[0, 0], trajectory[0, 1], 
                   color=colors[i], marker='s', s=100, edgecolors='black')
        ax2.scatter(trajectory[-1, 0], trajectory[-1, 1], 
                   color=colors[i], marker='X', s=150, edgecolors='black')
    
    ax2.set_title('Centroid Movement\n(Square=Start, X=End)')
    ax2.set_xlabel('Feature 1')
    ax2.set_ylabel('Feature 2')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def compare_with_sklearn(X, k=3):
    """
    对比手写实现和 sklearn 实现
    """
    print("=" * 50)
    print("手写 K-Means 实现")
    print("=" * 50)
    
    # 手写实现
    my_kmeans = KMeans(k=k, max_iters=100)
    my_kmeans.fit(X)
    
    print(f"\n最终质心:\n{my_kmeans.centroids}")
    print(f"SSE (惯性): {my_kmeans.inertia(X):.4f}")
    
    print("\n" + "=" * 50)
    print("sklearn K-Means 实现")
    print("=" * 50)
    
    # sklearn 实现
    sklearn_kmeans = SklearnKMeans(n_clusters=k, random_state=42, n_init=10)
    sklearn_kmeans.fit(X)
    
    print(f"\n最终质心:\n{sklearn_kmeans.cluster_centers_}")
    print(f"SSE (惯性): {sklearn_kmeans.inertia_:.4f}")
    
    return my_kmeans, sklearn_kmeans


def elbow_method(X, max_k=10):
    """
    肘部法则：帮助选择最佳 K 值
    """
    inertias = []
    K_range = range(1, max_k + 1)
    
    for k in K_range:
        kmeans = SklearnKMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
    ax.set_xlabel('K (Number of Clusters)')
    ax.set_ylabel('SSE (Inertia)')
    ax.set_title('Elbow Method\nFind the "elbow" point')
    ax.grid(True, alpha=0.3)
    
    # 标记可能的肘点
    ax.axvline(x=3, color='r', linestyle='--', alpha=0.5, label='Possible Elbow (K=3)')
    ax.legend()
    
    return fig


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    # 设置随机种子
    np.random.seed(42)
    
    # 生成测试数据：3个明显分离的簇
    print("生成测试数据...")
    X, y_true = make_blobs(
        n_samples=300,
        centers=3,
        cluster_std=1.0,
        random_state=42
    )
    
    print(f"数据形状: {X.shape}")
    print(f"样本数: {X.shape[0]}, 特征数: {X.shape[1]}")
    
    # 对比手写实现和 sklearn
    my_kmeans, sklearn_kmeans = compare_with_sklearn(X, k=3)
    
    # 可视化手写实现结果
    fig1 = visualize_kmeans(X, my_kmeans, "Manual K-Means Implementation")
    fig1.savefig('kmeans_result.png', dpi=150, bbox_inches='tight')
    print("\n聚类结果已保存到 kmeans_result.png")
    
    # 肘部法则
    fig2 = elbow_method(X)
    fig2.savefig('elbow_method.png', dpi=150, bbox_inches='tight')
    print("肘部法则图已保存到 elbow_method.png")
    
    # 展示核心代码逻辑
    print("\n" + "=" * 50)
    print("K-Means 核心算法伪代码")
    print("=" * 50)
    print("""
    1. 初始化: 随机选择 K 个数据点作为初始质心
    
    2. 重复直到收敛:
       
       a) 分配步骤 (E-step):
          for 每个数据点 x:
              计算 x 到所有质心的距离
              将 x 分配给最近的质心所属的簇
       
       b) 更新步骤 (M-step):
          for 每个簇 k:
              新质心 = 簇内所有点的均值
       
       c) 检查收敛:
          if 质心不再移动 (或移动很小):
              停止迭代
    
    3. 输出: 最终的簇分配和质心位置
    """)
    
    plt.show()