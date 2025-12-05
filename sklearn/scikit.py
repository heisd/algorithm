from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt


X, y = make_blobs(
    n_samples=300,      # 总样本数
    n_features=2,       # 特征维度
    centers=3,          # 生成几个簇
    cluster_std=1.0,    # 簇的标准差（越大越分散）
    random_state=42     # 随机种子，保证可复现
)
plt.scatter(X[:,0],X[:,1],c=y,s=50,cmap='viridis')
plt.title('KNN')
plt.show()

