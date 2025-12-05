from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
# 降维算法头文件
from sklearn.decomposition import PCA
# 一般在大数据时代，使用降维算法来实现可视化的方式是很多的，因为二维更加直观
import numpy as np
# 生成数据集->随机数据集
# 在有的工作里面要使用别人给的训练集来训练
X, y = make_blobs(
    n_samples=300,      # 总样本数
    # 特征值数
    n_features=3,
    # 类别数量也是簇的数量     
    centers=3,          # 生成几个簇
    # 簇的标准差（越大越分散）
    cluster_std=1.0,    
    # 随机种子，保证可复现
    random_state=42     
)
# 使用knn的方式拟合
KNN = KNeighborsClassifier(n_neighbors=3)
KNN.fit(X,y)
# 创建新的数据集来预测数值
X_new = np.array([[0,0,0],[10,10,10]])
y_new = KNN.predict(X_new)
print("X_new.shape:", X_new.shape)
print("y_new.shape:", y_new.shape)
# 打印数据集的特征和数量
print("X.shape:", X.shape)
print("y.shape:", y.shape)

# 可视化数据集plt.scatter()是生成二维可视化,如果是高维的需要通过降维的方式来可视化
# 常见的降维工具有PCA,TSNE等
# 这里演示PCA降维
# 1.首先需要创建PCA模型
pca = PCA(n_components=2)
# 2.然后需要拟合模型
pca.fit(X)
# 3.然后需要转换数据
X_pca = pca.transform(X)
# 之后需要在二维的基础上可视化
plt.scatter(X_pca[:,0],X_pca[:,1],c=y,s=50,cmap='viridis')
plt.title('KNN')
# 保存数据
plt.show()

