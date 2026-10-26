# 算法学习：KNN、K-Means 与 Q-Learning

本仓库包含机器学习与强化学习经典算法的 Python 实现示例，适合初学者理解算法原理。

## 目录结构

```
algorithm/
└── sklearn/
    ├── scikit.py          # KNN 分类 + PCA 降维示例
    ├── kmeans.py          # K-Means 聚类（手写实现 + sklearn 对比）
    ├── qlearning.py       # Q-Learning 强化学习（迷宫寻路）
    ├── kmeans_result.png  # K-Means 聚类可视化结果
    ├── elbow_method.png   # 肘部法则图（辅助选择 K 值）
    ├── qlearning_policy.png  # Q-Learning 策略可视化
    └── qlearning_curve.png   # Q-Learning 学习曲线
```

## 算法介绍

### 1. KNN 分类 + PCA 降维 (`scikit.py`)

使用 scikit-learn 演示 K 近邻分类器（KNN）与主成分分析（PCA）降维的基本用法。

**核心流程：**
- 用 `make_blobs` 生成 300 个三维随机样本，分为 3 个类别
- 用 `KNeighborsClassifier(n_neighbors=3)` 训练 KNN 模型并预测新样本
- 用 PCA 将三维数据降至二维，再用散点图可视化

**涉及概念：** KNN、PCA、降维、可视化

---

### 2. K-Means 聚类 (`kmeans.py`)

包含 K-Means 算法的**手写实现**与 **sklearn 版本对比**，并提供可视化和肘部法则。

**手写实现核心步骤：**

```
1. 初始化：从数据中随机选 K 个点作为初始质心
2. 分配：将每个数据点分配给距离最近的质心（欧几里得距离）
3. 更新：将每个簇的质心更新为簇内所有点的均值
4. 重复步骤 2-3，直到质心不再移动（收敛）
```

**主要特性：**
- `KMeans` 类：手写实现，支持记录每次迭代的质心轨迹
- `compare_with_sklearn()`：对比手写实现与 sklearn 的 SSE（簇内平方和）
- `elbow_method()`：肘部法则，帮助选择最优 K 值
- 可视化聚类结果与质心移动轨迹

**运行输出：**

| 文件 | 内容 |
|------|------|
| `kmeans_result.png` | 聚类结果 + 质心移动轨迹 |
| `elbow_method.png` | 不同 K 值对应的 SSE 曲线 |

---

### 3. Q-Learning 强化学习 (`qlearning.py`)

经典的迷宫寻路问题，演示 Q-Learning 算法从零开始学习最优策略的过程。

**迷宫设定（5×5）：**

```
S . . # .
# # . # .
. . . . .
. # # # .
. . X . G
```

| 符号 | 含义 | 奖励 |
|------|------|------|
| S | 起点 | — |
| G | 终点 | +100 |
| X | 陷阱 | -50 |
| `#` | 墙（不可通行） | — |
| `.` | 空地 | -1（每步） |

**Q-Learning 更新公式：**

```
Q(s, a) ← Q(s, a) + α × [r + γ × max Q(s', a') - Q(s, a)]
```

| 参数 | 含义 |
|------|------|
| α（学习率） | 新信息的权重，默认 0.1 |
| γ（折扣因子） | 未来奖励的重要性，默认 0.95 |
| ε（探索率） | 随机探索的概率，默认 0.2 |

**ε-贪婪策略：**
- 以概率 ε 随机探索（防止陷入局部最优）
- 以概率 1-ε 选择当前最优动作（利用已学知识）

**运行输出：**

| 文件 | 内容 |
|------|------|
| `qlearning_policy.png` | Q 表热力图 + 最优策略箭头图 |
| `qlearning_curve.png` | 每回合奖励与步数的学习曲线 |

---

## 获取仓库与建议运行环境

```bash
git clone --branch liquanyan https://github.com/heisd/algorithm.git
cd algorithm
python -m venv .venv
```

创建后按本机系统激活虚拟环境。脚本使用 NumPy、Matplotlib 与 scikit-learn；没有锁定全部版本，第一次复现时应记录 Python 和各库版本。

## 环境依赖

```bash
pip install numpy matplotlib scikit-learn
```

## 运行方式

```bash
# KNN + PCA 示例
python sklearn/scikit.py

# K-Means 聚类
python sklearn/kmeans.py

# Q-Learning 强化学习
python sklearn/qlearning.py
```

## 可视化示例

### K-Means 聚类结果
![K-Means 聚类结果](sklearn/kmeans_result.png)

### 肘部法则
![肘部法则](sklearn/elbow_method.png)

### Q-Learning 学习策略
![Q-Learning 策略](sklearn/qlearning_policy.png)

### Q-Learning 学习曲线
![Q-Learning 学习曲线](sklearn/qlearning_curve.png)

## 如何比较与复现

- `scikit.py` 使用 300 个三维样本、3 个中心和固定生成种子，训练 3 邻居分类器，再用 PCA 展示二维投影。
- `kmeans.py` 同时包含手写实现与 sklearn 对比。改变 K、初始质心或收敛阈值时，应一起观察 SSE 与可视化，而非只看一次聚类图。
- `qlearning.py` 将离散迷宫状态与上下左右动作映射到 Q 表；训练曲线、策略图和一次行走结果反映的含义不同。

学习时可按“保持其他设置不变，只调整一个参数 → 重新运行 → 记录图与终端输出”的方式比较。随机过程的结果不应被当作普遍结论，某个运行中的最优路径也不等于已验证所有环境。

## 输出与故障排查

脚本中的图片保存路径相对于当前工作目录解析；建议从仓库根目录运行。运行可能同时打开 Matplotlib 窗口，关闭窗口后程序才继续执行后续步骤。

| 现象 | 检查方式 |
| --- | --- |
| 无法导入 sklearn | 安装的包名是 scikit-learn，并使用当前解释器安装 |
| 中文标题显示方框 | 当前字体不包含所需字形，配置本机可用字体 |
| 图表窗口不显示 | 检查图形环境与 Matplotlib 后端 |
| 多次运行有差异 | 对齐数据、初始化和随机种子 |
| 运行后找不到生成图 | 检查当前工作目录和具体保存文件名 |

完整原始中文教程继续保留在 [README_CN.md](./README_CN.md)。本仓库是教学示例，未提供生产训练流水线或针对外部数据集的效果承诺。
