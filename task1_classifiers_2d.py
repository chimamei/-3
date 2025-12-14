import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import HistGradientBoostingClassifier

# --- 1. 数据准备 ---
iris = load_iris()
# 仅使用后两个特征: 花瓣长度(x2) 和 花瓣宽度(x3)
X = iris.data[:, 2:]
y = iris.target
X = StandardScaler().fit_transform(X) # 特征缩放
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4, random_state=42)

# --- 2. 定义分类器 ---
classifiers = {
    "Logistic Regression (C=0.1)": LogisticRegression(C=0.1),
    "Logistic Regression (C=100)": LogisticRegression(C=100),
    "Gaussian Process": GaussianProcessClassifier(kernel=1.0 * RBF([1.0, 1.0]), random_state=42),
    "Gradient Boosting": HistGradientBoostingClassifier(random_state=42),
}

# --- 3. 可视化设置 ---
h = .02  # 网格步长
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# 定义颜色和颜色图
cmap_light = ListedColormap(['#FFC0CB', '#90EE90', '#ADD8E6']) # 粉, 浅绿, 浅蓝
cmap_bold = ListedColormap(['#FF0000', '#008000', '#0000FF']) # 红, 绿, 蓝 (对应类别0, 1, 2)

# --- 4. 绘图 ---
fig, axes = plt.subplots(len(classifiers), 4, figsize=(20, 5 * len(classifiers)))
fig.suptitle("Classifier Decision Boundaries and Class Probabilities (Petal Length/Width)", fontsize=16)

# 遍历每个分类器
for i, (name, clf) in enumerate(classifiers.items()):
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    
    # 预测整个网格
    Z_pred = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z_pred = Z_pred.reshape(xx.shape)
    
    Z_proba = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()]) # 类别概率

    # --- 第一列: 整体决策边界 ---
    ax_boundary = axes[i, 0]
    ax_boundary.contourf(xx, yy, Z_pred, cmap=cmap_light, alpha=0.8)
    # 绘制训练数据点
    ax_boundary.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cmap_bold, edgecolors='k', s=20)
    # 绘制测试数据点
    ax_boundary.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cmap_bold, edgecolors='k', alpha=0.6, marker='x', s=50)
    ax_boundary.set_title(f"{name} (Accuracy: {score:.2f})")
    ax_boundary.set_xlabel("Petal Length (Scaled)")
    ax_boundary.set_ylabel("Petal Width (Scaled)")

    # --- 剩下三列: 每一类的概率图 ---
    for j in range(3): # 类别 0, 1, 2
        ax_prob = axes[i, j + 1]
        
        # 绘制概率等高线图
        contour = ax_prob.contourf(xx, yy, Z_proba[:, j].reshape(xx.shape), 
                                   levels=np.linspace(0, 1, 11), cmap=plt.cm.Blues, alpha=0.7)
        fig.colorbar(contour, ax=ax_prob, label=f'P(Class {j})')
        
        # 绘制所有数据点
        scatter = ax_prob.scatter(X[:, 0], X[:, 1], c=iris.target, 
                                  cmap=cmap_bold, edgecolors='k', s=20, alpha=1)
        
        ax_prob.set_title(f"Class {j} Probability")
        ax_prob.set_xlabel("Petal Length (Scaled)")
        if j == 0:
             ax_prob.set_ylabel("Petal Width (Scaled)")


plt.tight_layout(rect=[0, 0.03, 1, 0.97])
plt.show()

# 触发一个示例图片来辅助理解