import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# 启用3D绘图
from mpl_toolkits.mplot3d import Axes3D

# --- 1. 数据准备 (两分类, 三个特征) ---
iris = load_iris()
# 选择类别 1 (Versicolor) 和 类别 2 (Virginica)
X_full = iris.data
y_full = iris.target
mask = (y_full == 1) | (y_full == 2)
X_binary = X_full[mask, :]
y_binary = y_full[mask]
# 将类别 2 重新标记为 1，类别 1 重新标记为 0
y_binary[y_binary == 1] = 0 
y_binary[y_binary == 2] = 1

# 选择三个特征: x1(萼片宽度), x2(花瓣长度), x3(花瓣宽度)
# 对应原始数据的索引 1, 2, 3
X = X_binary[:, [1, 2, 3]] 
feature_names = ['Sepal Width (x1)', 'Petal Length (x2)', 'Petal Width (x3)']

# 特征缩放
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- 2. 训练逻辑回归模型 ---
# 逻辑回归在3D空间中形成一个超平面 (Decision Hyperplane)
model = LogisticRegression(solver='liblinear', random_state=42)
model.fit(X_scaled, y_binary)

# 模型的系数 W 和 截距 B
coef = model.coef_[0]
intercept = model.intercept_[0]

# --- 3. 3D 可视化 ---
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
fig.suptitle('3D Decision Boundary: Binary Classification (Versicolor vs Virginica)', fontsize=14)

# 绘制数据点
ax.scatter(X_scaled[y_binary == 0, 0], X_scaled[y_binary == 0, 1], X_scaled[y_binary == 0, 2], 
           c='green', marker='o', label='Versicolor (Class 0)')
ax.scatter(X_scaled[y_binary == 1, 0], X_scaled[y_binary == 1, 1], X_scaled[y_binary == 1, 2], 
           c='blue', marker='^', label='Virginica (Class 1)')

# 定义超平面函数：W0*x + W1*y + W2*z + B = 0
# 解出 z: z = (-W0*x - W1*y - B) / W2
x_range = np.linspace(X_scaled[:, 0].min(), X_scaled[:, 0].max(), 10)
y_range = np.linspace(X_scaled[:, 1].min(), X_scaled[:, 1].max(), 10)
X_plane, Y_plane = np.meshgrid(x_range, y_range)

# 确保 W2 (petal width feature weight) 不为零，否则需要调整平面绘制方式
if coef[2] != 0:
    Z_plane = (-coef[0] * X_plane - coef[1] * Y_plane - intercept) / coef[2]
else:
    # 处理 W2=0 的特殊情况 (超平面垂直于z轴)
    # 这在实践中很少见，但为了代码健壮性
    print("Warning: W2 is close to zero, planar surface is near vertical.")
    Z_plane = np.full_like(X_plane, 0) # 简单地显示z=0平面，或抛出错误

# 绘制决策超平面
ax.plot_surface(X_plane, Y_plane, Z_plane, alpha=0.5, color='red', label='Decision Boundary')

# 设置标签
ax.set_xlabel(feature_names[0] + ' (Scaled)')
ax.set_ylabel(feature_names[1] + ' (Scaled)')
ax.set_zlabel(feature_names[2] + ' (Scaled)')
ax.legend()
plt.show()

# 触发一个示例图片来辅助理解