import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D
from skimage import measure

# ---------------- 1. 数据 ----------------
iris = load_iris()
mask = iris.target != 0                      # 去掉 Setosa
X_raw = iris.data[mask][:, [1, 2, 3]]        # SW/PL/PW
y_raw = (iris.target[mask] == 2).astype(int)  # Virginica=1

scaler = StandardScaler()
X = scaler.fit_transform(X_raw)

# ---------------- 2. 模型 ----------------
model = SVC(kernel='rbf', gamma=0.5, C=1.0,
            probability=True, random_state=42)
model.fit(X, y_raw)

# ---------------- 3. 3D 网格 ----------------
N = 60
margin = 0.5
lims = np.array([X.min(axis=0) - margin, X.max(axis=0) + margin])
x_rng = np.linspace(lims[0, 0], lims[1, 0], N)
y_rng = np.linspace(lims[0, 1], lims[1, 1], N)
z_rng = np.linspace(lims[0, 2], lims[1, 2], N)
X_mesh, Y_mesh, Z_mesh = np.meshgrid(x_rng, y_rng, z_rng, indexing='ij')
grid = np.c_[X_mesh.ravel(), Y_mesh.ravel(), Z_mesh.ravel()]
prob = model.predict_proba(grid)[:, 1].reshape(X_mesh.shape)

spacing = (x_rng[1]-x_rng[0], y_rng[1]-y_rng[0], z_rng[1]-z_rng[0])

# ---------------- 4. 等值面 ----------------
# 0.5 决策边界
verts_50, faces_50, _, _ = measure.marching_cubes(prob, 0.5, spacing=spacing)
verts_50 += [x_rng[0], y_rng[0], z_rng[0]]
# 0.75 高概率
try:
    verts_75, faces_75, _, _ = measure.marching_cubes(prob, 0.75, spacing=spacing)
    verts_75 += [x_rng[0], y_rng[0], z_rng[0]]
except RuntimeError:
    verts_75, faces_75 = None, None

# ---------------- 5. 画图 ----------------
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')
ax.view_init(elev=25, azim=135)

clevs = np.linspace(0, 1, 11)

# 5.1 数学正确投影：沿视线方向平均
# 底面（Z 方向平均）
P_xy = prob.mean(axis=2)
ax.contourf(X_mesh[:, :, 0], Y_mesh[:, :, 0], P_xy,
            zdir='z', offset=lims[0, 2], levels=clevs, cmap='coolwarm')
ax.contour(X_mesh[:, :, 0], Y_mesh[:, :, 0], P_xy,
           zdir='z', offset=lims[0, 2], levels=clevs, colors='k', linewidths=0.3)

# 后侧面（Y 方向平均）
P_xz = prob.mean(axis=1)
ax.contourf(X_mesh[:, 0, :], P_xz, Z_mesh[:, 0, :],
            zdir='y', offset=lims[1, 1], levels=clevs, cmap='coolwarm')
ax.contour(X_mesh[:, 0, :], P_xz, Z_mesh[:, 0, :],
           zdir='y', offset=lims[1, 1], levels=clevs, colors='k', linewidths=0.3)

# 左侧面（X 方向平均）
P_yz = prob.mean(axis=0)
ax.contourf(P_yz, Y_mesh[0, :, :], Z_mesh[0, :, :],
            zdir='x', offset=lims[0, 0], levels=clevs, cmap='coolwarm')
ax.contour(P_yz, Y_mesh[0, :, :], Z_mesh[0, :, :],
           zdir='x', offset=lims[0, 0], levels=clevs, colors='k', linewidths=0.3)

# 5.2 实体等值面
ax.plot_trisurf(verts_50[:, 0], verts_50[:, 1], faces_50, verts_50[:, 2],
                color='red', alpha=0.35, lw=0, label='P=0.5 Decision Boundary')
if faces_75 is not None:
    ax.plot_trisurf(verts_75[:, 0], verts_75[:, 1], faces_75, verts_75[:, 2],
                    color='blue', alpha=0.35, lw=0, label='P=0.75 High-Prob Region')

# 5.3 原始数据散点
colors = ['gold' if yy == 0 else 'darkgreen' for yy in y_raw]
ax.scatter(X[:, 0], X[:, 1], X[:, 2],
           c=colors, edgecolors='k', s=60, depthshade=False, label='Data')

# 5.4 装饰
ax.xaxis.pane.fill = True
ax.yaxis.pane.fill = True
ax.zaxis.pane.fill = True
for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
    axis.pane.set_facecolor((0.97, 0.97, 0.97, 0.4))
    axis._axinfo['grid'].update(color=(0.85, 0.85, 0.85, 0.6), linestyle='-')

ax.set_xlabel('Sepal Width (scaled)')
ax.set_ylabel('Petal Length (scaled)')
ax.set_zlabel('Petal Width (scaled)')
ax.set_title('Task4: Correct 3D Boundary + Probability Map (Mean Projection)')
ax.legend(loc='upper left')

plt.tight_layout()
plt.show()