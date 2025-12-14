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
verts_50, faces_50, _, _ = measure.marching_cubes(prob, 0.5, spacing=spacing)
verts_50 += [x_rng[0], y_rng[0], z_rng[0]]

# ---------------- 5. 画图 ----------------
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.view_init(elev=25, azim=135)

# 5.1 0.5 等值面（散点云）
ax.scatter(verts_50[:, 0], verts_50[:, 1], verts_50[:, 2],
           c=verts_50[:, 2], cmap='coolwarm', s=1, lw=0)

# 5.2 数学正确投影：沿视线方向平均
clevs = np.linspace(0, 1, 11)

# 后侧面（Y 方向平均 → 贴在 Y=max）
P_xz = prob.mean(axis=1)          # (N, N)
X_xz, Z_xz = X_mesh[:, 0, :], Z_mesh[:, 0, :]
ax.contourf(X_xz, P_xz, Z_xz,
            zdir='y', offset=lims[1, 1], levels=clevs, cmap='coolwarm')
ax.contour(X_xz, P_xz, Z_xz,
           zdir='y', offset=lims[1, 1], levels=clevs, colors='k', linewidths=0.3)

# 左侧面（X 方向平均 → 贴在 X=min）
P_yz = prob.mean(axis=0)          # (N, N)
Y_yz, Z_yz = Y_mesh[0, :, :], Z_mesh[0, :, :]
ax.contourf(P_yz, Y_yz, Z_yz,
            zdir='x', offset=lims[0, 0], levels=clevs, cmap='coolwarm')
ax.contour(P_yz, Y_yz, Z_yz,
           zdir='x', offset=lims[0, 0], levels=clevs, colors='k', linewidths=0.3)

# 底面（Z 方向平均 → 贴在 Z=min）
P_xy = prob.mean(axis=2)          # (N, N)
X_xy, Y_xy = X_mesh[:, :, 0], Y_mesh[:, :, 0]
ax.contourf(X_xy, Y_xy, P_xy,
            zdir='z', offset=lims[0, 2], levels=clevs, cmap='coolwarm')
ax.contour(X_xy, Y_xy, P_xy,
           zdir='z', offset=lims[0, 2], levels=clevs, colors='k', linewidths=0.3)

# 5.3 坐标面格子
ax.xaxis.pane.fill = True
ax.yaxis.pane.fill = True
ax.zaxis.pane.fill = True
ax.xaxis.pane.set_facecolor((0.97, 0.97, 0.97, 0.4))
ax.yaxis.pane.set_facecolor((0.97, 0.97, 0.97, 0.4))
ax.zaxis.pane.set_facecolor((0.97, 0.97, 0.97, 0.4))
for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
    axis._axinfo['grid'].update(color=(0.85, 0.85, 0.85, 0.6), linestyle='-')

ax.set_xlabel('Sepal Width (scaled)')
ax.set_ylabel('Petal Length (scaled)')
ax.set_zlabel('Petal Width (scaled)')
ax.set_title('Mathematically Correct 3D Projections (Mean Along Ray)')

plt.tight_layout()
plt.show()