import torch
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.ops import sample_points_from_meshes
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # cần để plot 3D

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

# Load mesh
mesh = load_objs_as_meshes(["data/cow.obj"], device=device)

# Sample points từ mesh
points = sample_points_from_meshes(mesh, num_samples=10000)[0]  # (N,3)

# Normalize points về [0,1]
min_xyz = points.min(0).values
max_xyz = points.max(0).values
norm_points = (points - min_xyz) / (max_xyz - min_xyz + 1e-8)

# Chuyển sang voxel grid
R = 32
indices = (norm_points * (R-1)).long()
voxels = torch.zeros((R,R,R), dtype=torch.bool, device=device)
x, y, z = indices[:,0], indices[:,1], indices[:,2]
voxels[z, y, x] = True

# Convert về CPU để vẽ
voxels_np = voxels.cpu().numpy()

# Vẽ 3D voxel
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection='3d')

# Tạo lưới voxel
ax.voxels(voxels_np, facecolors='cyan', edgecolor='k')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.title("3D Voxel Cow")
plt.show()
