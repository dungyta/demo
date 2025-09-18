import torch
import matplotlib.pyplot as plt
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import (
    MeshRenderer, MeshRasterizer, RasterizationSettings,
    HardPhongShader, FoVPerspectiveCameras, look_at_view_transform
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load mesh
mesh = load_objs_as_meshes(["data/cow.obj"], device=device)

#Camera setup
R, T = look_at_view_transform(dist=3, elev=10, azim=0)
cameras = FoVPerspectiveCameras(device=device, R=R, T=T)

#Rasterization settings
raster_settings = RasterizationSettings(
    image_size=256,
    blur_radius=0.0,
    faces_per_pixel=1,
)

#  Renderer
renderer = MeshRenderer(
    rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
    shader=HardPhongShader(device=device, lights=None)
)

#Render depth map
fragments = renderer.rasterizer(mesh)
depth_map = fragments.zbuf[0, ..., 0].cpu()  # Lấy depth pixel gần nhất

# Compute normal map
verts = mesh.verts_packed()           # (V,3)
faces = mesh.faces_packed()           # (F,3)

# Face normals
v0 = verts[faces[:, 0]]
v1 = verts[faces[:, 1]]
v2 = verts[faces[:, 2]]
face_normals = torch.nn.functional.normalize(torch.cross(v1 - v0, v2 - v0), dim=1)  # (F,3)

# Pixel-wise normals
pix_to_face = fragments.pix_to_face[0, ..., 0]      # (H,W)
bary_coords = fragments.bary_coords[0, ..., 0, :]   # (H,W,3)

# Initialize normal map
H, W = pix_to_face.shape
normal_map = torch.zeros((H, W, 3), device=device)

# Assign interpolated normals
valid = pix_to_face >= 0
f = pix_to_face[valid]
bc = bary_coords[valid]
vnorms = face_normals[f]   # (N_valid, 3)
normal_map[valid] = bc[:, 0:1]*vnorms[:,0:1] + bc[:,1:2]*vnorms[:,1:2] + bc[:,2:3]*vnorms[:,2:3]
# normalize
normal_map = torch.nn.functional.normalize(normal_map, dim=2).cpu().numpy()
# map [-1,1] -> [0,1] for visualization
normal_map_vis = (normal_map + 1)/2.0

#  Show images
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.title("Depth Map")
plt.imshow(depth_map, cmap="gray")
plt.axis("off")

plt.subplot(1,2,2)
plt.title("Normal Map")
plt.imshow(normal_map_vis)
plt.axis("off")
plt.show()
