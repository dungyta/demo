import torch
import imageio
import numpy as np
from tqdm import tqdm
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import (
    FoVPerspectiveCameras, PointLights,
    RasterizationSettings, MeshRasterizer,
    HardPhongShader, MeshRenderer, TexturesVertex,
    look_at_view_transform
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_colored_cow(color1=[0,0,1], color2=[1,0,0]):
    # Load cow mesh
    mesh = load_objs_as_meshes(["data/cow_on_plane.obj"], device=device)
    verts = mesh.verts_packed()   # (V,3)
    z_vals = verts[:,2]

    # Lấy min và max theo trục z
    z_min, z_max = z_vals.min(), z_vals.max()

    # Tính alpha cho từng vertex
    alpha = (z_vals - z_min) / (z_max - z_min)

    # Tạo màu gradient
    color1 = torch.tensor(color1, device=device, dtype=torch.float32)
    color2 = torch.tensor(color2, device=device, dtype=torch.float32)
    colors = (1 - alpha).unsqueeze(1) * color1 + alpha.unsqueeze(1) * color2

    # Áp texture cho mesh
    mesh.textures = TexturesVertex(verts_features=colors.unsqueeze(0))
    return mesh

def get_renderer(image_size=512):
    lights = PointLights(device=device, location=[[2, 2, 2]])
    raster_settings = RasterizationSettings(image_size=image_size)
    rasterizer = MeshRasterizer(raster_settings=raster_settings)
    shader = HardPhongShader(device=device, lights=lights)
    renderer = MeshRenderer(rasterizer=rasterizer, shader=shader)
    return renderer

def render_cow(output_path="cow.gif", num_frames=36, image_size=256):
    mesh = load_colored_cow(color1=[0,0,1], color2=[1,0,0])  # xanh → đỏ
    renderer = get_renderer(image_size=image_size)
    images = []

    # Quay quanh con bò 360 độ
    angles = torch.linspace(0, 360, steps=num_frames)
    for azim in tqdm(angles):
        R, T = look_at_view_transform(dist=3, elev=10, azim=azim.item())
        cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
        image = renderer(mesh, cameras=cameras)
        img = image[0, ..., :3].cpu().numpy()
        img = (img * 255).astype(np.uint8)
        images.append(img)

    imageio.mimsave(output_path, images, duration=0.1)
    print(f"saved gif to: {output_path}")

render_cow()
