import torch
import numpy as np
import imageio
from tqdm.auto import tqdm
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    PointLights,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    TexturesVertex,
    look_at_view_transform,
)
from pytorch3d.io import save_obj


def get_mesh_renderer(image_size=512, device="cpu"):
    raster_settings = RasterizationSettings(
        image_size=image_size,
        blur_radius=0.0,
        faces_per_pixel=1,
    )
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(raster_settings=raster_settings),
        shader=SoftPhongShader(device=device),
    )
    return renderer


def make_tetrahedron(device="cpu"):
    # 4 vertices
    verts = torch.tensor(
        [
            [0, 0, 0],                      # v0
            [1, 0, 0],                      # v1
            [0.5, np.sqrt(3)/2, 0],         # v2
            [0.5, np.sqrt(3)/6, np.sqrt(2/3)]  # v3
        ], dtype=torch.float32, device=device
    )

    # 4 faces (each face is a triangle)
    faces = torch.tensor(
        [
            [0, 1, 2],  # base
            [0, 1, 3],
            [1, 2, 3],
            [0, 2, 3],
        ], dtype=torch.int64, device=device
    )

    # màu từng vertex (R,G,B)
    verts_rgb = torch.tensor(
        [
            [1.0, 0.0, 0.0],  # red
            [0.0, 1.0, 0.0],  # green
            [0.0, 0.0, 1.0],  # blue
            [1.0, 1.0, 0.0],  # yellow
        ], dtype=torch.float32, device=device
    )[None]  # thêm batch dimension

    textures = TexturesVertex(verts_features=verts_rgb)

    tetra_mesh = Meshes(verts=[verts], faces=[faces], textures=textures)
    return tetra_mesh


def render_tetrahedron(image_size=256, num_frames=36, device="cpu"):
    mesh = make_tetrahedron(device)
    renderer = get_mesh_renderer(image_size=image_size, device=device)
    lights = PointLights(device=device, location=[[2.0, 2.0, -2.0]])

    renders = []
    for angle in tqdm(np.linspace(0, 360, num_frames)):
        R, T = look_at_view_transform(dist=2.5, elev=30, azim=angle)
        cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
        rend = renderer(mesh, cameras=cameras, lights=lights)
        rend = rend[0, ..., :3].cpu().numpy()
        renders.append((rend * 255).astype(np.uint8))

    imageio.mimsave("tetrahedron.gif", renders, duration=0.1)


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    render_tetrahedron(image_size=256, num_frames=36, device=device)
    print("Saved tetrahedron.gif")
