import torch
import numpy as np
import pytorch3d
import matplotlib.pyplot as plt
import imageio
from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import (
    AlphaCompositor,
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
    FoVPerspectiveCameras,
    look_at_view_transform
)


device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def load_point_cloud(path="data/bridge_pointcloud.npz", stride=20):
    """
    Load point cloud từ file .npz
    Trả về Pointclouds object có tọa độ và màu.
    """
    point_cloud = np.load(path)

    points = torch.tensor(point_cloud["verts"][::stride], dtype=torch.float32, device=device)
    rgb = torch.tensor(point_cloud["rgb"][::stride], dtype=torch.float32, device=device)

    # batch dimension: (1, N, 3)
    points = points.unsqueeze(0)
    rgb = rgb.unsqueeze(0)

    return Pointclouds(points=points, features=rgb)


def get_points_renderer(image_size=256, radius=0.003, background_color=(1, 1, 1)):
    """
    Tạo renderer cho point cloud.
    """
    raster_settings = PointsRasterizationSettings(
        image_size=image_size,
        radius=radius,
    )
    renderer = PointsRenderer(
        rasterizer=PointsRasterizer(raster_settings=raster_settings),
        compositor=AlphaCompositor(background_color=background_color),
    )
    return renderer


def render_point_cloud(output_path="bridge.gif", num_frames=36, image_size=256):
    """
    Render point cloud bridge quay 360 độ và lưu thành gif.
    """
    point_clouds = load_point_cloud()
    renderer = get_points_renderer(image_size=image_size)

    images = []
    angles = torch.linspace(0, 360, steps=num_frames)

    for azim in angles:
        R, T = look_at_view_transform(dist=4, elev=10, azim=azim.item())
        cameras = FoVPerspectiveCameras(R=R, T=T, device=device)

        image = renderer(point_clouds, cameras=cameras)
        img = image[0, ..., :3].cpu().numpy()
        img = (img * 255).astype(np.uint8)
        images.append(img)

    # Lưu gif
    imageio.mimsave(output_path, images, duration=0.1)
    print(f"Saved gif to: {output_path}")
   
# def render_point_cloud(output_path="bridge.png", image_size=256):
#     """
#     Render point cloud bridge đứng yên và lưu thành PNG.
#     """
#     point_clouds = load_point_cloud()
#     renderer = get_points_renderer(image_size=image_size)

#     # Fix góc nhìn (camera đứng yên)
#     R, T = look_at_view_transform(dist=4, elev=10, azim=0)
#     cameras = FoVPerspectiveCameras(R=R, T=T, device=device)

#     # Render
#     image = renderer(point_clouds, cameras=cameras)
#     img = image[0, ..., :3].cpu().numpy()
#     img = (img * 255).astype(np.uint8)

#     # Lưu ảnh
#     imageio.imwrite(output_path, img)
#     print(f"Saved image to: {output_path}")


if __name__ == "__main__":
    render_point_cloud()
    print("data/bridge_pointcloud.npz")
