import torch
import numpy as np
import pytorch3d
import pytorch3d.io
from pytorch3d.renderer import (
    look_at_view_transform, 
    FoVPerspectiveCameras,
    TexturesVertex,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    PointLights
)
from pytorch3d.structures import Meshes
import imageio
import argparse

# Set device - QUAN TRỌNG: dùng device đã define
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def set_cam(dist=1.0, elev=0.0, azim=30, degrees=True):
    '''
    dist: khoang cach tu camera den object
    elev: goc cao so voi mat phang nag
    azim: goc quay quanh truc dung(xoay quanh object)
    '''
    R, T = look_at_view_transform(dist=dist, elev=elev, azim=azim, degrees=degrees)# dung look_at_view_transform de tao camera transform
    cameras = FoVPerspectiveCameras(R=R, T=T, device=device)  # Tra ve camera FOV prerspective(kieu camera 3d giong that)
    return cameras

def load_mesh(path:str):
    '''
    load .obj mesh -> tra ve vertices va face
    '''
    vertices, face, textures = pytorch3d.io.load_obj(path, device=device)
    faces = face.verts_idx# lay ra thong tin cac vertex de tao thanh tam giac
    vertices = vertices.unsqueeze(0)
    faces = faces.unsqueeze(0)
    
    # Ensure tensors on GPU, tu tao texture chu khong lay tu object
    texture_rgb = torch.ones_like(vertices, device=device)# text_rgb: tensor co kich thuoc [1,num_vertice,3] chua thong tin mau RGB cho tung dinh
    
    textures = TexturesVertex(texture_rgb)
    
    meshes = Meshes(verts=vertices, faces=faces, textures=textures)
    meshes = meshes.to(device)
    return meshes

def render_image(path:str, azim=30, dist=3.0, elev=0.0, image_size=512):
    # Lights on GPU, nhu mot cong tac bat den
    lights = PointLights(location=[[0, 0, -3]], device=device) #PointLights la 1 lop trong pytorch3d dai dien cho bong den, location dai dien cho vi tri dat bong den trong khong gian
    
    raster_settings = RasterizationSettings(image_size=image_size) # dinh nghia cac thiet lap cho qua trinh rasterizer
    rasterizer = MeshRasterizer(raster_settings=raster_settings)  #quet qua cac tam giac cua mo hinh 3d, tinh toan cac pixel bi che khuat hoac khong hien thi
    shader = SoftPhongShader(device=device) #shader quyet dinh cac hieu ung sang bong cho object 3d
    renderer = MeshRenderer(rasterizer=rasterizer, shader=shader)
    
    cameras = set_cam(dist=dist, azim=azim, elev=elev)
    meshes = load_mesh(path=path)
    
    image = renderer(meshes, cameras=cameras, lights=lights)
    return image

def render_360(path:str, num_frames=12, dist=3.0, elev=0.0):
    images = []
    angles = torch.linspace(0, 360, steps=num_frames, dtype=torch.float32)
    
    for a in angles:
        image = render_image(path=path, azim=a.item(), dist=dist, elev=elev)
        i = image[0, ..., :3].cpu().numpy()  # sau khi render image co shape(1,H,W,4)-> chuyen sang (H,W,3)(Bo kenh alpha)
        i = (i * 255).astype(np.uint8)
        images.append(i)

    duration = 1000 // 15
    imageio.mimsave('cow_gif.gif', images, duration=duration)
    print("GIF created successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rendering gif 360")
    parser.add_argument("--input", type=str, default="data/cow.obj", help="Path to object.obj")
    parser.add_argument("--output", type=str, default="mygif.gif", help="Output path for gif")
    parser.add_argument("--num_frames", type=int, default=12, help="Number of frames")
    parser.add_argument("--dist", type=float, default=3.0, help="Distance between object and camera")
    parser.add_argument("--elev", type=float, default=0.0, help="Elevation angle")
    args = parser.parse_args()
    
    render_360(args.input, args.num_frames, args.dist, args.elev)